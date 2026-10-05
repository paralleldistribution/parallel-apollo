"""Database activity leases and actionable storage diagnostics.

POSIX maintenance takes an exclusive lease; DataEngine lifetimes and individual
storage operations take shared leases. This prevents a maintenance check/start
race even while an agent is between SQLite transactions.
"""

from contextlib import ExitStack, contextmanager
import errno
import os
from pathlib import Path
import shutil
import sqlite3
import time


READ_ONLY_LEASE_TIMEOUT_SECONDS = 30.0


class ActiveSessionError(RuntimeError):
    """Artifact deletion is forbidden while this session is still running."""


@contextmanager
def image_lease(database: Path):
    """Serialize image file writes/GC, including on Windows desktop clients."""
    if os.name != "nt":
        path = Path(str(database) + ".images.activity.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with _path_lease(path, exclusive=True, blocking=True, create=True):
            yield
        return
    import msvcrt

    with Path(str(database) + ".images.activity.lock").open("a+b") as stream:
        deadline = time.monotonic() + 30
        while True:
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        try:
            # Byte-range locks may extend beyond EOF. Initialize only after
            # acquiring the lock so concurrent first users cannot append twice.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _path_lease(
    path: Path,
    *,
    exclusive: bool,
    blocking: bool,
    create: bool = False,
    timeout_s: float | None = None,
):
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT if create else os.O_RDONLY, 0o666)
    try:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + timeout_s if blocking and timeout_s is not None else None
        if not blocking or deadline is not None:
            flags |= fcntl.LOCK_NB
        while True:
            try:
                fcntl.flock(descriptor, flags)
                break
            except BlockingIOError as exc:
                if deadline is None:
                    raise RuntimeError(
                        "database is in use; maintenance requires idle Artemis sessions"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out after {timeout_s:g}s waiting for storage maintenance in "
                        f"{path}; retry trace inspection after maintenance finishes."
                    ) from exc
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def database_lease(
    database: Path,
    *,
    exclusive: bool = False,
    blocking: bool = True,
    read_only: bool = False,
):
    """Coordinate writers and read-only inspection with idle maintenance.

    Writers lease the sidecar, including before SQLite creates the database.
    Read-only inspection leases the existing parent directory instead, requiring
    no writable files or directories. Maintenance holds both locks, preventing
    either kind of operation from starting until maintenance has finished. The
    directory lock conservatively covers all database readers in that directory;
    locking the database inode itself would conflict with SQLite on macOS.
    Read-only operations wait at most 30 seconds for maintenance before failing.
    """
    if read_only and exclusive:
        raise ValueError("a read-only database lease cannot be exclusive")
    if os.name != "posix":
        if exclusive:
            raise RuntimeError("exclusive storage maintenance requires POSIX file locking")
        yield
        return

    if read_only:
        with _path_lease(
            database.resolve().parent,
            exclusive=False,
            blocking=blocking,
            timeout_s=READ_ONLY_LEASE_TIMEOUT_SECONDS,
        ):
            yield
        return

    path = Path(str(database) + ".activity.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as leases:
        leases.enter_context(_path_lease(path, exclusive=exclusive, blocking=blocking, create=True))
        if exclusive:
            leases.enter_context(
                _path_lease(database.resolve().parent, exclusive=True, blocking=blocking)
            )
        yield


def storage_diagnostics(
    database: Path, error: BaseException | None = None, *, connection=None
) -> dict:
    result = {"database_path": str(database)}
    if error is not None:
        code = getattr(error, "sqlite_errorcode", None)
        result.update(
            error_type=type(error).__name__,
            message=str(error),
            sqlite_errorcode=code,
            sqlite_errorname=getattr(error, "sqlite_errorname", None),
        )
        primary = code & 0xFF if isinstance(code, int) else None
        result["category"] = (
            "locked"
            if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            else "storage_full"
            if primary == sqlite3.SQLITE_FULL or getattr(error, "errno", None) == errno.ENOSPC
            else "disk_io"
            if primary == sqlite3.SQLITE_IOERR
            else "storage_error"
        )
    try:
        result["free_bytes"] = shutil.disk_usage(database.parent).free
    except OSError:
        pass
    result["files"] = {}
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database) + suffix)
        try:
            result["files"][path.name] = path.stat().st_size
        except OSError:
            pass
    if connection is not None or database.is_file():
        try:
            conn = connection or sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2
            )
            try:
                for name in ("page_size", "page_count", "max_page_count", "freelist_count"):
                    result[name] = conn.execute("PRAGMA " + name).fetchone()[0]
            finally:
                if connection is None:
                    conn.close()
        except sqlite3.Error:
            pass
    if result.get("category") == "storage_full":
        if result.get("page_count", 0) >= result.get("max_page_count", float("inf")):
            result["category"] = "database_page_limit"
        elif result.get("free_bytes", float("inf")) == 0:
            result["category"] = "disk_full"
    return result
