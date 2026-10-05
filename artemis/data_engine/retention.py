"""Database activity leases and actionable storage diagnostics.

POSIX maintenance takes an exclusive lease; DataEngine lifetimes and individual
storage operations take shared leases. This prevents a maintenance check/start
race even while an agent is between SQLite transactions.
"""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import sqlite3


class ActiveSessionError(RuntimeError):
    """Artifact deletion is forbidden while this session is still running."""


@contextmanager
def image_lease(database: Path):
    """Serialize image file writes/GC, including on Windows desktop clients."""
    if os.name != "nt":
        with database_lease(Path(str(database) + ".images"), exclusive=True):
            yield
        return
    import msvcrt
    import time

    with Path(str(database) + ".images.activity.lock").open("a+b") as stream:
        stream.write(b"\0")
        stream.flush()
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
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def database_lease(database: Path, *, exclusive: bool = False, blocking: bool = True):
    if os.name != "posix":
        if exclusive:
            raise RuntimeError("exclusive storage maintenance requires POSIX file locking")
        yield
        return
    import fcntl

    path = Path(str(database) + ".activity.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(stream, flags)
        except BlockingIOError as exc:
            raise RuntimeError(
                "database is in use; maintenance requires idle Artemis sessions"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


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
