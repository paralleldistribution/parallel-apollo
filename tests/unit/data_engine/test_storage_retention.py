"""ENG-2219: purge stays local, atomic and safe during concurrent runs."""

from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from artemis.data_engine.engine import DataEngine
from artemis.data_engine.models import ImageRecord
from artemis.data_engine.retention import ActiveSessionError, database_lease, storage_diagnostics
from artemis.data_engine.storage import StorageManager
from artemis.interfaces.cli.commands.trace import trace_app


@pytest.fixture
def storage(tmp_path):
    return StorageManager(tmp_path / "data_engine.db", tmp_path)


def session(storage, *, status="completed", pid=None):
    sid = uuid4()
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id,status,pid,start_time) VALUES (?,?,?,?)",
            (str(sid), status, pid, time.time()),
        )
        conn.execute(
            "INSERT INTO history_chunks(chunk_id,session_id) VALUES (?,?)", (str(uuid4()), str(sid))
        )
        conn.commit()
    directory = storage.base_trace_dir / str(sid)
    directory.mkdir()
    (directory / "evidence.txt").write_text("evidence")
    return sid


def count(storage, table):
    with storage._get_connection() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_delete_removes_history_without_vacuum_and_preserves_other_session(storage):
    first, second = session(storage), session(storage)
    statements = []
    original = storage._get_connection

    @contextmanager
    def traced():
        with original() as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    storage._get_connection = traced
    storage.delete_session(first)
    assert count(storage, "sessions") == count(storage, "history_chunks") == 1
    assert not any("VACUUM" in statement.upper() for statement in statements)
    assert not (storage.base_trace_dir / str(first)).exists()
    assert (storage.base_trace_dir / str(second) / "evidence.txt").is_file()


def test_sql_failure_rolls_back_every_table_but_files_are_reclaimed(storage):
    sid = session(storage)
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO traces(trace_id,session_id) VALUES (?,?)", (str(uuid4()), str(sid))
        )
        conn.execute(
            "CREATE TRIGGER block_delete BEFORE DELETE ON traces BEGIN SELECT RAISE(ABORT,'injected storage failure'); END"
        )
        conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="injected storage failure"):
        storage.delete_session(sid)
    assert (
        count(storage, "sessions")
        == count(storage, "history_chunks")
        == count(storage, "traces")
        == 1
    )
    assert not (storage.base_trace_dir / str(sid)).exists()


def test_shared_image_survives_until_last_owner_is_deleted(storage):
    first, second = session(storage), session(storage)
    data = b"shared screenshot"
    name = hashlib.sha256(data).hexdigest()
    image = ImageRecord(image_name=name)
    storage.store_session_image(image, first, data)
    storage.store_session_image(image, second, data)
    path = storage.base_trace_dir / "images" / f"{name}.jpg"
    storage.delete_session(first)
    assert path.read_bytes() == data
    assert count(storage, "images") == 1
    assert count(storage, "session_images") == 1
    storage.delete_session(second)
    assert not path.exists()
    assert count(storage, "images") == count(storage, "session_images") == 0


def test_cache_hit_restores_missing_legacy_file_and_keeps_legacy_ownership(storage):
    sid = session(storage)
    data = b"missing legacy screenshot"
    name = hashlib.sha256(data).hexdigest()
    storage.create_image(ImageRecord(image_name=name))
    engine = object.__new__(DataEngine)
    engine.storage, engine.current_session_id = storage, sid
    engine.global_base_dir = storage.base_trace_dir
    assert engine.get_or_create_image(data) == name
    path = storage.base_trace_dir / "images" / f"{name}.jpg"
    assert path.read_bytes() == data
    storage.delete_session(sid)
    assert path.exists()  # an unmapped old session may still reference it


def test_active_session_cannot_be_purged(storage):
    sid = session(storage, status="running", pid=os.getpid())
    with pytest.raises(ActiveSessionError):
        storage.delete_session(sid)
    result = CliRunner().invoke(
        trace_app, ["purge", str(sid), "--path", str(storage.base_trace_dir), "--json"]
    )
    assert result.exit_code == 2
    assert (storage.base_trace_dir / str(sid) / "evidence.txt").exists()
    assert count(storage, "sessions") == 1


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_maintenance_refuses_active_lease_and_legacy_process(storage):
    with database_lease(storage.db_path):
        with pytest.raises(RuntimeError, match="in use"):
            storage.maintain()
    session(storage, status="running", pid=os.getpid())
    with pytest.raises(RuntimeError, match="active sessions"):
        storage.maintain()


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_idle_maintenance_removes_orphans_but_keeps_legacy_trace_references(storage):
    retained = session(storage)
    referenced, orphan = "a" * 64, "b" * 64
    for name in (referenced, orphan):
        storage.create_image(ImageRecord(image_name=name, timestamp=1))
    image_dir = storage.base_trace_dir / "images"
    image_dir.mkdir()
    for name in (referenced, orphan):
        (image_dir / f"{name}.jpg").write_bytes(b"image")
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO traces(trace_id,session_id,payload) VALUES (?,?,?)",
            (str(uuid4()), str(retained), json.dumps({"post_screenshot": referenced})),
        )
        conn.execute(
            "INSERT INTO history_chunks(chunk_id,session_id) VALUES (?,?)",
            (str(uuid4()), str(uuid4())),
        )
        conn.commit()
    result = storage.maintain(image_min_age_s=0)
    assert result["history_chunks_removed"] == 1
    assert result["images_removed"] == 1
    assert not result["compacted"]
    assert (image_dir / f"{referenced}.jpg").exists()
    assert not (image_dir / f"{orphan}.jpg").exists()
    assert count(storage, "history_chunks") == 1


def test_purge_init_failure_still_reclaims_own_files_and_reports_diagnostics(tmp_path, monkeypatch):
    sid = uuid4()
    directory = tmp_path / str(sid)
    directory.mkdir()
    (directory / "log").write_text("failed run")
    unrelated = tmp_path / "another-session"
    unrelated.mkdir()
    error = sqlite3.OperationalError("database or disk is full")
    error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_FULL, "SQLITE_FULL"

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr("artemis.data_engine.storage.StorageManager", broken)
    result = CliRunner().invoke(trace_app, ["purge", str(sid), "--path", str(tmp_path), "--json"])
    assert result.exit_code == 0
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["cleanup_status"] == "partial"
    assert summary["diagnostics"][0]["sqlite_errorname"] == "SQLITE_FULL"
    assert not directory.exists()
    assert unrelated.exists()


def test_purge_rejects_path_traversal_and_traces_root(storage):
    sid = session(storage)
    for arguments in (
        ["purge", "../outside", "--path", str(storage.base_trace_dir), "--json"],
        [
            "purge",
            str(sid),
            "--path",
            str(storage.base_trace_dir),
            "--trace-dir",
            str(storage.base_trace_dir),
            "--json",
        ],
    ):
        result = CliRunner().invoke(trace_app, arguments)
        assert result.exit_code == 2
    assert (storage.base_trace_dir / str(sid)).exists()


@pytest.mark.parametrize(
    "code,name,category",
    [
        (sqlite3.SQLITE_BUSY, "SQLITE_BUSY", "locked"),
        (sqlite3.SQLITE_IOERR, "SQLITE_IOERR", "disk_io"),
        (sqlite3.SQLITE_FULL, "SQLITE_FULL", "storage_full"),
    ],
)
def test_storage_diagnostics_distinguish_failures(storage, code, name, category):
    error = sqlite3.OperationalError("injected")
    error.sqlite_errorcode, error.sqlite_errorname = code, name
    result = storage_diagnostics(storage.db_path, error)
    assert result["category"] == category
    assert result["sqlite_errorname"] == name
    assert "free_bytes" in result and "page_count" in result and "max_page_count" in result


def _concurrent_worker(database, root, result_queue):
    try:
        storage = StorageManager(database, root)
        for _ in range(8):
            sid = session(storage)
            for _ in range(15):
                with storage._get_connection() as conn:
                    conn.execute(
                        "INSERT INTO traces(trace_id,session_id,payload) VALUES (?,?,?)",
                        (str(uuid4()), str(sid), "x" * 1024),
                    )
                    conn.commit()
            storage.delete_session(sid)
        result_queue.put(None)
    except Exception as exc:
        result_queue.put(repr(exc))


def test_concurrent_process_writes_and_purges_preserve_retained_session(storage):
    retained = session(storage)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_worker, args=(storage.db_path, storage.base_trace_dir, queue)
        )
        for _ in range(4)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [queue.get(timeout=45) for _ in processes]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert outcomes == [None] * 4
        assert count(storage, "sessions") == count(storage, "history_chunks") == 1
        assert count(storage, "traces") == 0
        assert (storage.base_trace_dir / str(retained) / "evidence.txt").exists()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        queue.close()


def test_full_diagnostic_reports_connection_page_limit(storage):
    with storage._get_connection() as conn:
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count={pages}")
        with pytest.raises(sqlite3.OperationalError) as failure:
            conn.execute(
                "INSERT INTO traces(trace_id,payload) VALUES (?,?)", (str(uuid4()), "x" * 1000000)
            )
        result = storage_diagnostics(storage.db_path, failure.value, connection=conn)
        assert result["category"] == "database_page_limit"
        assert result["sqlite_errorcode"] == sqlite3.SQLITE_FULL


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_compaction_refuses_low_disk_space(storage, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "artemis.data_engine.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=1)
    )
    with pytest.raises(RuntimeError, match="insufficient free space"):
        storage.maintain(compact=True)
