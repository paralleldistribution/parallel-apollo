"""Maintenance coordination includes offline readers without writable sidecars."""

import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from artemis.data_engine import retention
from artemis.data_engine.storage import StorageManager


def test_windows_image_lock_does_not_grow_on_repeated_acquisition(tmp_path, monkeypatch):
    # Patch this module's platform adapter, not process-wide os.name: pathlib
    # must still create native paths on non-Windows regression-test hosts.
    monkeypatch.setattr(retention, "os", SimpleNamespace(name="nt", SEEK_END=os.SEEK_END))
    locking = Mock()
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=0)
    )
    database = tmp_path / "data_engine.db"
    lock_file = Path(str(database) + ".images.activity.lock")
    for _ in range(100):
        with retention.image_lease(database):
            assert lock_file.read_bytes() == b"\0"
    assert lock_file.stat().st_size == 1
    assert locking.call_count == 200


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_read_only_connection_prevents_maintenance_even_in_same_process(tmp_path):
    database = tmp_path / "data_engine.db"
    storage = StorageManager(database, tmp_path)
    reader = StorageManager(database, tmp_path, read_only=True)
    with reader._get_connection() as connection:
        connection.execute("SELECT COUNT(*) FROM sessions").fetchone()
        # The legacy open-file scan skips its own process, so only the lease
        # can reject compaction here, even between the reader's transactions.
        with pytest.raises(RuntimeError, match="database is in use"):
            storage.maintain(compact=True)
    assert storage.maintain(compact=True)["compacted"]


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_maintenance_conservatively_waits_for_other_database_readers_in_directory(tmp_path):
    storage = StorageManager(tmp_path / "first.db", tmp_path)
    StorageManager(tmp_path / "second.db", tmp_path)
    other_reader = StorageManager(tmp_path / "second.db", tmp_path, read_only=True)
    with other_reader._get_connection():
        with pytest.raises(RuntimeError, match="database is in use"):
            storage.maintain()
    assert storage.maintain()["history_chunks_removed"] == 0


@pytest.mark.skipif(os.name != "posix", reason="idle maintenance requires POSIX leases")
def test_reader_starting_after_maintenance_scan_waits_for_compaction(tmp_path, monkeypatch):
    database = tmp_path / "data_engine.db"
    storage = StorageManager(database, tmp_path)
    reader = StorageManager(database, tmp_path, read_only=True)
    scanned, finish_maintenance = threading.Event(), threading.Event()
    reader_started, reader_entered = threading.Event(), threading.Event()
    failures, results = [], []
    original_init = storage._init_db

    def pause_after_scan(connection=None):
        scanned.set()
        if not finish_maintenance.wait(5):
            raise RuntimeError("test did not release maintenance")
        original_init(connection=connection)

    monkeypatch.setattr(storage, "_init_db", pause_after_scan)

    def maintain():
        try:
            results.append(storage.maintain(compact=True))
        except Exception as exc:
            failures.append(exc)

    def read():
        try:
            reader_started.set()
            with reader._get_connection() as connection:
                reader_entered.set()
                assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        except Exception as exc:
            failures.append(exc)

    maintenance_thread = threading.Thread(target=maintain, daemon=True)
    reader_thread = threading.Thread(target=read, daemon=True)
    maintenance_thread.start()
    try:
        assert scanned.wait(5)
        reader_thread.start()
        assert reader_started.wait(5)
        assert not reader_entered.wait(0.1)
    finally:
        finish_maintenance.set()
        maintenance_thread.join(5)
        if reader_thread.ident is not None:
            reader_thread.join(5)
    assert not maintenance_thread.is_alive() and not reader_thread.is_alive()
    assert not failures
    assert results == [{"history_chunks_removed": 0, "images_removed": 0, "compacted": True}]
    assert reader_entered.is_set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX read-only directory regression")
def test_offline_reader_leases_legacy_database_without_creating_lock_file(tmp_path):
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE evidence(value TEXT)")
    connection.execute("INSERT INTO evidence VALUES ('retained')")
    connection.commit()
    connection.close()
    original = database.read_bytes()
    database.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        reader = StorageManager(database, tmp_path, read_only=True)
        with reader._get_connection() as connection:
            assert connection.execute("SELECT value FROM evidence").fetchone()[0] == "retained"
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM evidence")
        assert set(tmp_path.iterdir()) == {database}
        assert database.read_bytes() == original
    finally:
        tmp_path.chmod(0o755)
        database.chmod(0o644)
