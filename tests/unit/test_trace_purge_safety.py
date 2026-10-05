"""Purge failures preserve files unless session inactivity was verified."""

import json
import os
import sqlite3
import time
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from artemis.config import settings
from artemis.data_engine.storage import StorageManager
from artemis.interfaces.cli.commands.trace import trace_app


@pytest.fixture
def run_artifacts(tmp_path):
    storage = StorageManager(tmp_path / "data_engine.db", tmp_path)
    sid = str(uuid4())
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id,status,pid,start_time) VALUES (?,?,?,?)",
            (sid, "running", os.getpid(), time.time()),
        )
        conn.commit()
    session = tmp_path / sid
    compiled = tmp_path / "apollo-review_PASS_stamp"
    paths = [
        session / "notes.txt",
        compiled / "recording.mp4",
        tmp_path / f"{sid}.result.json",
        tmp_path / f"{sid}.artemis.log",
    ]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"active evidence")
    return storage, sid, compiled, paths


@pytest.mark.parametrize("failure_stage", ["initialization", "liveness_read"])
def test_database_failure_cannot_bypass_active_session_guard(
    run_artifacts, monkeypatch, failure_stage
):
    storage, sid, compiled, paths = run_artifacts
    error = sqlite3.OperationalError("injected disk I/O failure")
    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
    error.sqlite_errorname = "SQLITE_IOERR"

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        StorageManager,
        "_init_db" if failure_stage == "initialization" else "_session_is_active",
        fail,
    )
    result = CliRunner().invoke(
        trace_app,
        [
            "purge",
            sid,
            "--path",
            str(storage.base_trace_dir),
            "--trace-dir",
            str(compiled),
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["cleanup_status"] == "failed"
    assert summary["removed"] == []
    assert summary["diagnostics"][0]["sqlite_errorname"] == "SQLITE_IOERR"
    assert all(path.read_bytes() == b"active evidence" for path in paths)


def test_missing_derived_database_is_rejected_without_creating_or_deleting_files(
    run_artifacts, tmp_path, monkeypatch
):
    storage, sid, _, _ = run_artifacts
    custom = tmp_path / "different-traces"
    compiled = custom / "apollo-review_PASS_stamp"
    session = custom / sid
    for directory in (session, compiled):
        directory.mkdir(parents=True)
        (directory / "evidence").write_bytes(b"keep")
    monkeypatch.setattr(settings, "DATA_ENGINE_DB_PATH", storage.db_path)
    result = CliRunner().invoke(
        trace_app, ["purge", sid, "--path", str(custom), "--trace-dir", str(compiled), "--json"]
    )
    assert result.exit_code == 2, result.output
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["status"] == "rejected"
    assert "database does not exist" in summary["error"]
    assert not (custom / "data_engine.db").exists()
    assert (session / "evidence").read_bytes() == (compiled / "evidence").read_bytes() == b"keep"
    with storage._get_connection() as conn:
        assert (
            conn.execute("SELECT status FROM sessions WHERE session_id=?", (sid,)).fetchone()[0]
            == "running"
        )


@pytest.mark.parametrize("json_output", [True, False])
def test_deprecated_image_pruning_warns_in_both_output_modes(run_artifacts, json_output):
    storage, sid, compiled, _ = run_artifacts
    with storage._get_connection() as conn:
        conn.execute("UPDATE sessions SET status='completed' WHERE session_id=?", (sid,))
        conn.commit()
    image = storage.base_trace_dir / "images" / "stale.jpg"
    image.parent.mkdir()
    image.write_bytes(b"shared screenshot")
    os.utime(image, (0, 0))
    args = [
        "purge",
        sid,
        "--path",
        str(storage.base_trace_dir),
        "--trace-dir",
        str(compiled),
        "--prune-images-older-than",
        "60",
    ]
    result = CliRunner().invoke(trace_app, args + (["--json"] if json_output else []))
    assert result.exit_code == 0, result.output
    assert (
        "shared image pruning deferred; use artemis trace maintenance while idle" in result.stdout
    )
    assert image.read_bytes() == b"shared screenshot"
