"""Deletion receipts cannot be inferred from disappearing paths."""

from pathlib import Path
import shutil

import pytest

from artemis.data_engine.artifact_cleanup import ArtifactCleanupReport


@pytest.mark.parametrize("directory", [False, True])
def test_competing_removal_after_measurement_does_not_create_receipt(
    tmp_path, monkeypatch, directory
):
    report = ArtifactCleanupReport()
    owned = tmp_path / "owned"
    owned.write_bytes(b"removed by this operation")
    report.remove(owned)
    target = tmp_path / "competing"
    if directory:
        target.mkdir()
        (target / "evidence").write_bytes(b"removed by another cleanup")
        original = shutil.rmtree
    else:
        target.write_bytes(b"removed by another cleanup")
        original = Path.unlink

    def other_cleanup_wins(path):
        original(path)
        original(path)  # This operation loses the race and must not get credit.

    monkeypatch.setattr(
        shutil if directory else Path, "rmtree" if directory else "unlink", other_cleanup_wins
    )
    report.remove(target)
    assert not target.exists()
    assert report.removed == {str(owned): len(b"removed by this operation")}
    assert report.bytes_reclaimed == len(b"removed by this operation")


def test_missing_descendant_does_not_hide_an_unremoved_directory(tmp_path, monkeypatch):
    report = ArtifactCleanupReport()
    target = tmp_path / "remaining"
    target.mkdir()
    (target / "evidence").write_bytes(b"keep")

    def missing_child(path):
        raise FileNotFoundError("a descendant disappeared; root remains")

    monkeypatch.setattr(shutil, "rmtree", missing_child)
    with pytest.raises(FileNotFoundError):
        report.remove(target)
    assert (target / "evidence").read_bytes() == b"keep"
    assert report.removed == {}


@pytest.mark.parametrize("directory", [False, True])
def test_permission_failure_is_still_reported(tmp_path, monkeypatch, directory):
    report = ArtifactCleanupReport()
    target = tmp_path / "denied"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"keep")

    def denied(path):
        raise PermissionError("injected permission failure")

    monkeypatch.setattr(shutil if directory else Path, "rmtree" if directory else "unlink", denied)
    with pytest.raises(PermissionError, match="injected permission failure"):
        report.remove(target)
    assert target.exists()
    assert report.removed == {}


def test_file_only_removal_cannot_delete_an_unexpected_directory(tmp_path):
    report = ArtifactCleanupReport()
    target = tmp_path / "image.jpg"
    target.mkdir()
    (target / "keep").write_bytes(b"unrelated contents")
    with pytest.raises(OSError):
        report.remove(target, recursive=False)
    assert (target / "keep").read_bytes() == b"unrelated contents"
    assert report.removed == {}
