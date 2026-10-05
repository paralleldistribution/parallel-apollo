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
    with pytest.raises(FileNotFoundError):
        report.remove(target)
    assert not target.exists()
    assert report.removed == {str(owned): len(b"removed by this operation")}
    assert report.bytes_reclaimed == len(b"removed by this operation")
