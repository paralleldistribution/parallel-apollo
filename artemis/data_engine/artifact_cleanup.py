"""Receipts for artifact deletions performed by one cleanup operation."""

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import stat


@dataclass
class ArtifactCleanupReport:
    """Record completed deletions, including when later cleanup raises.

    Callers validate the target's ownership and location before removing it.
    Missing targets are harmless, including targets removed concurrently.
    Failed deletion calls never produce a receipt; other errors propagate.
    """

    removed: dict[str, int] = field(default_factory=dict)

    @property
    def bytes_reclaimed(self) -> int:
        return sum(self.removed.values())

    def remove(self, path: Path, *, recursive: bool = True) -> None:
        """Delete one artifact; file-only callers retain unlink semantics."""
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        try:
            if recursive and stat.S_ISDIR(metadata.st_mode):
                size = 0
                for child in path.rglob("*"):
                    try:
                        child_metadata = child.lstat()
                        if not stat.S_ISDIR(child_metadata.st_mode):
                            size += child_metadata.st_size
                    except OSError:
                        continue
                shutil.rmtree(path)
            else:
                size = metadata.st_size
                path.unlink()
        except FileNotFoundError:
            # The target can disappear after lstat. Do not fail cleanup or
            # claim another actor's deletion. A missing descendant alone must
            # still report failure if rmtree left the target directory behind.
            try:
                path.lstat()
            except FileNotFoundError:
                return
            raise
        # A successful deletion call, never subsequent absence, authorizes
        # reporting this target. In particular, preserve an empty report when
        # another cleanup removes the path before our deletion can complete.
        self.removed[str(path)] = size
