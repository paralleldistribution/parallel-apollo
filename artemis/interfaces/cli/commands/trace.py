# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace inspection and replay commands (artemis trace)."""

import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Annotated
from uuid import UUID

from artemis.config import settings
from artemis.utils.logger import get_logger
from rich.console import Console
from rich.table import Table
import typer

logger = get_logger(__name__)
trace_app = typer.Typer(help="Inspect, list, and query task execution traces.")


@trace_app.command("list")
def list_traces(
    traces_path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Custom traces directory path."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max sessions to list.")] = 20,
) -> None:
    """List recorded execution trace sessions."""
    base_dir = traces_path or settings.TRACES_PATH
    if not base_dir.exists():
        typer.secho(f"No traces found at: {base_dir}", fg=typer.colors.YELLOW)
        return

    sessions = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    console = Console()
    table = Table(title=f"Recorded Traces ({len(sessions)} total)")
    table.add_column("Session Name", style="cyan")
    table.add_column("Last Modified", style="green")
    table.add_column("Artifacts", style="white")

    for s in sessions[:limit]:
        artifacts = []
        if (s / "recording.mp4").exists() or (s / "recording.mkv").exists():
            artifacts.append("🎬 Video")
        if (s / "steps.json").exists():
            artifacts.append("📋 Steps")
        if (s / "notes").exists():
            artifacts.append("📝 Notes")
        mtime = time_str = time_to_str(s.stat().st_mtime)
        table.add_row(s.name, mtime, ", ".join(artifacts) or "Screenshots")

    console.print(table)


def time_to_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@trace_app.command("view")
def view_trace(
    session_name: Annotated[str, typer.Argument(help="Name of the trace session directory.")],
    traces_path: Annotated[Path | None, typer.Option("--path", "-p")] = None,
) -> None:
    """View step-by-step summary of a recorded trace session."""
    base_dir = traces_path or settings.TRACES_PATH
    session_dir = base_dir / session_name

    if not session_dir.exists():
        typer.secho(f"Session '{session_name}' not found in {base_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    steps_file = session_dir / "steps.json"
    console = Console()

    if steps_file.exists():
        try:
            steps_data = json.loads(steps_file.read_text(encoding="utf-8"))
            table = Table(title=f"Trace Steps: {session_name}")
            table.add_column("Step", justify="right", style="cyan")
            table.add_column("Action", style="green")
            table.add_column("Reasoning / Motivation", style="white")

            for step in steps_data:
                table.add_row(
                    str(step.get("step", "-")),
                    step.get("action", "-"),
                    step.get("motivation", step.get("thought", "-")),
                )
            console.print(table)
            return
        except Exception as e:
            logger.warning(f"Could not parse steps.json: {e}")

    # If steps.json not present, list files
    typer.secho(f"Session files in {session_dir}:", fg=typer.colors.CYAN)
    for f in session_dir.iterdir():
        typer.echo(f"  - {f.name}")


def _dir_size(path: Path) -> int:
    """Total bytes under a path (0 when it is missing)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _is_within(child: Path, parent: Path) -> bool:
    """Whether `child` resolves to something inside `parent`.

    Guards the explicit --trace-dir against deleting anything outside the traces
    tree, since it arrives from an untrusted argv.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


@trace_app.command("purge")
def purge_trace(
    session_id: Annotated[
        str,
        typer.Argument(help="Session UUID whose artifacts should be deleted."),
    ],
    trace_dir: Annotated[
        Path | None,
        typer.Option(
            "--trace-dir",
            help=(
                "Compiled trace directory to delete as well (e.g."
                " traces/<test-name>_PASS_<ts>). Needed because the compiled dir is"
                " only discoverable from the DB when a video row exists. Must be"
                " inside the traces directory."
            ),
        ),
    ] = None,
    prune_images_older_than: Annotated[
        float,
        typer.Option(
            "--prune-images-older-than",
            help=(
                "Also delete files in the shared traces/images cache last modified"
                " more than this many seconds ago. That cache is global and"
                " content-addressed, so it is pruned by age rather than by session:"
                " an age floor leaves a concurrently running job's working set"
                " alone. 0 disables image pruning."
            ),
        ),
    ] = 0.0,
    traces_path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Custom traces directory path."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable summary on stdout."),
    ] = False,
) -> None:
    """Delete every artifact of one finished session and report what was reclaimed.

    Intended for a caller that has already copied the evidence somewhere durable
    (parallel-wayfinder's `apollo` workflow uploads it to Supabase, then calls
    this). It removes the session's database rows, its session directory, its
    compiled trace directory including `recording.mp4`, and the per-session
    leftovers at the traces root.

    Exits non-zero only if nothing could be removed at all, so a caller can treat
    a partial purge as success and still reclaim the bulk of the space.
    """
    from artemis.data_engine.storage import StorageManager

    base_dir = traces_path or Path(settings.TRACES_PATH)
    reclaimed = 0
    removed: list[str] = []
    errors: list[str] = []

    if trace_dir is not None and not _is_within(trace_dir, base_dir):
        message = f"--trace-dir {trace_dir} is outside the traces directory {base_dir}"
        if as_json:
            typer.echo(json.dumps({"status": "rejected", "error": message}))
        else:
            typer.secho(message, fg=typer.colors.RED)
        raise typer.Exit(2)

    # Measure before deleting: the sizes are gone afterwards.
    session_dir = base_dir / session_id
    targets = [session_dir]
    if trace_dir is not None:
        targets.append(trace_dir)
    leftovers = [
        base_dir / f"{session_id}.result.json",
        base_dir / f"{session_id}.artemis.log",
    ]
    targets.extend(leftovers)
    planned = {str(t): _dir_size(t) for t in targets}

    # 1. Database rows + session directory + any video the DB knows about.
    try:
        storage = StorageManager(settings.DATA_ENGINE_DB_PATH, base_dir)
        storage.delete_session(UUID(session_id))
        removed.append("database rows")
    except ValueError:
        errors.append(f"{session_id!r} is not a valid UUID")
    except Exception as exc:  # noqa: BLE001 — reclaiming disk must not raise
        errors.append(f"database purge failed: {exc!r}")

    # 2. The compiled trace directory. delete_session only finds it when a video
    #    row exists, so remove the caller's resolved path explicitly too.
    if trace_dir is not None and trace_dir.exists():
        try:
            shutil.rmtree(trace_dir)
            removed.append(str(trace_dir))
        except OSError as exc:
            errors.append(f"could not delete {trace_dir}: {exc!r}")

    # 3. Per-session leftovers at the traces root.
    for leftover in leftovers:
        if leftover.exists():
            try:
                leftover.unlink()
                removed.append(str(leftover))
            except OSError as exc:
                errors.append(f"could not delete {leftover}: {exc!r}")

    reclaimed += sum(size for path, size in planned.items() if not Path(path).exists())

    # 4. The shared image cache, by age.
    images_removed = 0
    if prune_images_older_than > 0:
        images_dir = base_dir / "images"
        cutoff = time.time() - prune_images_older_than
        for image in images_dir.rglob("*") if images_dir.exists() else []:
            try:
                if image.is_file() and image.stat().st_mtime < cutoff:
                    size = image.stat().st_size
                    image.unlink()
                    images_removed += 1
                    reclaimed += size
            except OSError:
                continue
        if images_removed:
            removed.append(f"{images_removed} cached image(s)")

    summary = {
        "status": "purged" if removed else "nothing_removed",
        "session_id": session_id,
        "bytes_reclaimed": reclaimed,
        "removed": removed,
        "errors": errors,
    }
    if as_json:
        typer.echo(json.dumps(summary))
    else:
        typer.secho(
            f"Reclaimed {reclaimed / 1_048_576:.1f} MB for session {session_id}"
            f" ({len(removed)} item group(s))",
            fg=typer.colors.GREEN if removed else typer.colors.YELLOW,
        )
        for err in errors:
            typer.secho(f"  ! {err}", fg=typer.colors.YELLOW)

    if not removed:
        raise typer.Exit(1)
