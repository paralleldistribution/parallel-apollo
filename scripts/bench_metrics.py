#!/usr/bin/env python3
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

"""Score ARTEMIS run logs on the metrics that separate a fast run from a slow one.

Comparing providers used to mean counting log lines by hand, which is slow
enough that it only happens after something already looks wrong. This reads the
same logs and prints the table.

The metrics are chosen to be diagnostic rather than merely descriptive -- each
one points at a specific cause when it moves:

``notes/action``    how much the model writes per unit of progress. Note
                    arguments are re-sent on every following turn until they are
                    compressed, so this drives context growth directly.
``capsules``        segment compressions. Each one is an extra model call, and
                    they are triggered by transcript bulk, so this is the
                    downstream cost of the metric above.
``slow calls``      calls that took over ten seconds, i.e. the reasoning budget
                    and the request timeouts.
``max gap``         the largest stretch of silence, which is where a timed-out
                    and silently retried request hides.
``cached``          prompt-cache hit rate. A reasoning model that is not being
                    given its own reasoning back re-derives it every turn, and
                    this is the number that shows it.

Usage:
    scripts/bench_metrics.py <log>...
    scripts/bench_metrics.py benchmarks/apollo/logs/*.log
    scripts/bench_metrics.py --json run.log
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import sys

#: Leading ``2026-09-07 20:39:21,186`` on a log line. Lines without one are
#: continuations (rich wraps long records) and carry no independent timestamp.
_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\b")

#: The rich console format, ``[09/05/26 07:25:31]``, which appears mid-line in
#: logs captured straight off the CLI rather than through the apollo wrapper.
_TS_RICH = re.compile(r"\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]")

#: The CLI banner, ``▶ Claude Sonnet 5  (anthropic/claude-sonnet-5)``.
_BANNER = re.compile(
    r"\((google|vertexai|openai|anthropic|openrouter|xai|ollama|vllm|custom)/([\w.\-]+)\)"
)

#: ``profile={'provider': 'openai', 'model': 'gpt-6-astra'} mode='pro'`` from the
#: apollo wrapper, and the ARTEMIS-side ``provider=... model=...`` launch line.
_PROFILE = re.compile(r"'provider':\s*'([^']+)'.*?'model':\s*'([^']+)'")
_LAUNCH = re.compile(r"provider=(\S+)\s+model=(\S+)")
_MODE = re.compile(r"mode='?(\w+)'?")

#: Every agent announces its tool calls with its own wording; one pattern reads
#: all of them so a note written by the Planner counts the same as one written
#: by the Operator.
_TOOL = re.compile(
    r"(?:Operator requested tool|Planner called tool|Outputter executing tool):?\s+(\w+)"
)

#: Flash runs report every call the same way, device actions included, so their
#: tool names have to be classified rather than counted straight.
_FLASH_TOOL = re.compile(r"Executing Flash tool: (\w+)")
_ACTIONS = re.compile(r"Operator requested (\d+) action\(s\)")
_CAPSULE = re.compile(r"Chunk capsule \S+: source (\d+) chars -> capsule (\d+) chars")
_USAGE = re.compile(r"prompt_tokens=(\d+)\s*\(cached=(\d+)\)")
_RETURNED = re.compile(r"LLM call returned after ([\d.]+)s")
_EXIT = re.compile(r"ARTEMIS exited with (\d+) after ([\d.]+) min")

_NOTE_TOOLS = ("save_note", "update_note", "append_note")
_READ_TOOLS = ("read_note", "list_notes")

#: Flash tools that are not device actions: memory, perception, pacing and the
#: terminal status report.
_FLASH_NON_ACTIONS = frozenset(
    {
        *_NOTE_TOOLS,
        *_READ_TOOLS,
        "run_adb_command",
        "report_task_status",
        "ask_explorer",
        "search_history",
        "replay_steps",
        "get_step_screenshot",
    }
)


@dataclass
class RunMetrics:
    """One run, scored."""

    path: str
    provider: str = "?"
    model: str = "?"
    mode: str = "?"
    verdict: str = "?"
    wall_clock_s: float = 0.0
    action_turns: int = 0
    actions: int = 0
    note_writes: int = 0
    note_reads: int = 0
    adb_commands: int = 0
    slow_calls: int = 0
    slowest_call_s: float = 0.0
    max_gap_s: float = 0.0
    gaps_over_45s: int = 0
    capsules: int = 0
    capsule_source_chars: int = 0
    validator_runs: int = 0
    no_tool_call_turns: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    #: Recognized agent events. Zero means this file is not an agent run log --
    #: an MCP server log, say -- and its agent columns are unknown, not zero.
    agent_events: int = 0

    @property
    def scoreable(self) -> bool:
        return self.agent_events > 0

    @property
    def notes_per_action(self) -> float:
        return self.note_writes / self.action_turns if self.action_turns else 0.0

    @property
    def cached_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["notes_per_action"] = round(self.notes_per_action, 2)
        d["cached_ratio"] = round(self.cached_ratio, 3)
        return d


def _count_tool(m: RunMetrics, name: str) -> str:
    """Record one tool call against the right bucket."""
    m.agent_events += 1
    m.tool_calls[name] = m.tool_calls.get(name, 0) + 1
    if name in _NOTE_TOOLS:
        m.note_writes += 1
    elif name in _READ_TOOLS:
        m.note_reads += 1
    elif name == "run_adb_command":
        m.adb_commands += 1
    return name


def score(path: Path) -> RunMetrics:
    """Parse one log file into a :class:`RunMetrics`."""
    m = RunMetrics(path=str(path))
    stamps: list[datetime] = []

    for raw in path.read_text(errors="replace").splitlines():
        if ts := _TS.match(raw):
            stamps.append(
                datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S").replace(
                    microsecond=int(ts.group(2)) * 1000
                )
            )
        elif ts := _TS_RICH.search(raw):
            stamps.append(datetime.strptime(ts.group(1), "%m/%d/%y %H:%M:%S"))

        if m.provider == "?":
            # The apollo wrapper's profile dict, else ARTEMIS's own launch line.
            if hit := _PROFILE.search(raw):
                m.provider, m.model = hit.group(1), hit.group(2)
            elif "launching ARTEMIS" in raw and (hit := _LAUNCH.search(raw)):
                m.provider, m.model = hit.group(1), hit.group(2).rstrip(",")
            elif hit := _BANNER.search(raw):
                m.provider, m.model = hit.group(1), hit.group(2)
        if m.mode == "?" and (hit := _MODE.search(raw)):
            m.mode = hit.group(1)

        if hit := _TOOL.search(raw):
            _count_tool(m, hit.group(1))
        elif hit := _FLASH_TOOL.search(raw):
            if _count_tool(m, hit.group(1)) not in _FLASH_NON_ACTIONS:
                m.action_turns += 1
                m.actions += 1

        if hit := _ACTIONS.search(raw):
            m.agent_events += 1
            m.action_turns += 1
            m.actions += int(hit.group(1))
        if hit := _CAPSULE.search(raw):
            m.capsules += 1
            m.capsule_source_chars += int(hit.group(1))
        if hit := _USAGE.search(raw):
            m.prompt_tokens += int(hit.group(1))
            m.cached_tokens += int(hit.group(2))
        if hit := _RETURNED.search(raw):
            m.slowest_call_s = max(m.slowest_call_s, float(hit.group(1)))
        if hit := _EXIT.search(raw):
            m.verdict = "PASS" if hit.group(1) == "0" else f"EXIT{hit.group(1)}"
            m.wall_clock_s = float(hit.group(2)) * 60

        if "Waiting for LLM call response" in raw:
            m.slow_calls += 1
        if "Validator Agent" in raw and "Starting" not in raw:
            m.validator_runs += 1
        if "stopped without calling any tool" in raw:
            m.no_tool_call_turns += 1
        if "finalized the trace" in raw:
            for verdict in ("PASS", "FAIL"):
                if verdict in raw:
                    m.verdict = verdict

    stamps.sort()
    if stamps and not m.wall_clock_s:
        m.wall_clock_s = (stamps[-1] - stamps[0]).total_seconds()
    for a, b in zip(stamps, stamps[1:]):
        gap = (b - a).total_seconds()
        m.max_gap_s = max(m.max_gap_s, gap)
        if gap >= 45:
            m.gaps_over_45s += 1
    return m


_COLUMNS: tuple[tuple[str, int, str], ...] = (
    ("run", 26, "l"),
    ("model", 22, "l"),
    ("verdict", 7, "l"),
    ("wall", 7, "r"),
    ("acts", 5, "r"),
    ("notes", 6, "r"),
    ("n/act", 6, "r"),
    ("adb", 5, "r"),
    ("slow", 5, "r"),
    ("worst", 7, "r"),
    ("maxgap", 7, "r"),
    ("caps", 5, "r"),
    ("cached", 7, "r"),
)


def _row(m: RunMetrics) -> list[str]:
    head = [
        Path(m.path).stem[:26],
        f"{m.model}"[:22],
        m.verdict,
        f"{m.wall_clock_s / 60:.1f}m",
    ]
    if not m.scoreable:
        # No agent vocabulary in this file (an MCP server log, for instance).
        # Printing zeros here would read as "the run did nothing".
        return head + ["?"] * (len(_COLUMNS) - len(head))
    return head + [
        str(m.action_turns),
        str(m.note_writes),
        f"{m.notes_per_action:.1f}",
        str(m.adb_commands),
        str(m.slow_calls),
        f"{m.slowest_call_s:.0f}s" if m.slowest_call_s else "-",
        f"{m.max_gap_s:.0f}s",
        str(m.capsules),
        f"{m.cached_ratio:.0%}" if m.prompt_tokens else "-",
    ]


def render_table(runs: list[RunMetrics]) -> str:
    lines = [
        "  ".join(
            name.ljust(w) if align == "l" else name.rjust(w) for name, w, align in _COLUMNS
        ).rstrip(),
        "  ".join("-" * w for _, w, _ in _COLUMNS),
    ]
    for m in runs:
        cells = _row(m)
        lines.append(
            "  ".join(
                c.ljust(w) if align == "l" else c.rjust(w)
                for c, (_, w, align) in zip(cells, _COLUMNS)
            ).rstrip()
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="+", type=Path, help="ARTEMIS or apollo run logs")
    parser.add_argument("--json", action="store_true", help="emit raw metrics instead of a table")
    args = parser.parse_args(argv)

    runs = []
    for path in args.logs:
        if not path.is_file():
            print(f"skipping {path}: not a file", file=sys.stderr)
            continue
        runs.append(score(path))
    if not runs:
        return 1

    if args.json:
        print(json.dumps([m.as_dict() for m in runs], indent=2))
    else:
        print(render_table(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
