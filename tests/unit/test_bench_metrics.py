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

"""Tests for the run-log scorer behind the provider benchmark."""

import importlib.util
from pathlib import Path
import sys

import pytest

# scripts/ is not a package, so the scorer is loaded by path. It has to be
# registered before execution: @dataclass resolves its annotations through
# sys.modules[cls.__module__].
_SPEC = importlib.util.spec_from_file_location(
    "bench_metrics", Path(__file__).parents[2] / "scripts" / "bench_metrics.py"
)
bench_metrics = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_metrics"] = bench_metrics
_SPEC.loader.exec_module(bench_metrics)


PRO_LOG = """\
2026-09-07 20:25:31,159 INFO apollo start: device_serial=RF8M31JW88V profile={'provider': 'openai', 'model': 'gpt-6-astra'} mode='pro' prompt='/goal do a thing'
2026-09-07 20:25:32,000 INFO artemis | i Planner called tool: save_note with args: {'key': 'task_plan'}
2026-09-07 20:25:33,000 INFO artemis | i Operator requested tool: update_note
2026-09-07 20:25:34,000 INFO artemis | i Operator requested tool: run_adb_command
2026-09-07 20:25:35,000 INFO artemis | i Operator requested tool: read_note
2026-09-07 20:25:36,000 INFO artemis | i Operator requested 2 action(s). Translating and processing all.
2026-09-07 20:25:37,000 INFO artemis | i LLM usage: prompt_tokens=1000 (cached=250), completion_tokens=10
2026-09-07 20:25:38,000 INFO artemis | i Waiting for LLM call response... (operator openai:gpt-6-astra)
2026-09-07 20:27:00,000 INFO artemis | i LLM call returned after 92.4s (operator openai:gpt-6-astra)
2026-09-07 20:27:01,000 INFO artemis | i Chunk capsule chunk:1-3: source 8000 chars -> capsule 800 chars (10%).
2026-09-07 20:27:02,000 INFO artemis | v Validator Agent
2026-09-07 20:27:03,000 INFO artemis | ! LLM stopped without calling any tool. Encouraging action.
2026-09-07 20:27:04,000 INFO ARTEMIS exited with 0 after 1.6 min
"""

FLASH_LOG = """\
==============================================================================
> Claude Sonnet 5  (anthropic/claude-sonnet-5)
==============================================================================
[09/05/26 07:29:37] INFO  i Executing Flash tool: manage_app({'action': 'launch'})
[09/05/26 07:29:45] INFO  i Executing Flash tool: click({'target': [940, 52]})
[09/05/26 07:29:50] INFO  i Executing Flash tool: save_note({'key': 'x'})
[09/05/26 07:29:55] INFO  i Executing Flash tool: report_task_status({'status': 'done'})
"""

MCP_ONLY_LOG = """\
2026-09-05 08:59:52,865 INFO mcp.server.lowlevel.server: Processing request of type CallToolRequest
2026-09-05 09:00:03,980 INFO mcp.server.lowlevel.server: Processing request of type CallToolRequest
"""


@pytest.fixture
def write_log(tmp_path):
    def _write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body)
        return path

    return _write


def test_scores_a_pro_run(write_log):
    m = bench_metrics.score(write_log("run.log", PRO_LOG))

    assert (m.provider, m.model, m.mode) == ("openai", "gpt-6-astra", "pro")
    assert m.verdict == "PASS"
    assert m.wall_clock_s == pytest.approx(96.0)

    # A note written by the Planner counts the same as one written by the
    # Operator: both land in the transcript and both are re-sent.
    assert m.note_writes == 2
    assert m.note_reads == 1
    assert m.adb_commands == 1
    # One turn, two actions in a burst.
    assert (m.action_turns, m.actions) == (1, 2)
    assert m.notes_per_action == pytest.approx(2.0)

    assert m.capsules == 1
    assert m.capsule_source_chars == 8000
    assert m.slow_calls == 1
    assert m.slowest_call_s == pytest.approx(92.4)
    assert m.validator_runs == 1
    assert m.no_tool_call_turns == 1
    assert m.cached_ratio == pytest.approx(0.25)

    # The 82s stretch around the slow call is the one that matters.
    assert m.max_gap_s == pytest.approx(82.0)
    assert m.gaps_over_45s == 1


def test_scores_a_flash_run_from_its_own_vocabulary(write_log):
    """Flash reports device actions and memory writes through one tool line."""
    m = bench_metrics.score(write_log("flash.log", FLASH_LOG))

    assert (m.provider, m.model) == ("anthropic", "claude-sonnet-5")
    assert m.wall_clock_s == pytest.approx(18.0)
    # manage_app and click are actions; save_note and report_task_status are not.
    assert m.action_turns == 2
    assert m.note_writes == 1


def test_a_log_with_no_agent_lines_reports_unknown_not_zero(write_log):
    """An MCP server log says nothing about how the agent behaved.

    Scoring it as all-zeros would read as "the run did nothing", which is a
    worse answer than "this file cannot answer that".
    """
    m = bench_metrics.score(write_log("mcp.log", MCP_ONLY_LOG))

    assert m.scoreable is False
    assert m.wall_clock_s > 0  # the timestamps are still real
    row = bench_metrics._row(m)
    assert row[-1] == "?"
    assert "?" in bench_metrics.render_table([m])

    assert bench_metrics.score(write_log("pro.log", PRO_LOG)).scoreable is True
