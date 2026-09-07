# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from artemis.agents.operator.prompts import (
    apply_operator_prompt_contract,
    load_operator_prompts,
)


def test_operator_prompt_contract_matches_note_runtime_semantics():
    prompt = apply_operator_prompt_contract(load_operator_prompts()["main_template"])

    assert "**Write-Through Memory Tools** (`save_note`, `update_note`, `append_note`)" in prompt
    assert "May run alongside the turn's Turn-Ending Action (or Fast-Action Burst)" in prompt
    # The two execution tiers and the incident workflow are advertised.
    assert "**Vetted Single Action (default)**" in prompt
    # The burst ceiling is a context variable filled in by the later template render.
    assert (
        "**Fast-Action Burst (time-critical only)**: two to {{ max_burst_actions }}"
        " Turn-Ending Actions" in prompt
    )
    assert "--- Execution Incident (OPEN) ---" in prompt
    assert "Failure Analyzer" not in prompt
    assert (
        "(`read_note`, `list_notes`, `search_history`, `replay_steps`, `get_step_screenshot`)"
        in prompt
    )
    assert "recall_history" not in prompt
    assert "`save_note`, and `search_history`" not in prompt
    assert "Do NOT submit a Turn-Ending Action at the same time" not in prompt
    assert "memory note tools (`read_note`, `list_notes`, `save_note`, `update_note`" not in prompt


def test_operator_prompt_is_single_template_without_checker_dialogue():
    """The prompt is never switched by verification results: main_template is
    the only template, and no checker-dialogue machinery is advertised."""
    prompts = load_operator_prompts()
    assert set(prompts) == {"main_template"}
    prompt = apply_operator_prompt_contract(prompts["main_template"])
    assert "reply_to_checker" not in prompt
    assert "Checker" not in prompt


def test_operator_prompt_omits_environment_trust_and_explorer_directives():
    for template in load_operator_prompts().values():
        prompt = apply_operator_prompt_contract(template)
        assert "Untrusted Screen Content & Instruction Priority" not in prompt
        assert "Visual Explorer Rule" not in prompt


def test_operator_prompt_keeps_large_list_traversal_single_pass_until_boundary():
    for template in load_operator_prompts().values():
        prompt = apply_operator_prompt_contract(template)

        assert "do not advance to another milestone or mark it complete" in prompt
        assert "A single scroll that reveals no new content is not sufficient evidence" in prompt
        assert "one additional successful swipe in the same direction" in prompt
        assert "never reverse direction merely to prove completion" in prompt
        assert "even if older execution history has been pruned" in prompt
        assert "Terminate exploration when a definitive boundary is reached" not in prompt


def test_contract_strips_legacy_tool_literals():
    expected = apply_operator_prompt_contract(load_operator_prompts()["main_template"])

    assert "wait_for_delay(seconds=" not in expected
    assert 'press_key(key="home"' not in expected


def _render_operator_template(*, terse: bool) -> str:
    """The main_template as one provider's Operator actually receives it."""
    from jinja2 import Template

    from artemis.agents.operator.prompts import OPERATOR_MAX_TOOL_ITERATIONS

    template = apply_operator_prompt_contract(load_operator_prompts()["main_template"])
    return Template(template).render(
        initial_goal="goal",
        subgoals_status="",
        plan_and_history="",
        unified_history="",
        plan_grammar="",
        verification_active=False,
        checks_active=False,
        transcript_history=True,
        terse_memory_discipline=terse,
        max_burst_actions=4,
        max_tool_calls=OPERATOR_MAX_TOOL_ITERATIONS,
    )


def test_memory_discipline_is_bounded_for_models_that_over_apply_it():
    """OpenAI and Anthropic get bounded note wording; Gemini keeps its nudge.

    "Notes are zero-cost" is written against Gemini, which under-writes them. A
    model that follows it literally does the opposite: on the benchmark task
    gpt-6-astra wrote 27 notes to gemini-3.8-flash's 8, and the resulting
    transcript crossed the compression threshold three times.
    """
    gemini = _render_operator_template(terse=False)
    terse = _render_operator_template(terse=True)

    assert "zero-cost: saving more notes never burdens you" in gemini
    assert "zero-cost" not in terse
    assert "they are not free" in terse
    assert "no tables, no headers" in terse
    # Batching note writes onto the action turn is the difference between a note
    # costing nothing and a note costing a whole round trip.
    assert "same tool-call list as the Turn-Ending Action instead of spending a turn" in terse

    # The load-bearing half of the rule survives in both: a value that is gone
    # from the screen cannot be recovered.
    for rendered in (gemini, terse):
        assert "Will this action irreversibly alter the screen" in rendered
        assert "save it before exploring further" in rendered

    # The "clear key-value structures" advice is what becomes markdown tables.
    assert "organize the data in clear key-value structures" in gemini
    assert "organize the data in clear key-value structures" not in terse


def test_gemini_operator_prompt_is_unchanged_by_the_gate():
    """The gate must not perturb the tuned Gemini prompt by a single byte."""
    import json
    import subprocess

    from jinja2 import Template

    from artemis.agents.operator.prompts import OPERATOR_MAX_TOOL_ITERATIONS

    head = json.loads(
        subprocess.run(
            ["git", "show", "HEAD:artemis/agents/operator/operator.json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["main_template"]

    def render(template: str, **extra) -> str:
        return Template(apply_operator_prompt_contract(template)).render(
            initial_goal="goal",
            subgoals_status="",
            plan_and_history="",
            unified_history="",
            plan_grammar="",
            verification_active=False,
            checks_active=False,
            transcript_history=True,
            max_burst_actions=4,
            max_tool_calls=OPERATOR_MAX_TOOL_ITERATIONS,
            **extra,
        )

    assert render(head) == _render_operator_template(terse=False)
    # An unset flag is falsy, so any render path that forgets to pass it also
    # keeps the original wording rather than silently switching doctrine.
    assert render(load_operator_prompts()["main_template"]) == _render_operator_template(
        terse=False
    )
