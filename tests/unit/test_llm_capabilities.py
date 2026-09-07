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

"""Tests for the per-model capability table."""

import pytest

from artemis.llm.capabilities import capabilities_for, clamp_reasoning_effort


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-6-astra"),
        ("openai", "gpt-5.6-terra"),
        ("openai", "gpt-5.6-sol"),
        ("openai", "gpt-5-nano"),
        ("openai", "o3"),
    ],
)
def test_openai_reasoning_models(provider, model):
    """The reasoning line takes an effort, refuses temperature, and must have
    its own reasoning replayed to it."""
    caps = capabilities_for(provider, model)
    assert caps.supports_reasoning is True
    assert caps.reasoning_param == "reasoning_effort"
    assert caps.supports_temperature is False
    assert caps.carries_reasoning_inline is True
    assert caps.slow_first_token is True


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
def test_openai_non_reasoning_models(model):
    """gpt-4o and friends want temperature and reject reasoning_effort."""
    caps = capabilities_for("openai", model)
    assert caps.supports_reasoning is False
    assert caps.supports_temperature is True
    assert clamp_reasoning_effort("high", caps) is None


def test_anthropic_split():
    """Claude 5 / Haiku 4.5 take a thinking budget and refuse temperature;
    the older models are the other way round."""
    new = capabilities_for("anthropic", "claude-opus-5")
    assert (new.supports_reasoning, new.supports_temperature) == (True, False)
    assert capabilities_for("anthropic", "claude-sonnet-5").supports_reasoning is True
    assert capabilities_for("anthropic", "claude-haiku-4-5-20251001").supports_reasoning is True

    old = capabilities_for("anthropic", "claude-3-5-sonnet-20241022")
    assert (old.supports_reasoning, old.supports_temperature) == (False, True)


def test_gemini_thinking_level_is_generation_gated():
    """Gemini 3+ takes thinking_level; 2.5 and older do not. This is the test
    that used to be an inline substring check in the router's Google branch."""
    new = capabilities_for("google", "gemini-3.8-flash")
    assert new.supports_reasoning is True
    assert new.reasoning_param == "thinking_level"

    for legacy in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"):
        assert capabilities_for("google", legacy).supports_reasoning is False

    # Every Gemini model stays eligible for the native google-genai engines.
    assert capabilities_for("google", "gemini-robotics-er-2-preview").is_gemini_family is True


def test_provider_aliases_and_vertex():
    """Aliases resolve, and Vertex serves the same Gemini models under the same
    thinking-level scale."""
    assert capabilities_for("gemini", "gemini-3.8-flash").reasoning_param == "thinking_level"
    assert capabilities_for("claude", "claude-opus-5").supports_reasoning is True
    assert capabilities_for("vertex", "gemini-3.8-flash").supports_reasoning is True
    assert capabilities_for("vertexai", "gemini-3.8-flash").is_gemini_family is True


def test_unknown_models_keep_the_pre_table_behaviour():
    """An unrecognized model must degrade to what it did before this table
    existed: temperature sent, no reasoning knob sent."""
    for provider, model in [
        ("openai", "some-future-model"),
        ("xai", "grok-4"),
        ("openrouter", "whatever/thing"),
        ("ollama", "llama3.2-vision"),
    ]:
        caps = capabilities_for(provider, model)
        assert caps.supports_reasoning is False
        assert caps.supports_temperature is True
        assert caps.is_gemini_family is False


def test_clamp_maps_intent_onto_the_supported_vocabulary():
    """A value the model does not spell is moved to the nearest rung, because
    the configured intent ("think less here") still applies."""
    anthropic = capabilities_for("anthropic", "claude-opus-5")
    assert clamp_reasoning_effort("medium", anthropic) == "medium"
    # Anthropic has no "minimal" rung; "low" is the nearest.
    assert clamp_reasoning_effort("minimal", anthropic) == "low"

    astra = capabilities_for("openai", "gpt-6-astra")
    assert clamp_reasoning_effort("low", astra) == "low"
    # The reasoning models reject "none" outright; "minimal" is the nearest.
    assert clamp_reasoning_effort("none", astra) == "minimal"

    assert clamp_reasoning_effort(None, astra) is None
    assert clamp_reasoning_effort("", astra) is None
    assert clamp_reasoning_effort("nonsense", astra) is None


def test_terse_memory_discipline_is_resolved_per_node():
    """The prompt gate follows the model that will actually read the prompt."""
    from types import SimpleNamespace

    from artemis.agents.prompt_assembly import terse_memory_discipline
    from artemis.config.llm import apply_model_override, get_default_llm_config

    base = get_default_llm_config()

    def ctx_for(provider, model):
        return SimpleNamespace(llm_config=apply_model_override(base, provider, model))

    assert terse_memory_discipline(ctx_for("openai", "gpt-6-astra")) is True
    assert terse_memory_discipline(ctx_for("anthropic", "claude-opus-5"), "planner") is True

    # Gemini is where the original wording was tuned; it keeps it.
    assert terse_memory_discipline(ctx_for("google", "gemini-3.8-flash")) is False
    # So does anything we have not deliberately calibrated.
    assert terse_memory_discipline(ctx_for("xai", "grok-4")) is False

    # A missing context or node must never raise inside prompt assembly.
    assert terse_memory_discipline(None) is False
    assert terse_memory_discipline(SimpleNamespace()) is False
    assert terse_memory_discipline(ctx_for("openai", "gpt-6-astra"), "no_such_node") is False


def test_planner_plan_budget_is_gated_the_same_way():
    """The plan is recited in full on every turn, so its length is a per-turn cost."""
    import json
    from pathlib import Path

    from jinja2 import Template

    planner_json = Path(__file__).parents[2] / "artemis" / "agents" / "planner" / "planner.json"
    blocks = json.loads(planner_json.read_text())["blocks"]["output_protocol_initial"]

    gemini = Template(blocks).render(plan_grammar="", terse_memory_discipline=False)
    terse = Template(blocks).render(plan_grammar="", terse_memory_discipline=True)

    # The answer-card instruction is what produced a 1.5 KB markdown ledger table.
    assert "answer card / scratchpad skeletons with Markdown headers" in gemini
    assert "answer card / scratchpad skeletons" not in terse
    assert "Do not pre-generate skeletons, headers or tables" in terse

    assert "Keep the plan short: 3-5 top-level milestones" in terse
    assert "Keep the plan short" not in gemini
