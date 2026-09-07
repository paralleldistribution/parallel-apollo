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

"""Per-model capability lookup.

Every knob ARTEMIS wants to send is accepted by some models and rejected by
others, and the split is per-MODEL, not per-provider: ``gpt-6-astra`` refuses
``temperature`` while ``gpt-4o`` requires it, ``gpt-4o-mini`` answers
``reasoning_effort`` with ``400 Unrecognized request argument`` while
``gpt-6-astra`` needs it to be told how hard to think. Before this table those
distinctions lived as substring tests scattered across the router, the config
merge and two agent engines, which is why non-Google runs ended up with no
reasoning budget at all: the only safe blanket rule was "send nothing".

The table is consulted by:

* :func:`artemis.config.llm._node_override` -- whether ``thinking_level`` can be
  translated into ``reasoning_effort`` for the requested model.
* :meth:`artemis.llm.router.ModelFactory.create_model` -- which request
  parameters each provider branch may send.
* :func:`artemis.services.llm._resolve_endpoint` -- the per-node request
  timeout default.

Unknown models fall through to their provider's default, which is deliberately
the pre-table behaviour: send ``temperature``, send no reasoning knob. A model
we have never seen therefore behaves exactly as it did before, and adding it
here is what opts it into the newer knobs.
"""

from dataclasses import dataclass
import re

from artemis.utils.logger import get_logger

logger = get_logger(__name__)

#: The reasoning-effort vocabulary shared by current OpenAI reasoning models.
#: ``"none"`` is absent on purpose: the frontier reasoning models reject it, so
#: a node that wants cheap thinking asks for ``"minimal"``/``"low"`` instead.
_OPENAI_EFFORTS = ("minimal", "low", "medium", "high")

#: Anthropic has no native effort scale -- the router converts effort into an
#: extended-thinking token budget -- so the three mapped rungs are the vocabulary.
_ANTHROPIC_EFFORTS = ("low", "medium", "high")

#: Gemini's own scale, spelled ``thinking_level`` rather than ``reasoning_effort``.
_GOOGLE_THINKING_LEVELS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class ModelCapabilities:
    """What one model accepts, and what it needs to perform well."""

    #: Whether the model takes an explicit reasoning/thinking budget at all.
    #: False means "send no reasoning knob" -- both ``reasoning_effort`` and
    #: ``thinking_level`` are withheld, and a configured value is dropped
    #: rather than forwarded into a 400.
    supports_reasoning: bool = False

    #: The accepted spellings, most-frugal first. A configured value outside
    #: this tuple is clamped to the nearest supported rung by
    #: :func:`clamp_reasoning_effort`.
    reasoning_values: tuple[str, ...] = ()

    #: Google/Vertex read the budget as ``thinking_level``; everyone else reads
    #: ``reasoning_effort``. Both are driven from the same configured intent.
    reasoning_param: str = "reasoning_effort"

    #: Whether ``temperature`` may be sent. The newer OpenAI reasoning models
    #: and the newer Claude models reject every value, 0.0 included.
    supports_temperature: bool = True

    #: OpenAI only: the model emits reasoning items that must be replayed to
    #: avoid re-deriving the chain of thought every turn. ARTEMIS rewrites its
    #: message list on every turn (transcript ledger, screenshot scrubbing,
    #: capsule substitution), so server-side item ids do not survive and the
    #: reasoning has to travel inline as ``reasoning.encrypted_content``.
    carries_reasoning_inline: bool = False

    #: Whether a single call can plausibly run for minutes. Drives the request
    #: timeout default, which used to be a flat 60s for every node and model --
    #: short enough that a reasoning model's final review timed out and was
    #: silently retried.
    slow_first_token: bool = False

    #: Whether this is a Gemini-family model, and so eligible for the native
    #: google-genai engines (Files API video, context caching, explicit
    #: thinking config) rather than the universal LangChain path.
    is_gemini_family: bool = False


# --- Provider defaults ---------------------------------------------------------------
#
# Pre-table behaviour: temperature is sent, no reasoning knob is sent. Anything
# not matched below lands here and is unchanged by this module's introduction.

#: Providers that serve Gemini models, and so reach the native google-genai
#: engines. This is the canonical home of the test that used to be spelled
#: ``_THINKING_LEVEL_PROVIDERS`` in artemis.config.llm and as a bare
#: ``"gemini" in model_name`` in the explorer and video analyzer.
_GEMINI_PROVIDERS = frozenset({"google", "vertexai"})

#: Non-canonical provider spellings, mirroring ``ModelProvider.from_string``.
_PROVIDER_ALIASES = {
    "gemini": "google",
    "claude": "anthropic",
    "grok": "xai",
    "vertex": "vertexai",
}

_GEMINI_DEFAULT = ModelCapabilities(is_gemini_family=True)
_FALLBACK = ModelCapabilities()


# --- Per-model rules -----------------------------------------------------------------
#
# Ordered; first match wins, so put the narrow patterns above the broad ones.
# Patterns are matched case-insensitively against the model id with `re.search`.

_RULES: dict[str, tuple[tuple[str, ModelCapabilities], ...]] = {
    "google": (
        # Gemini 3+ takes an explicit thinking level. Anything older ignores the
        # parameter and errors on some builds -- this replaces the
        # `if any(v in model for v in ("2.5", "2.0", "1.5"))` test that used to
        # sit inline in the router's Google branch.
        (
            r"gemini-(?:[3-9]|\d{2,})",
            ModelCapabilities(
                supports_reasoning=True,
                reasoning_values=_GOOGLE_THINKING_LEVELS,
                reasoning_param="thinking_level",
                is_gemini_family=True,
            ),
        ),
        (r"gemini", ModelCapabilities(is_gemini_family=True)),
    ),
    "openai": (
        # The reasoning line: gpt-5.x / gpt-6+ codenamed models (astra, sol,
        # terra), the o-series, and the gpt-5 size variants. These reject
        # `temperature` outright, need `/v1/responses` to serve function tools,
        # and re-derive their chain of thought unless it is replayed to them.
        (
            r"^(?:gpt-(?:[5-9]|\d{2,})|o[1-9])",
            ModelCapabilities(
                supports_reasoning=True,
                reasoning_values=_OPENAI_EFFORTS,
                supports_temperature=False,
                carries_reasoning_inline=True,
                slow_first_token=True,
            ),
        ),
        # gpt-4o / gpt-4.1 and friends: temperature yes, reasoning knob no.
        (r"^gpt-4", ModelCapabilities()),
    ),
    "anthropic": (
        # Claude 5 family and Haiku 4.5: `temperature` is deprecated (any value
        # is a 400) and effort maps to an extended-thinking budget.
        (
            r"claude-(?:opus-[5-9]|sonnet-[5-9]|haiku-(?:4[.-]5|[5-9])|fable-[5-9])",
            ModelCapabilities(
                supports_reasoning=True,
                reasoning_values=_ANTHROPIC_EFFORTS,
                supports_temperature=False,
                slow_first_token=True,
            ),
        ),
        (r"claude", ModelCapabilities()),
    ),
}


_RULES["vertexai"] = _RULES["google"]


def capabilities_for(provider: str | None, model: str | None) -> ModelCapabilities:
    """The capabilities of ``model`` as served by ``provider``.

    An unrecognized model falls back to its provider's default, so it keeps the
    behaviour it had before this table existed: temperature sent, no reasoning
    knob sent. Provider aliases (``gemini``, ``claude``, ``grok``, ``vertex``)
    are accepted.
    """
    raw = str(getattr(provider, "value", provider) or "").strip().lower()
    key = _PROVIDER_ALIASES.get(raw, raw) or "google"
    name = str(model or "").strip().lower()
    for pattern, caps in _RULES.get(key, ()):
        if re.search(pattern, name):
            return caps
    return _GEMINI_DEFAULT if key in _GEMINI_PROVIDERS else _FALLBACK


def clamp_reasoning_effort(value: str | None, caps: ModelCapabilities) -> str | None:
    """``value`` expressed in the vocabulary ``caps`` actually accepts.

    Returns ``None`` when the model takes no reasoning knob, so callers can
    drop the parameter instead of forwarding it into a 400. A value the model
    does not spell (``"minimal"`` on Anthropic, say) is clamped to the nearest
    supported rung rather than dropped -- the configured intent was "think less
    here", and the nearest rung honours it.
    """
    if not caps.supports_reasoning or not caps.reasoning_values:
        return None
    wanted = str(value or "").strip().lower()
    if not wanted:
        return None
    if wanted in caps.reasoning_values:
        return wanted
    ladder = ("none", "minimal", "low", "medium", "high")
    if wanted not in ladder:
        logger.debug(f"Unknown reasoning effort {value!r}; leaving the parameter unset.")
        return None
    target = ladder.index(wanted)
    return min(caps.reasoning_values, key=lambda v: abs(ladder.index(v) - target))
