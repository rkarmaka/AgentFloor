"""Canonical types and the Provider protocol for the benchmark runner.

The runner never touches provider SDKs directly. It only constructs
`Message` objects, calls `Provider.complete()`, and reads the
`AssistantTurn` returned by the adapter. Adding a new backend means
writing one file that implements the `Provider` protocol — no changes
to runner or harness core.

Design invariants (see `doc/design.md` D1–D6):

1. **Native tool calling only.** Adapters MUST NOT parse JSON out of
   text content to synthesize `tool_calls`. If the provider's native
   tool-use field is empty, `AssistantTurn.tool_calls` is empty.

2. **No auto-repair.** If the model emits broken JSON in a tool call's
   arguments, the adapter returns `arguments={}`, `raw_arguments=<raw>`,
   and `parse_error=<reason>`. The runner feeds this through
   `ToolRouter.call()` which will log it as F2 (malformed call).

3. **Raw response archived verbatim.** `AssistantTurn.raw_response`
   holds the full provider JSON so post-hoc forensics and new metrics
   can be computed without re-running.

4. **Parallel tool calls preserved.** A single turn may emit zero or
   more tool calls; the runner dispatches each in order.

Canonical message shape:

    Message(role="system" | "user" | "assistant" | "tool", ...)

Adapters translate to/from this shape — the runner never sees
provider-specific formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


# ----------------------------------------------------------------------
# Core data types
# ----------------------------------------------------------------------


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class TokenUsage:
    """Per-turn token accounting reported by the provider.

    Some providers report cache hits / prompt caching separately; those
    are summed into `input_tokens` here. If a provider doesn't surface a
    field, it's left at 0 — check `raw_response` for per-provider detail.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCall:
    """A single tool call emitted by the model in one assistant turn.

    `call_id` is the provider-issued identifier that correlates this
    call with its eventual tool_result message. Anthropic calls this
    `id`, OpenAI calls it `tool_call_id` — adapters normalize to
    `call_id` here.

    `arguments` is the parsed dict. If parsing failed, it is `{}` and
    `parse_error` is set. The runner passes broken calls through to
    `ToolRouter.call()` anyway so F2 gets logged; it does not attempt
    repair. `raw_arguments` always preserves the exact string the
    model emitted for forensics.
    """

    call_id: str
    name: str
    arguments: dict = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class Message:
    """A single message in the canonical trajectory.

    The content shape depends on the role:

    - system / user: plain text in `text`. `tool_calls` and
      `tool_result` are unset.
    - assistant: optional `text`, plus zero-or-more `tool_calls`.
      This is the output of one `AssistantTurn`.
    - tool: one envelope per tool call in `tool_result`, correlated by
      `call_id`. One Message per tool_result.

    The runner appends to a `list[Message]` and each adapter translates
    that list into its provider's native format on every call.
    """

    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role="tool" messages:
    call_id: str | None = None
    tool_result: dict | None = None   # the envelope returned by ToolRouter.call()


@dataclass
class AssistantTurn:
    """A single completion returned by `Provider.complete()`.

    This is the canonical output format across all adapters. The runner
    only reads this; it never inspects `raw_response` for control flow
    (only for archival).
    """

    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str                      # "end_turn" | "tool_use" | "max_tokens" | "error" | other
    usage: TokenUsage
    latency_ms: int
    raw_response: dict                    # full provider JSON — archived verbatim
    provider_retries: int = 0             # transparent retries inside the adapter (429 / 5xx)


@dataclass
class ModelConfig:
    """Per-run model configuration.

    `model` is the provider-specific model ID (e.g. "claude-haiku-4-5",
    "gpt-4o-mini", "qwen2.5:3b"). `temperature` is locked to 0.0 by
    default per `registry.yaml`; override only for ablation experiments.

    `extra` is a dict of provider-specific knobs that pass straight
    through to the SDK call (e.g. `{"reasoning_effort": "low"}` for
    OpenAI o-series). Use sparingly — anything that biases the model
    should be recorded in `doc/design.md` first.
    """

    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Provider protocol
# ----------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """The single interface the runner talks to.

    Implementations live in `providers/anthropic.py`, `providers/openai.py`,
    and `providers/openai_compatible.py`. All three are swappable — the
    runner never branches on provider identity.
    """

    name: str
    """Stable identifier used in result file paths, e.g. 'anthropic',
    'openai', 'openai_compatible:vllm'. Adapters set this on
    instantiation."""

    def reset_run_state(self) -> None:
        """Reset any per-run mutable state on the provider instance.

        Called by the runner at the start of every `run_task` so that
        re-using a provider across runs does not leak state between them
        (e.g. monotonic ID counters used to synthesize tool-call ids when
        a backend doesn't supply them). Default is a no-op; providers
        that hold per-run state override this.
        """
        return None

    def format_tools(self, schemas: list[dict]) -> Any:
        """Translate canonical tool schemas (from `ToolRouter.get_schemas()`)
        into the provider's native tool-use format.

        Called once per run before the loop starts. The returned object
        is opaque to the runner and passed verbatim to `complete()`.
        """
        ...

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Any,
        config: ModelConfig,
    ) -> AssistantTurn:
        """Make one round-trip to the provider.

        `system` is the fixed system prompt (identical across all models).
        `messages` is the canonical trajectory so far (no system message
        — that's passed separately because Anthropic and OpenAI put it
        in different places).
        `tools` is whatever `format_tools()` returned.
        `config` controls temperature, max tokens, etc.

        Must return a fully-populated `AssistantTurn`. Must retry
        transient errors (HTTP 429, 5xx) up to 3 times with exponential
        backoff before raising. Must not mutate `messages`.
        """
        ...


# ----------------------------------------------------------------------
# Message constructors (ergonomics for the runner)
# ----------------------------------------------------------------------


def user_message(text: str) -> Message:
    """Build a user-role message from plain text."""
    return Message(role="user", text=text)


def assistant_message(
    text: str | None, tool_calls: list[ToolCall] | None = None
) -> Message:
    """Build an assistant-role message from an `AssistantTurn`'s fields."""
    return Message(role="assistant", text=text, tool_calls=tool_calls or [])


def tool_result_message(call_id: str, envelope: dict) -> Message:
    """Build a tool-role message carrying one envelope from `ToolRouter.call()`.

    The envelope is the full dict returned by the router — adapters
    will serialize its `result` / `error` fields into the provider's
    tool_result format at send time.
    """
    return Message(role="tool", call_id=call_id, tool_result=envelope)
