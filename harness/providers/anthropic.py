"""Anthropic Messages API adapter.

Translates canonical Message / ToolCall types to Anthropic's native
format and back. See `base.py` for the invariants this adapter must
preserve (no auto-repair, native tool calling only, raw response
archived verbatim).

Anthropic specifics:
- System prompt goes in a top-level `system` parameter, not in messages.
- Tool schemas go in a top-level `tools` parameter with the shape:
  `{"name": ..., "description": ..., "input_schema": {...}}`.
- Tool calls appear as `{"type": "tool_use", "id": ..., "name": ...,
  "input": {...}}` content blocks inside the assistant message.
- Tool results must be sent back as user-role messages containing
  `{"type": "tool_result", "tool_use_id": ..., "content": ...}` blocks.
- `stop_reason` values: "end_turn", "tool_use", "max_tokens",
  "stop_sequence", "pause_turn", "refusal".
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import (
    AssistantTurn,
    Message,
    ModelConfig,
    TokenUsage,
    ToolCall,
)


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_RETRY_BACKOFFS = [1.0, 2.0, 4.0]   # seconds


class AnthropicProvider:
    """Adapter for the Anthropic Messages API.

    Instantiate with an API key (or let the SDK pick it up from
    ANTHROPIC_API_KEY). Re-use a single instance across many task
    runs — the underlying SDK client is thread-safe and connection-
    pooled.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Schema translation
    # ------------------------------------------------------------------

    def format_tools(self, schemas: list[dict]) -> list[dict]:
        """Convert canonical tool schemas to Anthropic tool format.

        Canonical shape (from `TOOL_SCHEMAS` in schemas.py):
            {"name": ..., "description": ..., "parameters": {...}}

        Anthropic shape:
            {"name": ..., "description": ..., "input_schema": {...}}

        Note: Anthropic requires `input_schema` to be a JSONSchema object.
        The `compute_value` schema uses `oneOf` at the top level, which
        Anthropic's validator accepts as an `input_schema` body.
        """
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["parameters"],
            }
            for s in schemas
        ]

    # ------------------------------------------------------------------
    # Message translation
    # ------------------------------------------------------------------

    def _messages_to_anthropic(self, messages: list[Message]) -> list[dict]:
        """Convert canonical Messages to Anthropic wire format.

        Rules:
        - user/text → `{"role": "user", "content": [{"type": "text", ...}]}`
        - assistant with text+tool_calls → one message with mixed content
          blocks (text first, then tool_use blocks)
        - tool result → Anthropic requires role="user" carrying tool_result blocks;
          consecutive tool messages are merged into one user message
        """
        out: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tool_results():
            nonlocal pending_tool_results
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for m in messages:
            if m.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.call_id,
                        "content": json.dumps(m.tool_result, default=str),
                    }
                )
                continue

            # Any non-tool message flushes the pending tool_result batch
            flush_tool_results()

            if m.role == "user":
                out.append(
                    {"role": "user", "content": [{"type": "text", "text": m.text or ""}]}
                )
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    if tc.parse_error is not None:
                        raise ValueError(
                            "Anthropic cannot replay a malformed tool call: "
                            "tool_use input must already be parsed JSON"
                        )
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": tc.name,
                            "input": tc.arguments or {},
                        }
                    )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            else:
                # system messages are passed separately; should never appear here
                raise ValueError(f"Unexpected role in message list: {m.role}")

        flush_tool_results()
        return out

    def _parse_response(self, resp: Any, latency_ms: int, retries: int) -> AssistantTurn:
        """Convert an Anthropic Messages response into an AssistantTurn.

        Anthropic returns a Message object with:
        - `content`: list of blocks (text, tool_use, etc.)
        - `stop_reason`: termination hint
        - `usage`: {input_tokens, output_tokens, cache_creation_input_tokens, ...}
        """
        raw_dict = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                # Anthropic returns already-parsed JSON in `input`. If the
                # model emitted garbage that Anthropic couldn't parse,
                # this block wouldn't have reached us as tool_use at all —
                # but the input may still be the wrong shape for our tool
                # contract (e.g. a list instead of an object). Preserve
                # that as a parse_error so the runner logs F2.
                try:
                    raw_args = json.dumps(block.input)
                except (TypeError, ValueError):
                    raw_args = repr(block.input)
                parsed_args: dict = {}
                parse_err: str | None = None
                if isinstance(block.input, dict):
                    parsed_args = dict(block.input)
                else:
                    parse_err = (
                        "tool arguments parsed to non-dict: "
                        f"{type(block.input).__name__}"
                    )
                tool_calls.append(
                    ToolCall(
                        call_id=block.id,
                        name=block.name,
                        arguments=parsed_args,
                        raw_arguments=raw_args,
                        parse_error=parse_err,
                    )
                )
            # thinking / other block types are archived in raw_response but
            # not surfaced to the runner — they don't affect tool dispatch.

        text = "\n".join(text_parts) if text_parts else None
        stop_reason = getattr(resp, "stop_reason", None) or "unknown"

        usage = TokenUsage(
            input_tokens=(
                (getattr(resp.usage, "input_tokens", 0) or 0)
                + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0)
            ),
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )

        return AssistantTurn(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=raw_dict,
            provider_retries=retries,
        )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Any,
        config: ModelConfig,
    ) -> AssistantTurn:
        import anthropic

        payload: dict = {
            "model": config.model,
            "system": system,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if tools:
            payload["tools"] = tools
        if config.extra:
            payload.update(config.extra)

        retries = 0
        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_BACKOFFS) + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.messages.create(**payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return self._parse_response(resp, latency_ms, retries)
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status in _TRANSIENT_STATUS and attempt < len(_RETRY_BACKOFFS):
                    time.sleep(_RETRY_BACKOFFS[attempt])
                    retries += 1
                    last_exc = e
                    continue
                raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                if attempt < len(_RETRY_BACKOFFS):
                    time.sleep(_RETRY_BACKOFFS[attempt])
                    retries += 1
                    last_exc = e
                    continue
                raise

        # Unreachable — the loop either returns or raises — but satisfies type checkers
        raise RuntimeError("retry loop exited without result") from last_exc
