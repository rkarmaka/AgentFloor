"""OpenAI Chat Completions adapter.

Translates canonical Message / ToolCall types to OpenAI's native
tool-calling format and back.

OpenAI specifics:
- System prompt is a `{"role": "system", ...}` message at the head of
  the messages list.
- Tool schemas go in a top-level `tools` parameter with the shape:
  `{"type": "function", "function": {"name": ..., "description": ...,
  "parameters": {...}}}`.
- Tool calls appear on assistant messages as a `tool_calls` list:
  `[{"id": ..., "type": "function", "function": {"name": ...,
  "arguments": "<JSON string>"}}]`. Note: arguments is a JSON string,
  not a dict — this is where small models most often emit broken JSON,
  and it is the single most important parse-error source we must
  preserve faithfully.
- Tool results are sent back as `{"role": "tool", "tool_call_id": ...,
  "content": "<string>"}` messages.
- `finish_reason` values: "stop", "tool_calls", "length",
  "content_filter".

This class is subclassed by `OpenAICompatibleProvider` to cover vLLM,
Ollama, Together, Groq, Fireworks, etc. — same translation logic,
different base_url.
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
_RETRY_BACKOFFS = [1.0, 2.0, 4.0]

# Reasoning-class models (o-series, gpt-5 family) use a different parameter
# surface: `max_completion_tokens` instead of `max_tokens`, and they reject
# non-default `temperature`/`top_p`. Detect by prefix.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(name: str) -> bool:
    n = name.lower()
    return any(n == p or n.startswith(p + "-") for p in _REASONING_PREFIXES)


def _openai_safe_parameters(params: dict) -> dict:
    """Return a parameters schema that passes OpenAI's strict validator.

    Reasoning-class models reject top-level `oneOf` and `type: object`
    without `properties`. We collapse `oneOf` to a permissive object
    wrapper (the argument parser on the harness side still validates
    against the canonical schema after the call lands) and inject an
    empty `properties: {}` when missing.
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if "oneOf" in params and "type" not in params:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if params.get("type") == "object" and "properties" not in params:
        out = dict(params)
        out["properties"] = {}
        return out
    return params


class OpenAIProvider:
    """Adapter for the OpenAI Chat Completions API.

    Uses the official `openai` Python SDK. Instantiate with an API key
    (or let the SDK pick it up from OPENAI_API_KEY). Re-use a single
    instance across many runs.
    """

    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import openai
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs) if kwargs else openai.OpenAI()

    # ------------------------------------------------------------------
    # Schema translation
    # ------------------------------------------------------------------

    def format_tools(self, schemas: list[dict]) -> list[dict]:
        """Convert canonical tool schemas to OpenAI function format.

        Canonical:  {"name": ..., "description": ..., "parameters": {...}}
        OpenAI:     {"type": "function",
                     "function": {"name": ..., "description": ..., "parameters": {...}}}

        Adapts two shapes that OpenAI reasoning models (gpt-5, o-series)
        reject with 400 / invalid_function_parameters:
        - top-level `oneOf` — collapsed to a relaxed object schema
        - `type: object` without `properties` — `properties: {}` injected
        Other backends (Ollama, vLLM) accept the canonical schemas as-is,
        so we keep canonical untouched and adapt here.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": _openai_safe_parameters(s["parameters"]),
                },
            }
            for s in schemas
        ]

    # ------------------------------------------------------------------
    # Message translation
    # ------------------------------------------------------------------

    def _messages_to_openai(
        self, system: str, messages: list[Message]
    ) -> list[dict]:
        """Convert canonical Messages plus a system prompt to OpenAI wire format."""
        out: list[dict] = [{"role": "system", "content": system}]

        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant"}
                msg["content"] = m.text if m.text is not None else None
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                # Preserve the exact malformed JSON string if
                                # parsing failed. Replaying "{}" here would
                                # silently rewrite the model-visible history.
                                "arguments": (
                                    tc.raw_arguments
                                    if tc.parse_error is not None
                                    else json.dumps(tc.arguments or {})
                                ),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.call_id,
                        "content": json.dumps(m.tool_result, default=str),
                    }
                )
            else:
                raise ValueError(f"Unexpected role: {m.role}")

        return out

    def _parse_response(self, resp: Any, latency_ms: int, retries: int) -> AssistantTurn:
        """Convert an OpenAI ChatCompletion into an AssistantTurn.

        Critical: `tool_calls[i].function.arguments` is a JSON *string*.
        Small models frequently emit broken JSON here. We preserve the
        raw string in `raw_arguments` and set `parse_error` on failure —
        we do NOT attempt repair. See D3 / the no-auto-repair invariant.
        """
        raw_dict = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        choice = resp.choices[0]
        msg = choice.message

        text = msg.content if getattr(msg, "content", None) else None

        tool_calls: list[ToolCall] = []
        raw_tcs = getattr(msg, "tool_calls", None) or []
        for raw_tc in raw_tcs:
            fn = raw_tc.function
            raw_args = fn.arguments or ""
            parsed: dict = {}
            parse_err: str | None = None
            try:
                parsed_any = json.loads(raw_args) if raw_args else {}
                if isinstance(parsed_any, dict):
                    parsed = parsed_any
                else:
                    parse_err = f"tool arguments parsed to non-dict: {type(parsed_any).__name__}"
            except json.JSONDecodeError as e:
                parse_err = f"JSONDecodeError: {e.msg} at pos {e.pos}"
            tool_calls.append(
                ToolCall(
                    call_id=raw_tc.id,
                    name=fn.name,
                    arguments=parsed,
                    raw_arguments=raw_args,
                    parse_error=parse_err,
                )
            )

        stop_reason = getattr(choice, "finish_reason", None) or "unknown"
        usage_obj = getattr(resp, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
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
        import openai

        payload: dict = {
            "model": config.model,
            "messages": self._messages_to_openai(system, messages),
        }
        if _is_reasoning_model(config.model):
            payload["max_completion_tokens"] = config.max_output_tokens
        else:
            payload["temperature"] = config.temperature
            payload["top_p"] = config.top_p
            payload["max_tokens"] = config.max_output_tokens
        if tools:
            payload["tools"] = tools
        if config.extra:
            payload.update(config.extra)

        retries = 0
        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_BACKOFFS) + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return self._parse_response(resp, latency_ms, retries)
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status in _TRANSIENT_STATUS and attempt < len(_RETRY_BACKOFFS):
                    time.sleep(_RETRY_BACKOFFS[attempt])
                    retries += 1
                    last_exc = e
                    continue
                raise
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < len(_RETRY_BACKOFFS):
                    time.sleep(_RETRY_BACKOFFS[attempt])
                    retries += 1
                    last_exc = e
                    continue
                raise

        raise RuntimeError("retry loop exited without result") from last_exc
