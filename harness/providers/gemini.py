"""Google Gemini adapter (google-genai SDK).

Translates canonical Message / ToolCall types to the Gemini API's
Content / Part / FunctionCall format and back. Same invariants as the
other adapters: no auto-repair of malformed calls, native tool calling
only, raw response archived verbatim.

Gemini specifics:
- System prompt is `system_instruction` on `GenerateContentConfig`,
  not a message in the contents list.
- Tool schemas live inside a `Tool(function_declarations=[...])`
  passed via `config.tools`. We use `parameters_json_schema` to pass
  our JSONSchema dicts unchanged.
- Tool calls come back as `Part(function_call=FunctionCall(id, name, args))`
  on a `model`-role Content. The `args` field is already a parsed dict —
  Gemini parses the model's emission server-side. If the model emitted
  garbage Gemini couldn't parse, you instead get `finish_reason=
  MALFORMED_FUNCTION_CALL` and no FunctionCall part. We surface that as
  the AssistantTurn's stop_reason; the absence of tool_calls naturally
  causes the runner to terminate the turn (which the evaluator can then
  count against the model).
- Tool results are sent back as `Part(function_response=FunctionResponse(
  id, name, response={...}))` parts inside a `user`-role Content.
- Roles on the wire are "user" and "model" (not "assistant"). System
  messages don't appear in the contents list at all.
- Gemini's FunctionCall.id is optional but is populated for parallel
  tool calling — we preserve it when present and synthesize one when
  absent so our canonical call_id contract holds.

See `doc/design.md` D1–D6 for the underlying design decisions.
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


_RETRY_BACKOFFS = [1.0, 2.0, 4.0]
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class GeminiProvider:
    """Adapter for Google Gemini via the `google-genai` SDK.

    Instantiate with an API key (or let the SDK pick it up from
    GEMINI_API_KEY / GOOGLE_API_KEY). Re-use a single instance across
    runs — the SDK client is cheap to keep around.
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._synth_id_counter = 0

    def reset_run_state(self) -> None:
        """Reset the synthetic-call-id counter so each run starts fresh.

        Without this, re-using the same provider across runs causes ids
        to grow monotonically across the entire process lifetime, which
        is harmless for correctness (ids stay unique within each run's
        message list) but breaks byte-for-byte determinism between two
        replays of the same trajectory.
        """
        self._synth_id_counter = 0

    # ------------------------------------------------------------------
    # Schema translation
    # ------------------------------------------------------------------

    def format_tools(self, schemas: list[dict]) -> Any:
        """Convert canonical tool schemas to a single Gemini Tool object.

        Returns either a one-element list `[Tool(function_declarations=[...])]`
        ready to pass as `config.tools`, or `None` if no schemas were
        provided (Gemini rejects empty tool lists).

        We use `parameters_json_schema` to pass the JSONSchema body
        unchanged. The alternative is `parameters` with a `Schema`
        object, but our `compute_value` schema uses `oneOf` at the top
        level which the Schema class doesn't support cleanly.
        """
        from google.genai import types

        if not schemas:
            return None

        decls = [
            types.FunctionDeclaration(
                name=s["name"],
                description=s["description"],
                parameters_json_schema=s["parameters"],
            )
            for s in schemas
        ]
        return [types.Tool(function_declarations=decls)]

    # ------------------------------------------------------------------
    # Message translation
    # ------------------------------------------------------------------

    def _messages_to_gemini(self, messages: list[Message]) -> list[Any]:
        """Convert canonical Messages to Gemini Content list.

        Rules:
        - user text → Content(role="user", parts=[Part.from_text(...)])
        - assistant with text+tool_calls → Content(role="model", parts=[
            Part.from_text(text), Part.from_function_call(name, args), ...])
        - tool result → Content(role="user", parts=[
            Part.from_function_response(name, response)])
        - consecutive tool messages merge into one user Content
        """
        from google.genai import types

        out: list[Any] = []
        pending_tool_parts: list[Any] = []

        def flush_tool_parts():
            nonlocal pending_tool_parts
            if pending_tool_parts:
                out.append(types.Content(role="user", parts=pending_tool_parts))
                pending_tool_parts = []

        for m in messages:
            if m.role == "tool":
                # Build a function_response part. The wire schema requires
                # `response` to be a dict — we pass the full envelope so
                # the model sees status/result/error consistently.
                envelope = m.tool_result or {}
                # Gemini expects a JSON-serializable dict here. Our envelope
                # already is, but we ensure dict-shape just in case.
                response_dict: dict
                if isinstance(envelope, dict):
                    response_dict = envelope
                else:
                    response_dict = {"value": envelope}
                # Look up the original call to recover the tool name —
                # tool_result Messages only carry call_id, not name. We
                # need to fish the name out of the most recent assistant
                # message's tool_calls.
                tool_name = self._lookup_tool_name(messages, m.call_id)
                pending_tool_parts.append(
                    types.Part.from_function_response(
                        name=tool_name or "unknown_tool",
                        response=response_dict,
                    )
                )
                continue

            flush_tool_parts()

            if m.role == "user":
                out.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=m.text or "")],
                    )
                )
            elif m.role == "assistant":
                parts: list[Any] = []
                if m.text:
                    parts.append(types.Part.from_text(text=m.text))
                for tc in m.tool_calls:
                    if tc.parse_error is not None:
                        # Symmetric with Anthropic: Gemini parses args
                        # server-side, so it cannot round-trip a broken
                        # call. Refuse rather than silently replaying {}.
                        raise ValueError(
                            "Gemini cannot replay a malformed tool call: "
                            "function_call.args must already be parsed JSON"
                        )
                    parts.append(
                        types.Part.from_function_call(
                            name=tc.name,
                            args=tc.arguments or {},
                        )
                    )
                if parts:
                    out.append(types.Content(role="model", parts=parts))
            else:
                raise ValueError(f"Unexpected role in message list: {m.role}")

        flush_tool_parts()
        return out

    @staticmethod
    def _lookup_tool_name(messages: list[Message], call_id: str | None) -> str | None:
        """Find the tool name for a tool_result by its call_id.

        Walks backward from the end so we hit the most recent assistant
        message's tool_calls first.
        """
        if call_id is None:
            return None
        for m in reversed(messages):
            if m.role == "assistant":
                for tc in m.tool_calls:
                    if tc.call_id == call_id:
                        return tc.name
        return None

    def _next_synth_id(self) -> str:
        self._synth_id_counter += 1
        return f"gemini_synth_{self._synth_id_counter}"

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        """Decide whether a Gemini APIError should be retried.

        The previous version of this check tried `int(e.status)`, but
        `google.genai.errors.APIError.status` is the GCP status *name*
        (e.g. "RESOURCE_EXHAUSTED", "UNAVAILABLE"), not the HTTP code.
        That made every 429/5xx fall through as non-transient. The HTTP
        integer lives on `e.code` — use that. See doc/gaps.md G-013.

        Per-minute rate limits are by far the most common 429 in
        practice; daily-quota 429s won't recover within our retry
        budget but they cost only ~7s of wasted backoff before the
        adapter raises, which is acceptable.
        """
        from google.genai import errors as genai_errors

        if isinstance(e, genai_errors.ServerError):
            return True
        code = getattr(e, "code", None)
        return isinstance(code, int) and code in _TRANSIENT_STATUS

    def _parse_response(self, resp: Any, latency_ms: int, retries: int) -> AssistantTurn:
        """Convert a Gemini GenerateContentResponse into an AssistantTurn.

        Critical: when the model emits something Gemini can't parse as a
        valid function call, the response has finish_reason=
        MALFORMED_FUNCTION_CALL and no function_call part. We pass that
        through as stop_reason — the absence of tool_calls means the
        runner will treat the turn as a final answer (or empty), which
        the evaluator can flag as a malformed-call failure mode.
        """
        raw_dict = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "unknown"

        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            fr = getattr(cand, "finish_reason", None)
            stop_reason = fr.value if hasattr(fr, "value") else (str(fr) if fr else "unknown")

            content = getattr(cand, "content", None)
            if content is not None:
                for part in (getattr(content, "parts", None) or []):
                    # Text content
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

                    # Function call
                    fc = getattr(part, "function_call", None)
                    if fc is not None and getattr(fc, "name", None):
                        args = getattr(fc, "args", None) or {}
                        # Gemini normally parses args server-side and
                        # delivers them as a dict. If the model emitted
                        # unparseable JSON, Gemini reports
                        # MALFORMED_FUNCTION_CALL via finish_reason and
                        # we don't get a function_call part at all — so
                        # parse_error is usually None here. But if the
                        # args ever come through as a non-dict (e.g. a
                        # list), surface that as F2 — symmetric with
                        # the Anthropic adapter.
                        call_id = getattr(fc, "id", None) or self._next_synth_id()
                        try:
                            raw_args = json.dumps(args) if not isinstance(args, dict) else json.dumps(dict(args))
                        except (TypeError, ValueError):
                            raw_args = repr(args)
                        parsed_args: dict = {}
                        parse_err: str | None = None
                        if isinstance(args, dict):
                            parsed_args = dict(args)
                        else:
                            parse_err = (
                                "tool arguments parsed to non-dict: "
                                f"{type(args).__name__}"
                            )
                        tool_calls.append(
                            ToolCall(
                                call_id=call_id,
                                name=fc.name,
                                arguments=parsed_args,
                                raw_arguments=raw_args,
                                parse_error=parse_err,
                            )
                        )

        text = "\n".join(text_parts) if text_parts else None

        usage_md = getattr(resp, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_md, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_md, "candidates_token_count", 0) or 0,
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
        from google.genai import types
        from google.genai import errors as genai_errors

        gen_config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_output_tokens,
        }
        if tools:
            gen_config_kwargs["tools"] = tools
            # Force the model into AUTO function-calling mode (the default).
            # Setting this explicitly makes the contract with vLLM/Vertex
            # variants identical regardless of server defaults.
            gen_config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        if config.extra:
            gen_config_kwargs.update(config.extra)

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        contents = self._messages_to_gemini(messages)

        retries = 0
        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_BACKOFFS) + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(
                    model=config.model,
                    contents=contents,
                    config=gen_config,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return self._parse_response(resp, latency_ms, retries)
            except genai_errors.APIError as e:
                if self._is_transient(e) and attempt < len(_RETRY_BACKOFFS):
                    time.sleep(_RETRY_BACKOFFS[attempt])
                    retries += 1
                    last_exc = e
                    continue
                raise

        raise RuntimeError("retry loop exited without result") from last_exc
