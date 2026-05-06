"""The agentic loop for the benchmark.

`run_task(task_path, provider, config)` is the single entry point. It:

1. Loads the task YAML and constructs a FixtureDB + ToolRouter.
2. Presents the task's `example_prompt` to the model as the first user
   message, along with the task's allowed tool schemas.
3. Loops: call `provider.complete()`, append the assistant turn,
   dispatch every tool call through `ToolRouter.call()`, append each
   tool result, repeat.
4. Terminates on: (a) an assistant turn with zero tool calls (final
   answer), (b) `submit_decision` returning status=ok, (c) the step
   budget from the task YAML being exhausted, (d) the provider raising
   an error, or (e) a "malformed terminal call" — zero tool_calls plus
   a provider stop_reason that means the model emitted an unparseable
   tool call (e.g. Gemini's MALFORMED_FUNCTION_CALL). See G-012.
5. Returns a `RunResult` with the full trajectory, call log, token
   usage, latency, and termination reason.

The runner is adapter-agnostic — it only speaks the `Provider` protocol
from `providers/base.py`. Swapping Anthropic for OpenAI, or vLLM for
Ollama, is a one-line change at the caller.

See `doc/design.md` D1–D6 for the underlying design decisions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .db import FixtureDB
from .providers.base import (
    AssistantTurn,
    Message,
    ModelConfig,
    Provider,
    TokenUsage,
    ToolCall,
    assistant_message,
    tool_result_message,
    user_message,
)
from .tools import ToolRouter


# ----------------------------------------------------------------------
# Fixed system prompt (D4 in doc/design.md)
# ----------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are an agent that completes tasks by calling tools. For each task:\n"
    "1. Read the user's request carefully.\n"
    "2. Decide which tool(s) to call, and call them with the correct arguments.\n"
    "3. Use the results of tool calls to decide your next action.\n"
    "4. When the task is complete, respond with a short, direct answer "
    "(or call submit_decision if the task requires a final submission).\n"
    "Call tools using the provider's native tool-calling interface. "
    "Do not invent tools that are not in the provided list."
)


# Provider stop_reasons that mean "the model tried to emit a tool call
# and the provider couldn't parse it." When the assistant turn has zero
# tool_calls AND its stop_reason is in this set, the runner records
# termination="malformed_terminal_call" instead of "final_answer" — this
# keeps F2 (malformed call) separable from F5 (early resignation) in the
# downstream metrics layer. Add new entries here as we observe them.
# See doc/gaps.md G-012 for the rationale.
_MALFORMED_TERMINAL_STOP_REASONS = frozenset({
    "MALFORMED_FUNCTION_CALL",  # Gemini: server-side function-call parse failure
})


# ----------------------------------------------------------------------
# Task loading
# ----------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Parsed task YAML with fields the runner actually uses.

    `raw` holds the full YAML for the evaluator and metrics layers to
    consume later (they read oracle_evaluator, trace_requirements, etc.).
    """

    task_id: str
    level: str
    fixture_path: Path | None
    tools_available: list[str]
    max_steps: int
    requires_final_submission: bool
    example_prompt: str
    raw: dict
    source_path: Path


def load_task(
    task_path: str | Path,
    *,
    variant_id: str | None = None,
) -> TaskSpec:
    """Load a task YAML and resolve its fixture path against the repo root.

    Fixture references in task YAMLs are written relative to the repo
    root (e.g. `fixtures/A/A1_product_catalog.yaml`). The task file
    lives at `<repo>/tasks/<level>/<task>.yaml`, so three `.parent`
    hops from the task file land at the repo root.

    If `variant_id` is set to a non-trivial value (v1..v5), the
    `example_prompt` field is replaced with the matching entry from the
    sibling `<task_id>.variants.yaml` file. v0 / None preserve the
    original prompt — fully backward-compatible with all existing
    callers and tests.
    """
    path = Path(task_path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)

    fixture_ref = data.get("fixture")
    if fixture_ref:
        # tasks/<level>/<task>.yaml → parent.parent.parent = repo root
        repo_root = path.parent.parent.parent
        fixture_path: Path | None = (repo_root / fixture_ref).resolve()
    else:
        fixture_path = None

    example_prompt = data["example_prompt"]
    if variant_id and variant_id not in ("v0", "i0"):
        # Lazy import to avoid a hard dependency at module load time —
        # variants is a sibling package and will always exist in tree,
        # but this keeps the import surface minimal for callers that
        # don't use variants at all.
        if variant_id.startswith("v"):
            from .variants.loader import load_variant_text
            example_prompt = load_variant_text(
                path,
                variant_id,
                example_prompt=example_prompt,
            )
        elif variant_id.startswith("i"):
            # Instance variants change the concrete task instance (different
            # target ID, different expected answer). Apply by mutating `data`
            # before TaskSpec is built — the evaluator reads gold_state and
            # oracle_evaluator off TaskSpec.raw, so those overrides take
            # effect downstream without any evaluator changes.
            from .variants.loader import (
                deep_merge_overrides,
                load_instance_overrides,
            )
            text, overrides = load_instance_overrides(path, variant_id)
            example_prompt = text
            data = deep_merge_overrides(data, overrides)
        else:
            raise ValueError(
                f"unknown variant_id {variant_id!r}; must start with 'v' or 'i'"
            )

    return TaskSpec(
        task_id=data["task_id"],
        level=data.get("level", "unknown"),
        fixture_path=fixture_path,
        tools_available=list(data.get("tools_available") or []),
        max_steps=int(data.get("max_steps", 1)),
        requires_final_submission=bool(data.get("requires_final_submission", False)),
        example_prompt=example_prompt,
        raw=data,
        source_path=path,
    )


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass
class RunResult:
    """Everything produced by one run of one task on one model.

    This is what gets serialized to disk as JSON per the D5 results
    layout. `messages` is the canonical trajectory (human-readable);
    `raw_responses` archives the full provider JSON per turn for
    forensic re-analysis.
    """

    # Identity
    task_id: str
    level: str
    provider: str
    model: str

    # Trajectory
    messages: list[Message]
    call_log: list[dict]
    raw_responses: list[dict]

    # Termination
    termination: str                # "final_answer" | "submitted" | "step_budget_exhausted" | "provider_error" | "malformed_terminal_call"
    termination_detail: str | None  # free text, typically the stop_reason or error message
    final_text: str | None          # last assistant text (often the final answer)

    # Metrics
    total_turns: int
    total_usage: TokenUsage
    total_latency_ms: int
    provider_retries: int

    # Wall-clock
    timestamp_utc: str
    wall_clock_ms: int

    # Config echo
    model_config: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


def run_task(
    task_path: str | Path,
    provider: Provider,
    config: ModelConfig,
    *,
    max_steps_override: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    variant_id: str | None = None,
) -> RunResult:
    """Run one task on one model and return a RunResult.

    `max_steps_override` lets callers shorten/lengthen the budget for
    ablation runs; by default the budget comes from the task YAML.
    `system_prompt` is injectable only for testing — production runs
    should always use `SYSTEM_PROMPT` unchanged.
    `variant_id` selects a prompt variant from the task's
    `<task_id>.variants.yaml` sibling file. None or "v0" preserves the
    original example_prompt — fully backward-compatible.
    """
    wall_start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    task = load_task(task_path, variant_id=variant_id)
    db = FixtureDB(task.fixture_path)
    # FixtureDB's task_id comes from the fixture; for A0/A3 (no fixture)
    # we stamp the task_id on the empty DB so call_ids are still prefixed
    # correctly.
    if db.is_empty:
        db.data["task_id"] = task.task_id

    router = ToolRouter(db, task.tools_available)
    # Reset any per-run state on the provider (e.g. Gemini's synthetic
    # tool-call id counter) so re-using a provider across runs does not
    # leak state between them.
    provider.reset_run_state()
    tools_payload = provider.format_tools(router.get_schemas())

    messages: list[Message] = [user_message(task.example_prompt)]
    raw_responses: list[dict] = []
    total_usage = TokenUsage()
    total_latency_ms = 0
    total_retries = 0
    budget = max_steps_override or task.max_steps

    termination: str = "step_budget_exhausted"
    termination_detail: str | None = None
    final_text: str | None = None
    turns_used = 0

    for step_idx in range(budget):
        try:
            turn: AssistantTurn = provider.complete(
                system=system_prompt,
                messages=messages,
                tools=tools_payload,
                config=config,
            )
        except Exception as e:
            termination = "provider_error"
            termination_detail = f"{type(e).__name__}: {e}"
            break

        turns_used += 1
        raw_responses.append(turn.raw_response)
        total_usage.input_tokens += turn.usage.input_tokens
        total_usage.output_tokens += turn.usage.output_tokens
        total_latency_ms += turn.latency_ms
        total_retries += turn.provider_retries

        messages.append(assistant_message(turn.text, turn.tool_calls))
        if turn.text:
            final_text = turn.text

        # Case 1: no tool calls → either the model is done speaking, OR
        # it tried to emit a tool call and the provider couldn't parse it.
        # Some providers (notably Gemini) signal the broken-call case via
        # a specific stop_reason; we surface those as a distinct
        # termination so the metrics layer can separate F2 (malformed)
        # from F5 (early resignation). See doc/gaps.md G-012.
        if not turn.tool_calls:
            if turn.stop_reason in _MALFORMED_TERMINAL_STOP_REASONS:
                termination = "malformed_terminal_call"
            else:
                termination = "final_answer"
            termination_detail = turn.stop_reason
            break

        # Case 2: dispatch every tool call in the turn
        submitted_ok = False
        for tc in turn.tool_calls:
            envelope = _dispatch_tool_call(router, tc)
            messages.append(tool_result_message(tc.call_id, envelope))
            if tc.name == "submit_decision" and envelope["status"] == "ok":
                submitted_ok = True

        if submitted_ok:
            termination = "submitted"
            termination_detail = "submit_decision returned status=ok"
            break
        # otherwise: loop to next turn

    wall_clock_ms = int((time.perf_counter() - wall_start) * 1000)

    return RunResult(
        task_id=task.task_id,
        level=task.level,
        provider=provider.name,
        model=config.model,
        messages=messages,
        call_log=router.call_log,
        raw_responses=raw_responses,
        termination=termination,
        termination_detail=termination_detail,
        final_text=final_text,
        total_turns=turns_used,
        total_usage=total_usage,
        total_latency_ms=total_latency_ms,
        provider_retries=total_retries,
        timestamp_utc=timestamp,
        wall_clock_ms=wall_clock_ms,
        model_config={
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_output_tokens,
            "extra": config.extra,
        },
    )


def _dispatch_tool_call(router: ToolRouter, tc: ToolCall) -> dict:
    """Route one ToolCall through the ToolRouter.

    If the adapter reported a parse error (broken JSON in arguments),
    we synthesize a call with a sentinel key so the router logs it as
    a schema validation failure (F2) — consistent with how the router
    treats any other malformed call. `raw_arguments` is preserved in
    the canonical Message via `tc`, so forensics remain intact.
    """
    if tc.parse_error is not None:
        # Preserve provider-reported JSON parse failures as F2 even for
        # polymorphic tools whose schema may otherwise accept arbitrary keys.
        return router.call(
            tc.name,
            {"__parse_error__": tc.parse_error, "__raw__": tc.raw_arguments},
            malformed_override=(
                f"provider could not parse tool arguments: {tc.parse_error}"
            ),
        )
    return router.call(tc.name, tc.arguments)


# ----------------------------------------------------------------------
# Serialization helpers (used by the not-yet-written batch runner)
# ----------------------------------------------------------------------


def result_to_dict(result: RunResult) -> dict:
    """Convert a RunResult to a JSON-serializable dict.

    Used when writing to `results/{provider}/{model}/...json`.
    Messages are flattened into a simple list-of-dicts form; ToolCall
    objects become plain dicts.
    """
    return {
        "task_id": result.task_id,
        "level": result.level,
        "provider": result.provider,
        "model": result.model,
        "model_config": result.model_config,
        "messages": [_message_to_dict(m) for m in result.messages],
        "call_log": result.call_log,
        "raw_responses": result.raw_responses,
        "termination": result.termination,
        "termination_detail": result.termination_detail,
        "final_text": result.final_text,
        "total_turns": result.total_turns,
        "total_usage": {
            "input_tokens": result.total_usage.input_tokens,
            "output_tokens": result.total_usage.output_tokens,
            "total_tokens": result.total_usage.total_tokens,
        },
        "total_latency_ms": result.total_latency_ms,
        "wall_clock_ms": result.wall_clock_ms,
        "provider_retries": result.provider_retries,
        "timestamp_utc": result.timestamp_utc,
    }


def _message_to_dict(m: Message) -> dict:
    d: dict[str, Any] = {"role": m.role}
    if m.text is not None:
        d["text"] = m.text
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "call_id": tc.call_id,
                "name": tc.name,
                "arguments": tc.arguments,
                "raw_arguments": tc.raw_arguments,
                "parse_error": tc.parse_error,
            }
            for tc in m.tool_calls
        ]
    if m.call_id is not None:
        d["call_id"] = m.call_id
    if m.tool_result is not None:
        d["tool_result"] = m.tool_result
    return d


def write_result(result: RunResult, out_dir: str | Path, *, variant: str = "v0", run_idx: int = 0) -> Path:
    """Write a RunResult to the canonical path under `out_dir`.

    Path: `{out_dir}/{provider}/{model_slug}/{task_id}__{variant}__run{N}.json`.
    Creates directories as needed. Writes atomically via a .tmp rename.
    """
    base = Path(out_dir) / result.provider / _slug(result.model)
    base.mkdir(parents=True, exist_ok=True)
    fname = f"{result.task_id}__{variant}__run{run_idx}.json"
    final_path = base / fname
    tmp_path = base / (fname + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result_to_dict(result), f, indent=2, default=str)
    tmp_path.replace(final_path)
    return final_path


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")
