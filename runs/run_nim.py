#!/usr/bin/env python3
"""Run AgentFloor tasks against NVIDIA NIM (hosted) — and probe tool support.

NIM's hosted inference (https://integrate.api.nvidia.com/v1) is OpenAI-
compatible at the chat-completions level, so we slot it into the existing
`OpenAICompatibleProvider` adapter unchanged. Auth is `Bearer nvapi-...`.

Two modes:

1. **Run mode** (default) — same shape as `run_gemini.py`:

       python runs/run_nim.py --task A1
       python runs/run_nim.py --task B/B1 --model meta/llama-3.1-8b-instruct --runs 3 --save

2. **Probe mode** — sends ONE trivial tool-calling request to each model in
   a curated SLM list and classifies the response. This is the "does this
   model actually do native tool calling on NIM?" check that has to pass
   before any model is admitted to a benchmark sweep. NIM (vLLM-backed) can
   silently leak tool-call JSON into text content when its server-side
   parser can't extract a call — that case must be detected before it
   pollutes F2/F5 stats.

       python runs/run_nim.py --probe
       python runs/run_nim.py --probe --probe-models meta/llama-3.1-8b-instruct,qwen/qwen2.5-7b-instruct

   Probe output is a markdown table to stdout, plus a YAML file at
   results/nim_compatibility.yaml for archival.

Reads NVIDIA_API_KEY from .env.local at the repo root (or the environment).
Get a key from https://build.nvidia.com/settings/api-keys.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from harness.providers.base import (  # noqa: E402
    ModelConfig,
    user_message,
)
from harness.schemas import TOOL_SCHEMAS  # noqa: E402
from harness.runner import run_task, write_result  # noqa: E402


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Curated SLM probe list. Verified model IDs from build.nvidia.com / NIM
# API reference docs (research dated 2026-04-11). Update as the catalog
# rotates — NVIDIA deprecates entries without long warning windows.
DEFAULT_PROBE_MODELS = [
    # Strong baselines — should pass
    "meta/llama-3.1-8b-instruct",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    # Risk models — Phi tool-calling support unconfirmed in NIM docs
    "microsoft/phi-3-mini-4k-instruct",
    "microsoft/phi-3-medium-4k-instruct",
    # Risk models — Qwen tool-calling support unconfirmed in NIM docs
    # (vLLM has Qwen parsers but NIM may or may not expose them)
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwen2.5-coder-7b-instruct",
    "qwen/qwen2.5-coder-32b-instruct",
]


# ----------------------------------------------------------------------
# Env loading and shared setup
# ----------------------------------------------------------------------


def load_env_local(path: Path) -> None:
    """Tiny .env loader — KEY=VALUE per line, no quoting tricks."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def get_api_key() -> str:
    load_env_local(_REPO_ROOT / ".env.local")
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not key:
        raise SystemExit(
            "NVIDIA_API_KEY not set.\n"
            "  1. Sign up: https://build.nvidia.com (free with NVIDIA Developer Program)\n"
            "  2. Generate a key: https://build.nvidia.com/settings/api-keys\n"
            "  3. Add NVIDIA_API_KEY=nvapi-... to .env.local at the repo root"
        )
    return key


def make_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        backend="nim",
        base_url=NIM_BASE_URL,
        api_key=api_key,
    )


# ----------------------------------------------------------------------
# Run mode (parallel structure to run_gemini.py)
# ----------------------------------------------------------------------


def resolve_task_path(task_arg: str) -> Path:
    """Accept either 'A1', 'A/A1', or a full path. Returns absolute path."""
    p = Path(task_arg)
    if p.is_absolute() and p.exists():
        return p
    tasks_root = _HERE / "tasks"
    cand = tasks_root / task_arg
    if cand.with_suffix(".yaml").exists():
        return cand.with_suffix(".yaml")
    if cand.exists():
        return cand
    bare = task_arg.removesuffix(".yaml")
    if bare.startswith("A0"):
        level = "A0"
    else:
        level = bare[0]
    cand = tasks_root / level / f"{bare}.yaml"
    if cand.exists():
        return cand
    raise SystemExit(f"could not find task {task_arg!r} under {tasks_root}")


def print_run_summary(result, run_idx: int) -> None:
    print(f"\n--- run {run_idx}: {result.task_id} on {result.model} ---")
    print(f"  termination     : {result.termination}")
    print(f"  detail          : {result.termination_detail}")
    print(f"  turns           : {result.total_turns}")
    print(f"  tool calls      : {len(result.call_log)}")
    print(
        f"  tokens          : in={result.total_usage.input_tokens} "
        f"out={result.total_usage.output_tokens}"
    )
    print(f"  latency_ms      : {result.total_latency_ms}")
    print(f"  wall_clock_ms   : {result.wall_clock_ms}")
    print(f"  provider_retries: {result.provider_retries}")
    if result.call_log:
        print("  call log:")
        for i, c in enumerate(result.call_log):
            flags = []
            if c.get("is_hallucinated"):
                flags.append("HALLU")
            if c.get("is_malformed"):
                flags.append("MALFORMED")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            print(
                f"    {i+1}. {c['tool_name']}({c.get('args')}) "
                f"-> {c['response_status']}{flag_str}"
            )
    if result.final_text:
        snippet = result.final_text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  final_text      : {snippet}")


def cmd_run(args) -> None:
    api_key = get_api_key()
    provider = make_provider(api_key)
    config = ModelConfig(model=args.model, max_output_tokens=args.max_output_tokens)
    task_path = resolve_task_path(args.task)

    print(f"task    : {task_path.relative_to(_REPO_ROOT)}")
    print(f"model   : {args.model}")
    print(f"runs    : {args.runs}")
    print(f"variant : {args.variant}")

    for i in range(args.runs):
        result = run_task(task_path, provider, config, variant_id=args.variant)
        print_run_summary(result, run_idx=i)
        if args.save:
            out = write_result(result, args.results_dir, variant=args.variant, run_idx=i)
            print(f"  saved -> {out}")


# ----------------------------------------------------------------------
# Probe mode
# ----------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of one one-call tool-support probe.

    `status` is one of:
      native        — provider returned non-empty tool_calls (the goal)
      text_leaked   — tool_calls empty, but text content looks like a
                      JSON tool call (G-014: NIM/vLLM parser failure)
      no_tools      — tool_calls empty, text is normal English (model
                      may not support tool calling, or refused)
      error         — request raised an exception
    """

    model: str
    status: str
    tool_calls: int
    stop_reason: str | None
    text_excerpt: str | None
    notes: str
    latency_ms: int


# Tool name we'll use for the probe. Picked because lookup_record is the
# simplest single-arg required-field schema in TOOL_SCHEMAS — minimizes
# the chance that a model fails the probe due to argument complexity.
_PROBE_TOOL_NAME = "lookup_record"
_PROBE_PROMPT = (
    "Look up the product record with ID 'P104' using the lookup_record tool. "
    "Call the tool — do not answer from memory."
)


def _classify_response(turn, tool_name: str) -> tuple[str, str]:
    """Classify a probe response shape. Returns (status, notes).

    G-014 detector: looks for "leaked tool call" — tool_calls empty AND
    text content looks like JSON containing the tool name. We do NOT parse
    or attempt to recover the call (D3 forbids that); we only flag the
    shape so the operator knows the model failed silently.
    """
    if turn.tool_calls:
        return "native", f"emitted {len(turn.tool_calls)} tool_call(s) natively"

    text = (turn.text or "").strip()
    if not text:
        return "no_tools", "empty response, no tool_calls"

    # Heuristic G-014 detector: tool name appears in text AND text has
    # JSON-shaped structure. Conservative — needs both signals.
    has_tool_name = tool_name in text
    has_json_shape = bool(re.search(r"[\{\[].*[\}\]]", text, re.DOTALL))
    if has_tool_name and has_json_shape:
        return (
            "text_leaked",
            "tool_calls empty BUT text contains tool name + JSON shape (G-014)",
        )
    if has_tool_name:
        return (
            "text_leaked",
            "tool_calls empty BUT text mentions tool name (weak G-014 signal)",
        )
    return "no_tools", "tool_calls empty, text is normal prose"


def probe_model(
    provider: OpenAICompatibleProvider,
    model: str,
    *,
    max_output_tokens: int = 256,
) -> ProbeResult:
    """Send one tool-calling probe request to a single model."""
    schema = TOOL_SCHEMAS[_PROBE_TOOL_NAME]
    tools_payload = provider.format_tools([schema])
    config = ModelConfig(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
    )
    messages = [user_message(_PROBE_PROMPT)]

    try:
        turn = provider.complete(
            system=(
                "You are an agent that completes tasks by calling tools. "
                "When a tool is provided, call it using the provider's "
                "native tool-calling interface."
            ),
            messages=messages,
            tools=tools_payload,
            config=config,
        )
    except Exception as e:
        return ProbeResult(
            model=model,
            status="error",
            tool_calls=0,
            stop_reason=None,
            text_excerpt=None,
            notes=f"{type(e).__name__}: {str(e)[:200]}",
            latency_ms=0,
        )

    status, notes = _classify_response(turn, _PROBE_TOOL_NAME)
    excerpt = (turn.text or "").strip().replace("\n", " ")[:140] or None
    return ProbeResult(
        model=model,
        status=status,
        tool_calls=len(turn.tool_calls),
        stop_reason=turn.stop_reason,
        text_excerpt=excerpt,
        notes=notes,
        latency_ms=turn.latency_ms,
    )


def print_probe_table(results: list[ProbeResult]) -> None:
    """Print a fixed-width table of probe outcomes to stdout."""
    headers = ("Model", "Status", "Calls", "Latency", "Notes")
    rows = [
        (
            r.model,
            r.status,
            str(r.tool_calls),
            f"{r.latency_ms}ms",
            r.notes[:60],
        )
        for r in results
    ]
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    print()

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"Summary: {summary}")


def write_probe_yaml(results: list[ProbeResult], path: Path) -> None:
    """Persist probe results as a small YAML compatibility matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# NIM tool-calling compatibility matrix",
        f"# Generated by runs/run_nim.py --probe at {timestamp}",
        "# text_leaked status: server-side tool parser failed; the call JSON",
        "# appears in text content instead of in the tool_calls list.",
        f"generated_at: {timestamp}",
        f"probe_tool: {_PROBE_TOOL_NAME}",
        f"probe_prompt: {_PROBE_PROMPT!r}",
        "results:",
    ]
    for r in results:
        lines.append(f"  - model: {r.model}")
        lines.append(f"    status: {r.status}")
        lines.append(f"    tool_calls: {r.tool_calls}")
        lines.append(f"    stop_reason: {r.stop_reason!r}")
        lines.append(f"    latency_ms: {r.latency_ms}")
        lines.append(f"    notes: {r.notes!r}")
        if r.text_excerpt:
            lines.append(f"    text_excerpt: {r.text_excerpt!r}")
    path.write_text("\n".join(lines) + "\n")


def cmd_probe(args) -> None:
    api_key = get_api_key()
    provider = make_provider(api_key)
    models = (
        [m.strip() for m in args.probe_models.split(",") if m.strip()]
        if args.probe_models
        else DEFAULT_PROBE_MODELS
    )
    print(f"Probing {len(models)} model(s) on {NIM_BASE_URL}")
    print(f"Probe tool: {_PROBE_TOOL_NAME}")

    results: list[ProbeResult] = []
    for i, model in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] {model} ...", end=" ", flush=True)
        r = probe_model(provider, model, max_output_tokens=args.max_output_tokens)
        results.append(r)
        print(f"{r.status} ({r.tool_calls} call(s), {r.latency_ms}ms)")

    print_probe_table(results)

    out_path = Path(args.probe_out) if args.probe_out else _REPO_ROOT / "results" / "nim_compatibility.yaml"
    write_probe_yaml(results, out_path)
    print(f"Wrote {out_path.relative_to(_REPO_ROOT)}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Run AgentFloor tasks on NVIDIA NIM, or probe NIM model tool support.",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Probe mode: send one tool-calling request to each model in --probe-models.",
    )
    # Run-mode args
    ap.add_argument(
        "--task",
        help="(run mode) Task id ('A1'), tier-qualified id ('A/A1'), or full path.",
    )
    ap.add_argument(
        "--model",
        default="nvidia/nvidia-nemotron-nano-9b-v2",
        help="(run mode) NIM model id (default: nvidia/nvidia-nemotron-nano-9b-v2).",
    )
    ap.add_argument("--runs", type=int, default=1, help="(run mode) Number of runs (default: 1).")
    ap.add_argument(
        "--save",
        action="store_true",
        help="(run mode) Write results to results/{provider}/{model_slug}/...json.",
    )
    ap.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "results"),
        help="(run mode) Output directory when --save is set.",
    )
    ap.add_argument(
        "--variant",
        default="v0",
        choices=["v0", "v1", "v2", "v3", "v4", "v5"],
        help="(run mode) Prompt variant. v0 is the original; v1-v5 require a sibling <task>.variants.yaml.",
    )
    # Shared
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Per-turn output token cap (default: 1024 for run mode, 256 used for probe).",
    )
    # Probe-mode args
    ap.add_argument(
        "--probe-models",
        help="(probe mode) Comma-separated list of model ids to probe. "
        f"Default: {len(DEFAULT_PROBE_MODELS)} curated SLMs.",
    )
    ap.add_argument(
        "--probe-out",
        help="(probe mode) Path to write the YAML compatibility matrix. "
        "Default: results/nim_compatibility.yaml.",
    )

    args = ap.parse_args()

    if args.probe:
        cmd_probe(args)
        return
    if not args.task:
        ap.error("--task is required in run mode (or pass --probe for probe mode)")
    cmd_run(args)


if __name__ == "__main__":
    main()
