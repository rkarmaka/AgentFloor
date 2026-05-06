#!/usr/bin/env python3
"""Run AgentFloor tasks against a local vLLM server — and probe tool support.

vLLM exposes an OpenAI-compatible HTTP server (`vllm serve <model>` →
http://localhost:8000/v1), so we slot it into the existing
`OpenAICompatibleProvider` adapter. Unlike NIM/Together, **vLLM serves
exactly one model per server process** — there is no model routing key
in the request. To probe multiple models you start a new server with a
different `--model` argument and re-run this script.

Two modes:

1. **Run mode** — same shape as run_nim.py:

       python runs/run_vllm.py --task A1
       python runs/run_vllm.py --task B/B1 --runs 3 --save

   `--model` is auto-detected from /v1/models if not provided.

2. **Probe mode** — sends one trivial tool-calling request to the
   currently-served model and appends one row to the compatibility
   matrix. Run once per model after each server restart:

       python runs/run_vllm.py --probe                    # probes the loaded model
       python runs/run_vllm.py --probe --model qwen/...   # forces a model id

   Output appends to results/vllm_compatibility.yaml so the matrix
   accumulates across sessions.

No API key required. Pass `--base-url` to point at a non-localhost
server, `--host`/`--port` for the standard local case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
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


DEFAULT_BASE_URL = "http://localhost:8000/v1"


# ----------------------------------------------------------------------
# Server discovery
# ----------------------------------------------------------------------


def detect_loaded_model(base_url: str, timeout: float = 5.0) -> str | None:
    """Query /v1/models on the vLLM server and return the loaded model id.

    vLLM serves exactly one model per process, so /v1/models always
    returns a single-entry list. If the server isn't reachable or the
    response shape is unexpected, return None — the caller will fail
    loudly with a clearer message than a deep stack trace.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    items = data.get("data") or []
    if not items:
        return None
    return items[0].get("id")


def make_provider(base_url: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        backend="vllm",
        base_url=base_url,
        api_key="dummy",  # vLLM ignores; OpenAI SDK requires *some* value
    )


# ----------------------------------------------------------------------
# Run mode
# ----------------------------------------------------------------------


def resolve_task_path(task_arg: str) -> Path:
    """Accept either 'A1', 'A/A1', or a full path."""
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


def cmd_run(args, model: str) -> None:
    provider = make_provider(args.base_url)
    config = ModelConfig(model=model, max_output_tokens=args.max_output_tokens)
    task_path = resolve_task_path(args.task)

    print(f"task     : {task_path.relative_to(_REPO_ROOT)}")
    print(f"model    : {model}")
    print(f"base_url : {args.base_url}")
    print(f"runs     : {args.runs}")
    print(f"variant  : {args.variant}")

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
      text_leaked   — tool_calls empty, text contains tool name + JSON or
                      Python-call shape (server-side parser failure)
      no_tools      — tool_calls empty, text is normal English
      error         — request raised an exception
    """

    model: str
    status: str
    tool_calls: int
    stop_reason: str | None
    text_excerpt: str | None
    notes: str
    latency_ms: int


_PROBE_TOOL_NAME = "lookup_record"
_PROBE_PROMPT = (
    "Look up the product record with ID 'P104' using the lookup_record tool. "
    "Call the tool — do not answer from memory."
)

# Tightened text-leak detector compared to run_nim.py: also catches
# Python-call shapes like `lookup_record('P104')` that the original NIM
# probe missed on Phi-3 Mini.
_JSON_SHAPE_RE = re.compile(r"[\{\[].*[\}\]]", re.DOTALL)
_PYCALL_SHAPE_RE = re.compile(rf"\b{re.escape(_PROBE_TOOL_NAME)}\s*\(")


def _classify_response(turn, tool_name: str) -> tuple[str, str]:
    if turn.tool_calls:
        return "native", f"emitted {len(turn.tool_calls)} tool_call(s) natively"

    text = (turn.text or "").strip()
    if not text:
        return "no_tools", "empty response, no tool_calls"

    has_tool_name = tool_name in text
    has_json_shape = bool(_JSON_SHAPE_RE.search(text))
    has_pycall_shape = bool(_PYCALL_SHAPE_RE.search(text))

    if has_tool_name and (has_json_shape or has_pycall_shape):
        shape = "JSON" if has_json_shape else "Python-call"
        return (
            "text_leaked",
            f"tool_calls empty BUT text contains tool name + {shape} shape",
        )
    if has_tool_name:
        return (
            "text_leaked",
            "tool_calls empty BUT text mentions tool name (weak leak signal)",
        )
    return "no_tools", "tool_calls empty, text is normal prose"


def probe_model(
    provider: OpenAICompatibleProvider,
    model: str,
    *,
    max_output_tokens: int = 256,
) -> ProbeResult:
    """Send one tool-calling probe request."""
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


def print_probe_row(r: ProbeResult) -> None:
    """Print one probe result in human-readable form."""
    print()
    print(f"  model        : {r.model}")
    print(f"  status       : {r.status}")
    print(f"  tool_calls   : {r.tool_calls}")
    print(f"  stop_reason  : {r.stop_reason}")
    print(f"  latency_ms   : {r.latency_ms}")
    print(f"  notes        : {r.notes}")
    if r.text_excerpt:
        print(f"  text_excerpt : {r.text_excerpt}")


def append_probe_yaml(result: ProbeResult, path: Path) -> None:
    """Append one probe row to the running compatibility matrix.

    Format: each entry is a YAML list item under `results:`. We do
    not parse and re-emit existing content (would require pyyaml as a
    write dependency); we append fresh YAML at the end. If the file
    doesn't exist yet, write the header first.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# vLLM tool-calling compatibility matrix\n"
            f"# First entry written by runs/run_vllm.py --probe at {timestamp}\n"
            f"probe_tool: {_PROBE_TOOL_NAME}\n"
            f"probe_prompt: {_PROBE_PROMPT!r}\n"
            "results:\n"
        )

    with open(path, "a") as f:
        f.write(f"  - model: {result.model}\n")
        f.write(f"    probed_at: {timestamp}\n")
        f.write(f"    status: {result.status}\n")
        f.write(f"    tool_calls: {result.tool_calls}\n")
        f.write(f"    stop_reason: {result.stop_reason!r}\n")
        f.write(f"    latency_ms: {result.latency_ms}\n")
        f.write(f"    notes: {result.notes!r}\n")
        if result.text_excerpt:
            f.write(f"    text_excerpt: {result.text_excerpt!r}\n")


def cmd_probe(args, model: str) -> None:
    provider = make_provider(args.base_url)
    print(f"Probing model: {model}")
    print(f"  base_url: {args.base_url}")
    print(f"  tool    : {_PROBE_TOOL_NAME}")

    result = probe_model(provider, model, max_output_tokens=args.max_output_tokens)
    print_probe_row(result)

    out_path = (
        Path(args.probe_out)
        if args.probe_out
        else _REPO_ROOT / "results" / "vllm_compatibility.yaml"
    )
    append_probe_yaml(result, out_path)
    print(f"\nAppended to {out_path.relative_to(_REPO_ROOT)}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Run AgentFloor tasks against a local vLLM server, or probe its tool support.",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Probe mode: send one tool-calling request to the loaded model.",
    )
    ap.add_argument(
        "--task",
        help="(run mode) Task id ('A1'), tier-qualified id ('A/A1'), or full path.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. If omitted, auto-detected from {base_url}/models.",
    )
    ap.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"vLLM server base URL (default: {DEFAULT_BASE_URL}).",
    )
    ap.add_argument("--runs", type=int, default=1, help="(run mode) Number of runs.")
    ap.add_argument(
        "--save",
        action="store_true",
        help="(run mode) Persist results to results/{provider}/{model_slug}/...json.",
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
        help="(run mode) Prompt variant to use. v0 is the original example_prompt; "
        "v1-v5 require a sibling <task_id>.variants.yaml file. Default: v0.",
    )
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Per-turn output token cap (default 1024 for run, 256 used for probe).",
    )
    ap.add_argument(
        "--probe-out",
        help="(probe mode) Path to append probe results to. "
        "Default: results/vllm_compatibility.yaml.",
    )

    args = ap.parse_args()

    # Resolve the model id once: explicit --model wins, otherwise
    # auto-detect from the server.
    model = args.model or detect_loaded_model(args.base_url)
    if not model:
        raise SystemExit(
            f"Could not determine model id.\n"
            f"  Tried GET {args.base_url}/models — no response or empty result.\n"
            f"  Either start a vLLM server (`vllm serve <model>`) or pass --model explicitly."
        )

    if args.probe:
        cmd_probe(args, model)
        return
    if not args.task:
        ap.error("--task is required in run mode (or pass --probe for probe mode)")
    cmd_run(args, model)


if __name__ == "__main__":
    main()
