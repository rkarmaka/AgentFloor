#!/usr/bin/env python3
"""Run a single AgentFloor task against Gemini.

Usage:

    python runs/run_gemini.py --task A1
    python runs/run_gemini.py --task B/B1 --model gemini-2.5-pro
    python runs/run_gemini.py --task A1 --runs 3 --save

Reads GEMINI_API_KEY from .env.local at the repo root (or the environment).
Default model is gemini-2.5-flash. With --save, results are written under
results/{provider}/{model_slug}/...json via runner.write_result.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.providers import get_provider  # noqa: E402
from harness.providers.base import ModelConfig  # noqa: E402
from harness.runner import run_task, write_result  # noqa: E402


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


def resolve_task_path(task_arg: str) -> Path:
    """Accept either 'A1' or 'A/A1' or a full path. Returns absolute path."""
    p = Path(task_arg)
    if p.is_absolute() and p.exists():
        return p
    tasks_root = _REPO_ROOT / "tasks"
    # full relative form like "A/A1" or "A/A1.yaml"
    cand = tasks_root / task_arg
    if cand.with_suffix(".yaml").exists():
        return cand.with_suffix(".yaml")
    if cand.exists():
        return cand
    # bare id like "A1" — infer level from prefix
    bare = task_arg.removesuffix(".yaml")
    if bare.startswith("A0") or bare.startswith("A01") or bare.startswith("A02"):
        level = "A0"
    else:
        level = bare[0]
    cand = tasks_root / level / f"{bare}.yaml"
    if cand.exists():
        return cand
    raise SystemExit(f"could not find task {task_arg!r} under {tasks_root}")


def print_summary(result, run_idx: int) -> None:
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


def main():
    ap = argparse.ArgumentParser(description="Run one AgentFloor task on Gemini.")
    ap.add_argument(
        "--task",
        required=True,
        help="Task id ('A1'), tier-qualified id ('A/A1'), or full path to a task YAML.",
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model id (default: gemini-2.5-flash).",
    )
    ap.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1).")
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Per-turn output token cap (default: 1024).",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Write results to results/{provider}/{model_slug}/...json.",
    )
    ap.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "results"),
        help="Output directory when --save is set (default: results/).",
    )
    ap.add_argument(
        "--variant",
        default="v0",
        choices=["v0", "v1", "v2", "v3", "v4", "v5"],
        help="Prompt variant. v0 is the original; v1-v5 require a sibling <task>.variants.yaml.",
    )
    args = ap.parse_args()

    load_env_local(_REPO_ROOT / ".env.local")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add it to .env.local or export it."
        )

    task_path = resolve_task_path(args.task)
    print(f"task    : {task_path.relative_to(_REPO_ROOT)}")
    print(f"model   : {args.model}")
    print(f"runs    : {args.runs}")
    print(f"variant : {args.variant}")

    provider = get_provider("gemini", api_key=api_key)
    config = ModelConfig(model=args.model, max_output_tokens=args.max_output_tokens)

    for i in range(args.runs):
        result = run_task(task_path, provider, config, variant_id=args.variant)
        print_summary(result, run_idx=i)
        if args.save:
            out = write_result(result, args.results_dir, variant=args.variant, run_idx=i)
            print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
