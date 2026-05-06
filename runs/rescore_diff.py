#!/usr/bin/env python3
"""Re-score a results directory with the current scorer and diff against
existing .score.json files.

Use this after a scorer change to see how reported numbers shift before
overwriting old scores. The script reads each `*.json` result file under
`results_root`, runs the current evaluator on it, and prints:

  - per-run TCR flips (PASS<->FAIL), with which check leg flipped
  - per-(model, tier) TCR delta vs the existing scores
  - aggregate counts (PASS->FAIL, FAIL->PASS, unchanged)

By default this is read-only: it writes nothing. Pass `--write` to overwrite
the .score.json files with the re-scored verdicts after you've inspected
the diff.

Usage:

    python runs/rescore_diff.py results/
    python runs/rescore_diff.py results/openai_compatible_vllm/
    python runs/rescore_diff.py results/ --write
    python runs/rescore_diff.py results/ --tasks tasks/ --details
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.evaluator import (  # noqa: E402
    evaluate_result_file,
    score_to_dict,
    write_score_alongside,
)


def _load_existing_score(result_path: Path) -> dict | None:
    score_path = result_path.with_suffix(".score.json")
    if not score_path.exists():
        return None
    try:
        with open(score_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _verdict_legs(score_dict: dict) -> dict[str, bool]:
    """Pull the per-leg pass booleans from a score dict."""
    return {
        "final_answer": bool((score_dict.get("final_answer") or {}).get("passed")),
        "submission":   bool((score_dict.get("submission")   or {}).get("passed")),
        "trajectory":   bool((score_dict.get("trajectory")   or {}).get("passed")),
        "forbidden":    all(
            bool(r.get("passed")) for r in (score_dict.get("forbidden") or [])
        ),
        "tcr":          bool(score_dict.get("tcr_pass")),
    }


def _tier_of(task_id: str) -> str:
    if task_id.startswith("A0"):
        return "A0"
    if task_id and task_id[0] in "ABCDE":
        return task_id[0]
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-score result files and diff against existing .score.json verdicts.",
    )
    ap.add_argument("results_root", help="Directory containing result JSON files (recurses).")
    ap.add_argument(
        "--tasks",
        default=str(_REPO_ROOT / "tasks"),
        help="Tasks directory (default: tasks/).",
    )
    ap.add_argument(
        "--details",
        action="store_true",
        help="Print every flipped run, not just the summary.",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="After diffing, overwrite the existing .score.json files with the "
        "re-scored verdicts. Default is read-only.",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    tasks_dir = Path(args.tasks)

    if not results_root.exists():
        print(f"results_root {results_root} does not exist", file=sys.stderr)
        return 1

    flips: list[dict] = []          # one entry per run whose tcr_pass flipped
    leg_flips_only: list[dict] = [] # tcr unchanged but a leg flipped
    new_only: list[Path] = []        # no existing .score.json
    errored: list[tuple[Path, str]] = []
    n_total = 0
    n_unchanged = 0
    pass_to_fail = 0
    fail_to_pass = 0

    by_cell_old: dict[tuple[str, str], list[bool]] = defaultdict(list)
    by_cell_new: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for path in sorted(results_root.rglob("*.json")):
        if path.name.endswith(".score.json"):
            continue
        n_total += 1
        try:
            new_score = evaluate_result_file(path, tasks_dir)
        except Exception as e:  # noqa: BLE001
            errored.append((path, f"{type(e).__name__}: {e}"))
            continue
        new_dict = score_to_dict(new_score)
        new_legs = _verdict_legs(new_dict)
        cell = (new_score.model, _tier_of(new_score.task_id))
        by_cell_new[cell].append(new_legs["tcr"])

        existing = _load_existing_score(path)
        if existing is None:
            new_only.append(path)
            continue

        old_legs = _verdict_legs(existing)
        by_cell_old[cell].append(old_legs["tcr"])

        if old_legs["tcr"] == new_legs["tcr"]:
            # tcr unchanged; check if any leg flipped (interesting context)
            leg_changes = {
                k: (old_legs[k], new_legs[k])
                for k in ("final_answer", "submission", "trajectory", "forbidden")
                if old_legs[k] != new_legs[k]
            }
            if leg_changes:
                leg_flips_only.append(
                    {
                        "path": str(path.relative_to(results_root)),
                        "task": new_score.task_id,
                        "model": new_score.model,
                        "variant": new_score.variant,
                        "leg_changes": leg_changes,
                    }
                )
            else:
                n_unchanged += 1
            continue

        # tcr flipped — figure out which legs drove it
        leg_changes = {
            k: (old_legs[k], new_legs[k])
            for k in ("final_answer", "submission", "trajectory", "forbidden")
            if old_legs[k] != new_legs[k]
        }
        flips.append(
            {
                "path": str(path.relative_to(results_root)),
                "task": new_score.task_id,
                "model": new_score.model,
                "variant": new_score.variant,
                "old_tcr": old_legs["tcr"],
                "new_tcr": new_legs["tcr"],
                "leg_changes": leg_changes,
            }
        )
        if old_legs["tcr"] and not new_legs["tcr"]:
            pass_to_fail += 1
        else:
            fail_to_pass += 1

    # ---- Summary ----------------------------------------------------------
    print()
    print("=" * 72)
    print("Re-scoring diff summary")
    print("=" * 72)
    print(f"  total result files     : {n_total}")
    print(f"  errored                : {len(errored)}")
    print(f"  no prior .score.json   : {len(new_only)}")
    print(f"  diffed                 : {n_total - len(errored) - len(new_only)}")
    print(f"  unchanged              : {n_unchanged}")
    print(f"  TCR flips (total)      : {len(flips)}")
    print(f"    PASS -> FAIL         : {pass_to_fail}")
    print(f"    FAIL -> PASS         : {fail_to_pass}")
    print(f"  leg changed, TCR same  : {len(leg_flips_only)}")
    print()

    if flips:
        # Which leg drove each flip?
        leg_counter: Counter[str] = Counter()
        for f in flips:
            for leg in f["leg_changes"]:
                leg_counter[leg] += 1
        print("TCR-flip drivers (which leg changed):")
        for leg, n in leg_counter.most_common():
            print(f"  {leg:<14} {n}")
        print()

    if by_cell_old:
        # Per-(model, tier) TCR delta. Only cells that exist in BOTH old and new.
        print("Per-(model, tier) TCR delta  (positive = up after fix)")
        print("-" * 72)
        cells = sorted(set(by_cell_old) & set(by_cell_new))
        worst = []
        for c in cells:
            old_runs = by_cell_old[c]
            new_runs = by_cell_new[c]
            if not old_runs or not new_runs:
                continue
            old_pct = 100.0 * sum(old_runs) / len(old_runs)
            new_pct = 100.0 * sum(new_runs) / len(new_runs)
            delta = new_pct - old_pct
            if abs(delta) >= 0.01:
                worst.append((delta, c, old_pct, new_pct, len(old_runs)))
        worst.sort(key=lambda x: x[0])
        print(f"  {'model':<35} {'tier':>5}  {'old':>6}  {'new':>6}  {'delta':>7}  {'n':>4}")
        for delta, (model, tier), old_pct, new_pct, n in worst:
            print(
                f"  {model:<35} {tier:>5}  "
                f"{old_pct:>5.1f}% {new_pct:>5.1f}%  {delta:+7.1f}  {n:>4}"
            )
        if not worst:
            print("  (no cells changed)")
        print()

    if args.details and flips:
        print("Per-run flips:")
        for f in flips[:200]:
            change_str = ", ".join(
                f"{leg}: {a}->{b}" for leg, (a, b) in f["leg_changes"].items()
            )
            print(
                f"  [{f['old_tcr']!s:5} -> {f['new_tcr']!s:5}]  "
                f"{f['model']:<30} {f['task']:<5} {f['variant']:<3}  "
                f"{change_str}"
            )
        if len(flips) > 200:
            print(f"  ... ({len(flips) - 200} more, rerun without --details to omit)")
        print()

    if errored:
        print(f"Errored files ({len(errored)}):")
        for p, err in errored[:20]:
            print(f"  {p.relative_to(results_root)}: {err}")
        if len(errored) > 20:
            print(f"  ... ({len(errored) - 20} more)")
        print()

    if args.write:
        # Re-score a second time to get TaskScore objects, then persist.
        # (We could persist on the first pass, but the diff pass throws
        # away the TaskScore — keep the loops separate for clarity.)
        n_written = 0
        for path in sorted(results_root.rglob("*.json")):
            if path.name.endswith(".score.json"):
                continue
            try:
                score = evaluate_result_file(path, tasks_dir)
            except Exception:  # noqa: BLE001
                continue
            write_score_alongside(score)
            n_written += 1
        print(f"Wrote {n_written} .score.json files (overwriting prior versions).")

    return 0 if not errored else 1


if __name__ == "__main__":
    raise SystemExit(main())
