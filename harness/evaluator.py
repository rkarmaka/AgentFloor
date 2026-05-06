"""AgentFloor evaluator: turn a RunResult into a TaskScore.

The evaluator wires four checker families together to produce one
pass/fail TCR verdict per (task, model, variant, run_idx) combination:

  1. final_answer  — does the model's free text contain the expected answer?
  2. submission    — does the submit_decision payload match the gold state?
  3. trajectory    — did the call_log follow the required tool sequence
                     and satisfy all named predicates?
  4. forbidden     — did the run avoid every declared forbidden behavior?

`tcr_pass` is the AND of all four. The detailed per-check results are
preserved in TaskScore.details so the metrics layer (SDR/THI/LSR/ERT/ERR)
can read them later.

Three entry points:
  evaluate_run(result, task)               — in-memory dicts
  evaluate_result_file(path, tasks_dir)    — load from disk
  evaluate_directory(root, tasks_dir)      — score every result file under root

CLI:
  python -m harness.evaluator results/openai_compatible_vllm/
  python -m harness.evaluator results/openai_compatible_vllm/ --tasks tasks --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .eval_checks import CheckResult  # noqa: E402
from .eval_checks.final_answer import check_final_answer  # noqa: E402
from .eval_checks.forbidden import check_all_forbidden  # noqa: E402
from .eval_checks.submission import check_submission  # noqa: E402
from .eval_checks.trajectory import check_trajectory  # noqa: E402


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class TaskScore:
    """Per-run TCR verdict and the four checker breakdowns.

    `tcr_pass` is the headline boolean; the metrics layer will aggregate
    these into per-model TCR percentages.
    """

    task_id: str
    level: str
    model: str
    variant: str
    run_idx: int

    tcr_pass: bool

    final_answer: CheckResult
    submission: CheckResult
    trajectory: CheckResult
    forbidden: list[CheckResult] = field(default_factory=list)

    # Per-trajectory-flag breakdowns for the metrics layer
    trajectory_subchecks: dict[str, CheckResult] = field(default_factory=dict)

    summary: str = ""

    # Reference back to source for provenance
    result_path: str | None = None
    task_path: str | None = None


# ----------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------


def evaluate_run(
    result: dict,
    task: dict,
    *,
    variant: str = "v0",
    run_idx: int = 0,
) -> TaskScore:
    """Score one run against its task spec.

    `result` is the RunResult-as-dict (the JSON that runner.write_result
    produces). `task` is the parsed task YAML.

    Order matters: submission is evaluated first because the
    final_answer checker needs to know whether the submission passed
    (the E1 `bundle_validator_passes` case defers to submission validity).
    """
    oracle = task.get("oracle_evaluator") or {}
    gold_state = task.get("gold_state") or {}
    trace_reqs = task.get("trace_requirements") or {}
    forbidden_list = trace_reqs.get("forbidden_behaviors") or []

    # Stash gold_state and the full task spec on the result so downstream
    # checkers can access them without changing the public API.
    result_with_gold = {**result, "_gold_state": gold_state, "_task": task}

    # 1. Submission
    submission_result = check_submission(result_with_gold, oracle)

    # 2. Final answer (knows whether submission passed)
    final_answer_result = check_final_answer(
        result_with_gold,
        oracle.get("final_answer_check") or {},
        submission_passed=submission_result.passed if oracle.get("submission_check", {}).get("required") else None,
    )

    # 3. Trajectory
    trajectory_top, trajectory_subchecks = check_trajectory(
        result_with_gold,
        oracle,
        gold_state,
        trace_reqs,
    )

    # 4. Forbidden behaviors
    forbidden_results = check_all_forbidden(result_with_gold, forbidden_list, gold_state)

    # AND everything for the headline TCR verdict
    tcr_pass = (
        final_answer_result.passed
        and submission_result.passed
        and trajectory_top.passed
        and all(r.passed for r in forbidden_results)
    )

    summary = _build_summary(
        tcr_pass,
        final_answer_result,
        submission_result,
        trajectory_top,
        forbidden_results,
    )

    return TaskScore(
        task_id=task.get("task_id", result.get("task_id", "?")),
        level=task.get("level", result.get("level", "?")),
        model=result.get("model", "?"),
        variant=variant,
        run_idx=run_idx,
        tcr_pass=tcr_pass,
        final_answer=final_answer_result,
        submission=submission_result,
        trajectory=trajectory_top,
        forbidden=forbidden_results,
        trajectory_subchecks=trajectory_subchecks,
        summary=summary,
    )


def _build_summary(
    tcr_pass: bool,
    final_answer: CheckResult,
    submission: CheckResult,
    trajectory: CheckResult,
    forbidden: list[CheckResult],
) -> str:
    """One-line human-readable verdict naming the proximate failure(s)."""
    if tcr_pass:
        return "PASS"
    failures = []
    if not final_answer.passed:
        failures.append(f"final_answer({final_answer.reason})")
    if not submission.passed:
        failures.append(f"submission({submission.reason})")
    if not trajectory.passed:
        failures.append(f"trajectory({trajectory.reason})")
    bad_fbs = [r for r in forbidden if not r.passed]
    if bad_fbs:
        failures.append(
            f"forbidden({len(bad_fbs)}: {bad_fbs[0].details.get('behavior_type', '?')})"
        )
    return "FAIL — " + "; ".join(failures)


# ----------------------------------------------------------------------
# File-based entry points
# ----------------------------------------------------------------------


def evaluate_result_file(
    result_path: Path,
    tasks_dir: Path,
) -> TaskScore:
    """Load a result JSON file + its matching task YAML and score it.

    The variant id is parsed from the result filename
    (`<task>__<variant>__runN.json`). For instance variants (`i*`) the
    task spec must include the instance-specific overrides — without them
    the scorer would compare the model's answer against the canonical
    gold_state and report a false negative for any instance whose target
    differs from v0. Route through `load_task` so both v* and i* paths
    apply their respective overrides.
    """
    result_path = Path(result_path)
    with open(result_path) as f:
        result = json.load(f)

    task_id = result["task_id"]
    level = result.get("level", "")
    task_path = _resolve_task_path(task_id, level, tasks_dir)

    # Parse variant and run_idx from filename: {task_id}__{variant}__run{N}.json
    variant = "v0"
    run_idx = 0
    stem = result_path.stem  # strips .json
    parts = stem.split("__")
    if len(parts) >= 3:
        variant = parts[1]
        last = parts[2]
        if last.startswith("run"):
            try:
                run_idx = int(last[3:])
            except ValueError:
                pass

    # Apply variant overrides via load_task. For v0/v* prompt variants the
    # gold_state and oracle_evaluator are unchanged; for i* instance
    # variants the instance file's overrides deep-merge into the task
    # before scoring.
    from .runner import load_task
    task = load_task(task_path, variant_id=variant).raw

    score = evaluate_run(result, task, variant=variant, run_idx=run_idx)
    score.result_path = str(result_path)
    score.task_path = str(task_path)
    return score


def _resolve_task_path(task_id: str, level: str, tasks_dir: Path) -> Path:
    """Find the task YAML for `task_id` under `tasks_dir`.

    Tries `tasks_dir/{level}/{task_id}.yaml` first, then falls back to
    inferring the level from the task_id prefix (A0/A/B/C/D/E).
    """
    tasks_dir = Path(tasks_dir)
    if level:
        cand = tasks_dir / level / f"{task_id}.yaml"
        if cand.exists():
            return cand
    # Infer level from prefix: A01-A05 → A0; otherwise first character
    if task_id.startswith(("A0",)) and len(task_id) >= 2 and task_id[1] == "0":
        inferred_level = "A0"
    else:
        inferred_level = task_id[0] if task_id else "?"
    cand = tasks_dir / inferred_level / f"{task_id}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"could not find task YAML for {task_id!r} in {tasks_dir}"
    )


def evaluate_directory(
    results_root: Path,
    tasks_dir: Path,
    *,
    fail_fast: bool = False,
) -> list[TaskScore]:
    """Evaluate every result file under `results_root` (skipping .score.json).

    By default, files that fail to score are skipped with an stderr warning
    and the count surfaces in a final summary — the denominator change is
    visible to the caller. Pass `fail_fast=True` to raise on the first
    failure instead.
    """
    results_root = Path(results_root)
    scores: list[TaskScore] = []
    errors: list[tuple[Path, BaseException]] = []
    n_files = 0
    for path in sorted(results_root.rglob("*.json")):
        if path.name.endswith(".score.json"):
            continue
        n_files += 1
        try:
            score = evaluate_result_file(path, tasks_dir)
        except Exception as e:  # noqa: BLE001
            if fail_fast:
                raise
            errors.append((path, e))
            print(
                f"  ERROR scoring {path.relative_to(results_root)}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        scores.append(score)
    if errors:
        print(
            f"\nWARNING: {len(errors)} of {n_files} result file(s) failed to score "
            f"and were excluded from the scored set "
            f"(scored N = {len(scores)}). "
            f"Pass --fail-fast to abort on the first error instead.\n",
            file=sys.stderr,
        )
    return scores


# ----------------------------------------------------------------------
# Score serialization
# ----------------------------------------------------------------------


def score_to_dict(score: TaskScore) -> dict:
    """JSON-serializable dict for writing alongside result files."""
    def cr_dict(cr: CheckResult) -> dict:
        return {"passed": cr.passed, "reason": cr.reason, "details": cr.details}

    return {
        "task_id": score.task_id,
        "level": score.level,
        "model": score.model,
        "variant": score.variant,
        "run_idx": score.run_idx,
        "tcr_pass": score.tcr_pass,
        "summary": score.summary,
        "final_answer": cr_dict(score.final_answer),
        "submission": cr_dict(score.submission),
        "trajectory": cr_dict(score.trajectory),
        "forbidden": [cr_dict(r) for r in score.forbidden],
        "trajectory_subchecks": {k: cr_dict(v) for k, v in score.trajectory_subchecks.items()},
        "result_path": score.result_path,
        "task_path": score.task_path,
    }


def write_score_alongside(score: TaskScore) -> Path:
    """Write `<task_id>__<variant>__run<N>.score.json` next to the result.

    Atomic via .tmp rename.
    """
    if not score.result_path:
        raise ValueError("score.result_path is required to write alongside")
    rp = Path(score.result_path)
    out_path = rp.with_name(rp.stem + ".score.json")
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(score_to_dict(score), f, indent=2, default=str)
    tmp.replace(out_path)
    return out_path


# ----------------------------------------------------------------------
# Pretty-printing
# ----------------------------------------------------------------------


def print_tcr_table(scores: list[TaskScore]) -> None:
    """Print a per-model × per-task TCR table to stdout.

    Cells: ✓ pass, ✗ fail, blank if no run for that combo. Per-model
    TCR percentage in the rightmost column. Per-task pass rate in the
    bottom row.
    """
    if not scores:
        print("No scores to display.")
        return

    # Build the matrix: model → task_id → TaskScore
    models: list[str] = sorted({s.model for s in scores})
    tasks: list[str] = sorted({s.task_id for s in scores}, key=_task_sort_key)
    matrix: dict[str, dict[str, TaskScore]] = {m: {} for m in models}
    for s in scores:
        # If multiple runs per (model, task) — keep the worst (any-fail = fail)
        existing = matrix[s.model].get(s.task_id)
        if existing is None or (existing.tcr_pass and not s.tcr_pass):
            matrix[s.model][s.task_id] = s

    model_col_w = max(len(m) for m in models)
    model_col_w = max(model_col_w, len("model"))
    task_col_w = max(max(len(t) for t in tasks), 4)

    # Header
    header = " " * (model_col_w + 2) + "  ".join(f"{t:>{task_col_w}}" for t in tasks) + "  TCR"
    print()
    print(header)
    print("-" * len(header))

    # Per-model rows
    for m in models:
        passes = 0
        total = 0
        cells = []
        for t in tasks:
            s = matrix[m].get(t)
            if s is None:
                cells.append(" " * task_col_w)
                continue
            total += 1
            if s.tcr_pass:
                passes += 1
                cells.append(f"{'✓':>{task_col_w}}")
            else:
                cells.append(f"{'✗':>{task_col_w}}")
        tcr_pct = f"{100 * passes / total:.0f}%" if total else "—"
        print(f"{m:<{model_col_w}}  " + "  ".join(cells) + f"  {tcr_pct} ({passes}/{total})")

    # Per-task pass rate footer
    footer_cells = []
    for t in tasks:
        passes = sum(
            1 for m in models if matrix[m].get(t) and matrix[m][t].tcr_pass
        )
        total = sum(1 for m in models if matrix[m].get(t))
        if total == 0:
            footer_cells.append(" " * task_col_w)
        else:
            footer_cells.append(f"{passes}/{total}".rjust(task_col_w))
    print("-" * len(header))
    print(f"{'pass/runs':<{model_col_w}}  " + "  ".join(footer_cells))
    print()


def _task_sort_key(task_id: str) -> tuple:
    """Sort A0 < A < B < ... and within tier by numeric suffix."""
    if task_id.startswith("A0"):
        tier_order = 0
        rest = task_id[2:]
    else:
        tier_order = "ABCDE".index(task_id[0]) + 1 if task_id and task_id[0] in "ABCDE" else 99
        rest = task_id[1:]
    try:
        num = int(rest)
    except ValueError:
        num = 999
    return (tier_order, num, task_id)


def print_failure_breakdown(scores: list[TaskScore]) -> None:
    """For every failing run, print the proximate cause."""
    fails = [s for s in scores if not s.tcr_pass]
    if not fails:
        return
    print(f"Failure breakdown ({len(fails)} fails):")
    for s in fails:
        print(f"  {s.model[:30]:<30} {s.task_id:<5}  {s.summary}")
    print()


# ----------------------------------------------------------------------
# Per-variant table — for analyzing variant variance
# ----------------------------------------------------------------------


def print_by_variant_table(scores: list[TaskScore]) -> None:
    """Print one table per model showing TCR by (task, variant).

    Each cell is `passes/runs` across all runs of that (task, variant).
    Right column is the per-task mean across all variants. Bottom row
    is the per-variant mean across all tasks. Used to surface which
    axes are most brittle for which model.
    """
    if not scores:
        print("No scores to display.")
        return

    models = sorted({s.model for s in scores})
    tasks = sorted({s.task_id for s in scores}, key=_task_sort_key)
    variants = sorted({s.variant for s in scores}, key=_variant_sort_key)

    # Build matrix: model → task → variant → list[TaskScore]
    matrix: dict[str, dict[str, dict[str, list[TaskScore]]]] = {}
    for s in scores:
        matrix.setdefault(s.model, {}).setdefault(s.task_id, {}).setdefault(
            s.variant, []
        ).append(s)

    task_col_w = max(max(len(t) for t in tasks), 4)
    variant_col_w = max(max(len(v) for v in variants), 5)

    for m in models:
        print()
        print(f"=== {m} ===")
        # Header
        header_cells = [f"{v:>{variant_col_w}}" for v in variants]
        print(
            " " * (task_col_w + 2)
            + "  ".join(header_cells)
            + f"  {'mean':>6}"
        )
        print("-" * (task_col_w + 2 + (variant_col_w + 2) * len(variants) + 8))

        # Per-task rows
        per_variant_totals = {v: [0, 0] for v in variants}  # variant → [passes, total]
        for t in tasks:
            row_cells = []
            row_passes = 0
            row_total = 0
            for v in variants:
                runs = matrix.get(m, {}).get(t, {}).get(v) or []
                if not runs:
                    row_cells.append(" " * variant_col_w)
                    continue
                p = sum(1 for r in runs if r.tcr_pass)
                row_cells.append(f"{p}/{len(runs)}".rjust(variant_col_w))
                row_passes += p
                row_total += len(runs)
                per_variant_totals[v][0] += p
                per_variant_totals[v][1] += len(runs)
            mean_str = f"{100 * row_passes / row_total:.0f}%" if row_total else "—"
            print(
                f"{t:<{task_col_w}}  "
                + "  ".join(row_cells)
                + f"  {mean_str:>6}"
            )

        # Per-variant footer
        footer_cells = []
        all_passes = 0
        all_total = 0
        for v in variants:
            p, total = per_variant_totals[v]
            all_passes += p
            all_total += total
            if total == 0:
                footer_cells.append(" " * variant_col_w)
            else:
                pct = f"{100 * p / total:.0f}%"
                footer_cells.append(pct.rjust(variant_col_w))
        all_mean = f"{100 * all_passes / all_total:.0f}%" if all_total else "—"
        print(
            f"{'mean':<{task_col_w}}  "
            + "  ".join(footer_cells)
            + f"  {all_mean:>6}"
        )


def print_axes_breakdown(scores: list[TaskScore], task_dir: Path | None = None) -> None:
    """For every (model, task) where any variant failed, list which axes broke.

    Reads the variant axis labels from the matching .variants.yaml files
    so we can attribute failures to a specific perturbation type. If
    task_dir is None, attempts to infer from the score's task_path.
    """
    if not scores:
        return

    # Build a (model, task) → {variant: failed_run_count} map
    fail_counts: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    score_paths: dict[tuple[str, str], str] = {}
    for s in scores:
        key = (s.model, s.task_id)
        v_map = fail_counts.setdefault(key, {})
        passes, total = v_map.get(s.variant, (0, 0))
        v_map[s.variant] = (
            passes + (1 if s.tcr_pass else 0),
            total + 1,
        )
        if s.task_path and key not in score_paths:
            score_paths[key] = s.task_path

    print()
    print("Axis breakdown (variants that failed at least one run):")
    any_failures = False
    for (model, task), v_map in sorted(fail_counts.items()):
        failures = [
            (v, p, t) for v, (p, t) in sorted(v_map.items(), key=lambda kv: _variant_sort_key(kv[0]))
            if p < t
        ]
        if not failures:
            continue
        any_failures = True
        # Resolve axis labels by reading the variants.yaml
        axis_map = _load_variant_axis_map(score_paths.get((model, task)))
        bits = []
        for v, p, t in failures:
            axis = axis_map.get(v, "?")
            bits.append(f"{v}({axis}): {t - p}/{t}")
        print(f"  {model[:35]:<35} {task:<5}  " + "; ".join(bits))
    if not any_failures:
        print("  (no failures detected)")
    print()


def _load_variant_axis_map(task_path_str: str | None) -> dict[str, str]:
    """Read <task>.variants.yaml and return {variant_id: axis_label}."""
    if not task_path_str:
        return {}
    task_path = Path(task_path_str)
    variants_path = task_path.with_name(task_path.stem + ".variants.yaml")
    if not variants_path.exists():
        return {}
    try:
        data = yaml.safe_load(variants_path.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return {}
    out: dict[str, str] = {}
    for entry in data.get("variants") or []:
        if isinstance(entry, dict):
            vid = entry.get("id")
            axis = entry.get("axis")
            if isinstance(vid, str) and isinstance(axis, str):
                out[vid] = axis
    return out


def _variant_sort_key(variant_id: str) -> tuple:
    """Sort v0 < v1 < v2 < ... ; unknown ids last."""
    if variant_id.startswith("v") and variant_id[1:].isdigit():
        return (0, int(variant_id[1:]))
    return (1, variant_id)


# ----------------------------------------------------------------------
# Variant validation (no-op runner — just confirms files parse)
# ----------------------------------------------------------------------


def validate_all_variants(tasks_dir: Path) -> tuple[int, int, list[str]]:
    """Walk every task YAML and try to load each declared variant.

    Validates both prompt variants (v*) and instance variants (i*). Skips
    task-adjacent variant files (`*.variants.yaml`, `*.instance_variants.yaml`)
    so they aren't themselves treated as tasks.

    Returns (passed_count, failed_count, error_messages).
    """
    from .variants.loader import (
        InstanceVariantNotFoundError,
        VariantNotFoundError,
        list_instance_variant_ids,
        list_variant_ids,
        load_instance_overrides,
        load_variant_text,
    )

    tasks_dir = Path(tasks_dir)
    task_files = sorted(
        p for p in tasks_dir.glob("*/*.yaml")
        if p.name not in ("registry.yaml", "task_template.yaml")
        and not p.name.endswith("variants.yaml")
    )
    passed = 0
    failed = 0
    errors: list[str] = []
    for tp in task_files:
        # Prompt variants (v1..v5)
        for vid in list_variant_ids(tp):
            try:
                text = load_variant_text(tp, vid)
                if not text or not text.strip():
                    raise ValueError(f"empty variant text for {vid}")
                passed += 1
            except (VariantNotFoundError, KeyError, ValueError) as e:
                failed += 1
                errors.append(f"{tp.parent.name}/{tp.stem}/{vid}: {type(e).__name__}: {e}")
        # Instance variants (i1..i5) — only present on tasks chosen for the
        # input-instance ablation
        for vid in list_instance_variant_ids(tp):
            try:
                text, overrides = load_instance_overrides(tp, vid)
                if not text or not text.strip():
                    raise ValueError(f"empty instance text for {vid}")
                if not isinstance(overrides, dict):
                    raise ValueError(f"non-dict overrides for {vid}")
                passed += 1
            except (InstanceVariantNotFoundError, KeyError, ValueError) as e:
                failed += 1
                errors.append(f"{tp.parent.name}/{tp.stem}/{vid}: {type(e).__name__}: {e}")
    return passed, failed, errors


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Score AgentFloor result files against task gold states.",
    )
    ap.add_argument(
        "results_root",
        nargs="?",
        help="Directory containing result JSON files (recurses). Omit if --validate-variants.",
    )
    ap.add_argument(
        "--tasks",
        default=str(_REPO_ROOT / "tasks"),
        help="Tasks directory (default: tasks/).",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Don't write .score.json files; only print the table.",
    )
    ap.add_argument(
        "--show-failures",
        action="store_true",
        help="Print a per-failure breakdown after the table.",
    )
    ap.add_argument(
        "--by-variant",
        action="store_true",
        help="Print a per-(model, task, variant) breakdown table instead of "
        "the default aggregated view. Use this for variant-variance analysis.",
    )
    ap.add_argument(
        "--show-axes",
        action="store_true",
        help="After --by-variant table, list which axes broke for each "
        "(model, task) where any variant failed.",
    )
    ap.add_argument(
        "--validate-variants",
        action="store_true",
        help="No-op mode: walk every task YAML, attempt to load each declared "
        "variant, report parse errors. Doesn't read any result files.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort scoring on the first per-file error rather than skipping "
        "and continuing. Useful in CI to catch silently-dropped files.",
    )
    args = ap.parse_args()

    # --validate-variants is its own short-circuit mode
    if args.validate_variants:
        passed, failed, errors = validate_all_variants(Path(args.tasks))
        print(f"Validated {passed + failed} variant(s) across all tasks.")
        print(f"  passed: {passed}")
        print(f"  failed: {failed}")
        if errors:
            print()
            print("Errors:")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)
        return

    if not args.results_root:
        ap.error("results_root is required (or pass --validate-variants)")

    scores = evaluate_directory(
        Path(args.results_root), Path(args.tasks), fail_fast=args.fail_fast,
    )
    if not scores:
        print("No scores produced. Check that result files exist under the given root.")
        return

    if args.by_variant:
        print_by_variant_table(scores)
        if args.show_axes:
            print_axes_breakdown(scores, Path(args.tasks))
    else:
        print_tcr_table(scores)
        if args.show_failures:
            print_failure_breakdown(scores)

    if not args.no_write:
        for s in scores:
            if s.result_path:
                write_score_alongside(s)
        print(f"Wrote {len(scores)} .score.json files.")


if __name__ == "__main__":
    main()
