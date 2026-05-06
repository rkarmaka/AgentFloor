#!/usr/bin/env python3
"""CLI for aggregating AgentFloor scores into the headline tables.

Reads .score.json + result.json files under a results directory and prints
per-model x per-tier TCR, capability floor, diagnostic metrics (SDR/THI/LSR/
ERR/ERT), failure breakdown (F1-F7), and cost/efficiency tables.

Usage:

    python runs/run_metrics.py results/
    python runs/run_metrics.py results/ --subset paper_baseline
    python runs/run_metrics.py results/openai_compatible_vllm/
    python runs/run_metrics.py results/ --table tcr
    python runs/run_metrics.py results/ --table floor \\
        --models-yaml sweep_configs/<your-config>.yaml
    python runs/run_metrics.py results/ --format csv --output report/
    python runs/run_metrics.py results/ --format json --output leaderboard.json

Tables:
  tcr          — Per-model x per-tier TCR percentages (default)
  floor        — Capability floor: smallest params_b at >=80% TCR per tier
  diagnostics  — SDR, THI, LSR, ERR, ERT per model
  failures     — F1-F7 distribution per model x tier
  cost         — Mean tokens, latency per model x tier
  all          — All tables (default when format is csv/json)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.metrics import (  # noqa: E402
    add_cis_to_diagnostics,
    add_cis_to_tcr_matrix,
    compute_capability_floor,
    compute_cost_efficiency,
    compute_diagnostics,
    compute_failure_breakdown,
    compute_stubbed_coverage,
    compute_tcr_matrix,
    count_unscored,
    extract_model_params,
    load_entries,
)
from harness.paper_subsets import entry_in_subset  # noqa: E402


TIERS = ["A0", "A", "B", "C", "D", "E"]
FAILURE_CODES = ["F1", "F2", "F2_F5", "F3", "F4", "F5", "F5b", "F6", "F7"]


# ----------------------------------------------------------------------
# Console formatters
# ----------------------------------------------------------------------


def print_tcr_table(tcr_matrix: dict, stubbed: dict | None = None) -> None:
    """Per-model x per-tier TCR percentages.

    When ``stubbed`` is provided, any tier cell where >0 passing runs leaned
    on a stubbed predicate gets a trailing ``*`` (plus a summary footer
    explaining the marker). Surfaces stub coverage alongside TCR so
    readers can see when a passing cell relied on a default-pass check.
    """
    if not tcr_matrix:
        print("(no scored runs)")
        return
    models = sorted(tcr_matrix.keys())
    model_w = max(max(len(m) for m in models), len("model"))

    print()
    print("=" * 80)
    print("TCR Matrix (per-model x per-tier pass rate)")
    if stubbed is not None:
        print("  * = at least one passing run in this cell leaned on a stubbed check")
        print("      (run `--table stubbed` for the full breakdown)")
    print("=" * 80)
    header = f"{'model':<{model_w}}  " + "  ".join(f"{t:>12}" for t in TIERS) + "  overall"
    print(header)
    print("-" * len(header))
    for m in models:
        cells = []
        for t in TIERS:
            cell = tcr_matrix[m].get(t)
            if cell and cell["total"]:
                marker = ""
                if stubbed:
                    sc = (stubbed.get(m) or {}).get(t) or {}
                    if sc.get("passes_with_stubs", 0) > 0:
                        marker = "*"
                ci_lo = cell.get("ci_low")
                ci_hi = cell.get("ci_high")
                if ci_lo is not None and ci_hi is not None:
                    cells.append(f"{100*cell['rate']:>3.0f}%[{100*ci_lo:.0f},{100*ci_hi:.0f}]{marker}")
                else:
                    cells.append(f"{100*cell['rate']:>4.0f}%{marker:<1}      ")
            else:
                cells.append("      —     ")
        overall = tcr_matrix[m].get("overall") or {"rate": 0, "passes": 0, "total": 0}
        ci_lo = overall.get("ci_low")
        ci_hi = overall.get("ci_high")
        if ci_lo is not None and ci_hi is not None:
            ov_str = f"{100*overall['rate']:>3.0f}%[{100*ci_lo:.0f},{100*ci_hi:.0f}]"
        else:
            ov_str = f"{100*overall['rate']:>3.0f}%"
        print(f"{m:<{model_w}}  " + "  ".join(cells) + f"  {ov_str}")
    print()


def print_stubbed_table(stubbed: dict) -> None:
    """Per-model x per-tier stubbed-check coverage of TCR passes.

    For every (model, tier) cell, reports the count of passing runs whose
    verdict leaned on a stubbed predicate, and the set of stubbed names
    observed across those passes. Use together with the TCR matrix to
    judge which scores are robust vs. which are propped up by stubs.
    """
    if not stubbed:
        print("(no scored passing runs to analyze)")
        return
    print()
    print("=" * 80)
    print("Stubbed-Check Coverage of Passing Runs")
    print("=" * 80)
    print("  passes_with_stubs / passes — a nonzero ratio means that some "
          "TCR passes in")
    print("  this cell relied on a predicate currently marked as stubbed. "
          "See doc/metrics.md")
    print("  for the list of stubbed predicates and their replacement "
          "(LLM-judge) plan.")
    print()

    models = sorted(stubbed.keys())
    model_w = max(max(len(m) for m in models), len("model"))
    header = f"{'model':<{model_w}}  " + "  ".join(f"{t:>7}" for t in TIERS) + "  overall"
    print(header)
    print("-" * len(header))
    any_stubbed = False
    for m in models:
        cells = []
        for t in TIERS:
            c = stubbed[m].get(t)
            if not c or not c["passes"]:
                cells.append("    —  ")
                continue
            stub = c["passes_with_stubs"]
            total = c["passes"]
            if stub > 0:
                any_stubbed = True
            cells.append(f"{stub:>2}/{total:<2}  ")
        ov = stubbed[m].get("overall") or {"passes": 0, "passes_with_stubs": 0}
        ov_s = f"{ov['passes_with_stubs']:>2}/{ov['passes']:<2}"
        print(f"{m:<{model_w}}  " + "".join(cells) + f"  {ov_s}")

    if not any_stubbed:
        print()
        print("  (no passing run in any cell leaned on a stubbed predicate)")
        return

    print()
    print("Stubbed predicates observed (by model):")
    for m in models:
        ov = stubbed[m].get("overall") or {}
        names = ov.get("stubs_set") or []
        if names:
            print(f"  {m}:")
            for n in names:
                print(f"    - {n}")
    print()


def print_floor_table(floor: dict) -> None:
    """Capability floor: smallest params_b with >=80% TCR per tier."""
    print()
    print("=" * 80)
    print("Capability Floor (smallest params_b >= 80% TCR per tier)")
    print("=" * 80)
    header = f"{'tier':<6}  {'model':<30}  {'params_b':>10}  {'tcr':>8}"
    print(header)
    print("-" * len(header))
    for t in TIERS:
        cell = floor.get(t)
        if cell is None:
            print(f"{t:<6}  {'(none meets threshold)':<30}  {'—':>10}  {'—':>8}")
        else:
            print(f"{t:<6}  {cell['model']:<30}  {cell['params_b']:>10.1f}  "
                  f"{100*cell['tcr']:>7.0f}%")
    print()


def print_diagnostics_table(diagnostics: dict) -> None:
    """SDR, THI, LSR, ERR, ERT per model.

    Each cell shows ``point[ci_low,ci_high]`` when 95 % bootstrap intervals
    are available (populated by ``add_cis_to_diagnostics``). Cells with <5
    runs (or LSR resamples that all degenerated to inf) show no bracket.
    """
    if not diagnostics:
        print("(no diagnostic data)")
        return
    print()
    print("=" * 80)
    print("Diagnostic Metrics (per-model averages across all runs)")
    print("=" * 80)
    print(f"  SDR  syntax degradation rate  = malformed / total_calls")
    print(f"  THI  tool hallucination index = hallucinated / total_calls")
    print(f"  LSR  loop-to-success ratio    = total_calls / ok_calls (1.0 = perfect)")
    print(f"  ERR  error recovery rate      = recovered / malformed")
    print(f"  ERT  early resignation rate   = fails-by-final_answer / total runs")
    print(f"  Brackets show 95% bootstrap CIs (resampling unit = run, n_boot=10k).")
    print()

    def _pct_cell(point, lo, hi) -> str:
        if point is None:
            return f"{'—':>4}                "
        if lo is not None and hi is not None:
            return f"{100*point:>4.1f}%[{100*lo:>4.1f},{100*hi:>4.1f}]"
        return f"{100*point:>4.1f}%             "

    def _ratio_cell(point, lo, hi) -> str:
        if point is None:
            return f"{'—':>4}              "
        if point == float("inf"):
            return "  inf              "
        if lo is not None and hi is not None:
            return f"{point:>5.2f}[{lo:>4.2f},{hi:>4.2f}]"
        return f"{point:>5.2f}             "

    models = sorted(diagnostics.keys())
    model_w = max(max(len(m) for m in models), len("model"))
    cell_w = 18
    header = (
        f"{'model':<{model_w}}  "
        f"{'SDR':>{cell_w}}  {'THI':>{cell_w}}  {'LSR':>{cell_w}}  "
        f"{'ERR':>{cell_w}}  {'ERT':>{cell_w}}  "
        f"{'n_runs':>7}  {'n_calls':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        d = diagnostics[m]
        cells = [
            _pct_cell(d["sdr"], d.get("sdr_ci_low"), d.get("sdr_ci_high")),
            _pct_cell(d["thi"], d.get("thi_ci_low"), d.get("thi_ci_high")),
            _ratio_cell(d["lsr"], d.get("lsr_ci_low"), d.get("lsr_ci_high")),
            _pct_cell(d["err"], d.get("err_ci_low"), d.get("err_ci_high")),
            _pct_cell(d["ert"], d.get("ert_ci_low"), d.get("ert_ci_high")),
        ]
        print(
            f"{m:<{model_w}}  "
            + "  ".join(cells)
            + f"  {d['n_runs']:>7}  {d['n_calls']:>8}"
        )
    print()


def print_failure_table(failures: dict) -> None:
    """F1-F7 distribution per model x tier (flattened to one row per model, aggregating tiers)."""
    if not failures:
        print("(no failure data)")
        return
    print()
    print("=" * 80)
    print("Failure Mode Distribution (F1-F7 counts per model)")
    print("=" * 80)
    print("  F1 hallucination | F2 malformed   | F2_F5 malformed→resign | F3 amnesia")
    print("  F4 loop          | F5 early resign | F5b plan-no-execute   | F6 wrong tool | F7 partial")
    print()
    models = sorted(failures.keys())
    model_w = max(max(len(m) for m in models), len("model"))
    header = (f"{'model':<{model_w}}  "
              + "  ".join(f"{c:>4}" for c in FAILURE_CODES)
              + f"  {'PASS':>5}  {'TOTAL':>6}")
    print(header)
    print("-" * len(header))
    for m in models:
        totals: dict[str, int] = {c: 0 for c in FAILURE_CODES}
        totals["PASS"] = 0
        totals["INFRA_ERROR"] = 0
        total = 0
        for tier, codes in failures[m].items():
            for code, n in codes.items():
                totals[code] = totals.get(code, 0) + n
                total += n
        fail_cells = "  ".join(f"{totals.get(c,0):>4}" for c in FAILURE_CODES)
        print(f"{m:<{model_w}}  {fail_cells}  {totals['PASS']:>5}  {total:>6}")
    print()


def print_cost_table(cost: dict) -> None:
    """Mean tokens and latency per model x tier."""
    if not cost:
        print("(no cost data)")
        return
    print()
    print("=" * 80)
    print("Cost / Efficiency (per-model averages)")
    print("=" * 80)
    models = sorted(cost.keys())
    # One row per (model, tier) — but aggregate across tiers to keep it compact
    print(f"  {'model':<30}  {'tier':>5}  {'in_tok':>9}  {'out_tok':>9}  "
          f"{'lat_ms':>9}  {'pass/n':>8}  {'tok/pass':>10}")
    print("-" * 90)
    for m in models:
        for tier in TIERS:
            c = cost[m].get(tier)
            if not c or c["n_runs"] == 0:
                continue
            tok_per_pass = c["tokens_per_pass"]
            tpp_s = f"{int(tok_per_pass):>10}" if tok_per_pass != float("inf") else "      inf"
            print(f"  {m:<30}  {tier:>5}  "
                  f"{c['mean_input_tokens']:>9.0f}  {c['mean_output_tokens']:>9.0f}  "
                  f"{c['mean_latency_ms']:>9.0f}  "
                  f"{int(c['n_pass']):>3}/{int(c['n_runs']):<4}  "
                  f"{tpp_s}")
    print()


# ----------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------


def write_tcr_csv(tcr_matrix: dict, path: Path) -> None:
    tier_cols = []
    for t in TIERS:
        tier_cols += [t, f"{t}_ci_low", f"{t}_ci_high"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + tier_cols + [
            "overall_pct", "overall_ci_low", "overall_ci_high",
            "overall_pass", "overall_total",
        ])
        for m in sorted(tcr_matrix.keys()):
            row = [m]
            for t in TIERS:
                cell = tcr_matrix[m].get(t)
                if cell and cell["total"]:
                    row += [
                        f"{cell['rate']:.4f}",
                        f"{cell.get('ci_low', ''):.4f}" if cell.get("ci_low") is not None else "",
                        f"{cell.get('ci_high', ''):.4f}" if cell.get("ci_high") is not None else "",
                    ]
                else:
                    row += ["", "", ""]
            overall = tcr_matrix[m].get("overall") or {"rate": 0, "passes": 0, "total": 0}
            row += [
                f"{overall['rate']:.4f}",
                f"{overall.get('ci_low', ''):.4f}" if overall.get("ci_low") is not None else "",
                f"{overall.get('ci_high', ''):.4f}" if overall.get("ci_high") is not None else "",
                overall["passes"], overall["total"],
            ]
            w.writerow(row)


def write_floor_csv(floor: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "floor_model", "params_b", "tcr"])
        for t in TIERS:
            cell = floor.get(t)
            if cell:
                w.writerow([t, cell["model"], cell["params_b"], f"{cell['tcr']:.4f}"])
            else:
                w.writerow([t, "", "", ""])


def write_diagnostics_csv(diagnostics: dict, path: Path) -> None:
    """Write diagnostics CSV with point estimates and 95 % bootstrap CI bounds.

    Each metric exports three columns: ``<m>``, ``<m>_ci_low``, ``<m>_ci_high``.
    Empty CI cells indicate either fewer than 5 runs or (for LSR only) every
    bootstrap resample degenerated to inf.
    """
    def _fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and v == float("inf"):
            return ""
        return f"{v:.4f}"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["model"]
        for m in ("sdr", "thi", "lsr", "err", "ert"):
            cols += [m, f"{m}_ci_low", f"{m}_ci_high"]
        cols += ["n_runs", "n_calls", "n_malformed", "n_hallucinated"]
        w.writerow(cols)
        for m in sorted(diagnostics.keys()):
            d = diagnostics[m]
            row = [m]
            for metric in ("sdr", "thi", "lsr", "err", "ert"):
                row += [
                    _fmt(d.get(metric)),
                    _fmt(d.get(f"{metric}_ci_low")),
                    _fmt(d.get(f"{metric}_ci_high")),
                ]
            row += [d["n_runs"], d["n_calls"], d["n_malformed"], d["n_hallucinated"]]
            w.writerow(row)


def write_failures_csv(failures: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        codes = FAILURE_CODES + ["PASS", "INFRA_ERROR"]
        w.writerow(["model", "tier"] + codes + ["total"])
        for m in sorted(failures.keys()):
            for t in TIERS:
                cd = failures[m].get(t) or {}
                row = [m, t] + [cd.get(c, 0) for c in codes] + [sum(cd.values())]
                if sum(cd.values()) > 0:
                    w.writerow(row)


def write_stubbed_csv(stubbed: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tier", "passes", "passes_with_stubs", "stub_rate",
                    "stubs"])
        for m in sorted(stubbed.keys()):
            for t in TIERS + ["overall"]:
                c = stubbed[m].get(t)
                if not c or not c.get("passes"):
                    continue
                rate = c.get("rate") or 0.0
                stubs = ";".join(c.get("stubs_set") or [])
                w.writerow([m, t, c["passes"], c["passes_with_stubs"],
                           f"{rate:.4f}", stubs])


def write_cost_csv(cost: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tier", "n_runs", "n_pass",
                    "mean_input_tokens", "mean_output_tokens",
                    "mean_latency_ms", "mean_wall_ms", "tokens_per_pass"])
        for m in sorted(cost.keys()):
            for t in TIERS:
                c = cost[m].get(t)
                if not c or c["n_runs"] == 0:
                    continue
                tpp = c["tokens_per_pass"]
                tpp_s = "" if tpp == float("inf") else f"{tpp:.1f}"
                w.writerow([m, t, c["n_runs"], c["n_pass"],
                           f"{c['mean_input_tokens']:.1f}",
                           f"{c['mean_output_tokens']:.1f}",
                           f"{c['mean_latency_ms']:.1f}",
                           f"{c['mean_wall_ms']:.1f}",
                           tpp_s])


# ----------------------------------------------------------------------
# JSON writer
# ----------------------------------------------------------------------


def write_leaderboard_json(
    tcr_matrix: dict, floor: dict, diagnostics: dict,
    failures: dict, cost: dict, stubbed: dict, path: Path,
) -> None:
    def fix_inf(d):
        if isinstance(d, dict):
            return {k: fix_inf(v) for k, v in d.items()}
        if isinstance(d, list):
            return [fix_inf(x) for x in d]
        if isinstance(d, float) and d == float("inf"):
            return None
        if isinstance(d, set):
            return sorted(d)
        return d

    payload = {
        "tcr_matrix": fix_inf(tcr_matrix),
        "capability_floor": fix_inf(floor),
        "diagnostics": fix_inf(diagnostics),
        "failure_breakdown": fix_inf(failures),
        "cost_efficiency": fix_inf(cost),
        "stubbed_coverage": fix_inf(stubbed),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="Directory containing .json result files")
    ap.add_argument("--table", choices=["tcr", "floor", "diagnostics", "failures", "cost", "stubbed", "all"],
                    default="all", help="Which table(s) to print")
    ap.add_argument("--format", choices=["console", "csv", "json"], default="console")
    ap.add_argument("--output", help="Output dir (for csv) or file path (for json)")
    ap.add_argument("--models-yaml", help="Path to models.yaml for params_b joins (floor needs this)")
    ap.add_argument("--models", help="CSV filter: only these model slugs")
    ap.add_argument(
        "--subset",
        choices=["all", "paper_baseline", "gpt5_extsteps"],
        default="all",
        help="Restrict aggregation to a named paper-facing result slice",
    )
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Pass rate threshold for capability floor (default 0.80)")
    ap.add_argument("--require-scored", action="store_true",
                    help="Skip results without a .score.json companion")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir}")

    models_yaml = None
    if args.models_yaml:
        models_yaml = Path(args.models_yaml)
        if not models_yaml.is_absolute():
            for base in (Path.cwd(), _REPO_ROOT):
                alt = base / args.models_yaml
                if alt.exists():
                    models_yaml = alt
                    break

    print(f"Loading entries from {results_dir}...", file=sys.stderr)
    entries = load_entries(
        results_dir, models_yaml=models_yaml, require_scored=args.require_scored,
    )
    print(f"  loaded {len(entries)} run(s)", file=sys.stderr)

    if args.subset != "all":
        entries = [e for e in entries if entry_in_subset(e, args.subset)]
        print(f"  filtered to {len(entries)} run(s) in subset {args.subset}", file=sys.stderr)

    if args.models:
        wanted = {s.strip() for s in args.models.split(",")}
        entries = [e for e in entries if e["model"] in wanted]
        print(f"  filtered to {len(entries)} run(s) matching --models", file=sys.stderr)

    if not entries:
        raise SystemExit("no entries to aggregate")

    unscored = count_unscored(entries)
    if unscored:
        print(
            f"  WARNING: {unscored} run(s) have no .score.json — they will be "
            f"excluded from TCR/diagnostics/failure tables (aggregators skip "
            f"unscored entries so mid-flight sweeps don't skew numbers). "
            f"Run the evaluator to score them, or pass --require-scored to "
            f"make this an error.",
            file=sys.stderr,
        )

    # Compute all tables
    tcr_matrix = compute_tcr_matrix(entries)
    add_cis_to_tcr_matrix(tcr_matrix, entries)
    model_params = extract_model_params(entries)
    floor = compute_capability_floor(tcr_matrix, model_params, threshold=args.threshold)
    diagnostics = compute_diagnostics(entries)
    add_cis_to_diagnostics(diagnostics, entries)
    failures = compute_failure_breakdown(entries)
    cost = compute_cost_efficiency(entries)
    stubbed = compute_stubbed_coverage(entries)

    # Dispatch by format
    if args.format == "console":
        if args.table in ("tcr", "all"):
            print_tcr_table(tcr_matrix, stubbed=stubbed if stubbed else None)
        if args.table in ("floor", "all"):
            if not model_params:
                print("\n(floor table needs --models-yaml to resolve params_b)")
            else:
                print_floor_table(floor)
        if args.table in ("diagnostics", "all"):
            print_diagnostics_table(diagnostics)
        if args.table in ("failures", "all"):
            print_failure_table(failures)
        if args.table in ("cost", "all"):
            print_cost_table(cost)
        if args.table in ("stubbed", "all"):
            print_stubbed_table(stubbed)

    elif args.format == "csv":
        if not args.output:
            raise SystemExit("--output DIR required for --format csv")
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tcr_csv(tcr_matrix, out_dir / "tcr_matrix.csv")
        write_floor_csv(floor, out_dir / "capability_floor.csv")
        write_diagnostics_csv(diagnostics, out_dir / "diagnostics.csv")
        write_failures_csv(failures, out_dir / "failure_breakdown.csv")
        write_cost_csv(cost, out_dir / "cost_efficiency.csv")
        write_stubbed_csv(stubbed, out_dir / "stubbed_coverage.csv")
        print(f"wrote CSV tables to {out_dir}/", file=sys.stderr)

    elif args.format == "json":
        if not args.output:
            raise SystemExit("--output FILE required for --format json")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_leaderboard_json(
            tcr_matrix, floor, diagnostics, failures, cost, stubbed, out_path,
        )
        print(f"wrote leaderboard JSON to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
