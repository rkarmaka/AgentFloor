"""Aggregation metrics for the AgentFloor benchmark.

The evaluator (evaluator.py) scores individual runs into .score.json files.
This module aggregates those per-run scores plus the underlying result JSONs
into the headline tables:

  1. compute_tcr_matrix()      — per-model x per-tier TCR (% pass rate)
  2. compute_capability_floor()  — smallest params_b >= 80% TCR per tier
  3. compute_diagnostics()     — SDR, THI, LSR, ERR, ERT per model
  4. compute_failure_breakdown() — F1-F7 distribution per model x tier
  5. compute_cost_efficiency() — tokens, latency per (model, tier)

All functions are pure — no file I/O. The CLI wrapper (run_metrics.py) is
responsible for loading files and formatting output (console/CSV/JSON).

Metric formulas follow doc/metrics.md:
  SDR = sum(is_malformed) / sum(len(call_log)) across runs
  THI = sum(is_hallucinated) / sum(len(call_log))
  LSR = sum(len(call_log)) / sum(successful_calls) where status=="ok"
  ERR = malformed calls self-corrected on the next call / total malformed
  ERT = runs that terminated early with wrong answer / total runs

Failure classification priority (doc/metrics.md, matches evaluation/metrics.py:357):
  F4 loop            — termination == "step_budget_exhausted"
  F1 hallucination   — any is_hallucinated in call_log
  F2 malformed       — any is_malformed in call_log (or malformed_terminal_call)
  F5 early resign    — termination in {final_answer, malformed_terminal_call}
                       AND not tcr_pass
  F6 wrong tool      — trajectory check failed but no F1/F2
  F7 partial         — some checker passed, not all
  F3 context amnesia — default for unexplained failures
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Dataclass shapes (plain dicts; keep dependency surface minimal)
# ----------------------------------------------------------------------
# Each "run entry" is a dict with these keys, paired from result.json + score.json:
#
#   {
#     "task_id": str, "level": str, "model": str, "variant": str, "run_idx": int,
#     "tcr_pass": bool, "summary": str,
#     "final_answer_passed": bool, "submission_passed": bool,
#     "trajectory_passed": bool, "forbidden_all_passed": bool,
#     "call_log": list[dict], "termination": str, "total_turns": int,
#     "total_usage": {input_tokens, output_tokens, total_tokens},
#     "total_latency_ms": int, "wall_clock_ms": int,
#     "params_b": float | None, "tier": str | None,  # from models.yaml join
#   }


# ----------------------------------------------------------------------
# TCR matrix (model x tier)
# ----------------------------------------------------------------------


def compute_tcr_matrix(entries: list[dict]) -> dict[str, dict[str, dict[str, int | float]]]:
    """Per-model, per-tier TCR rates.

    Returns: {model_slug: {tier: {passes: int, total: int, rate: float}}}
    plus a synthetic "overall" tier aggregating across tiers for each model.

    Entries with `tcr_pass is None` are skipped (they have no companion
    .score.json — they are unscored runs, not failures). Counting them
    as fails would silently skew leaderboard numbers when a sweep is
    mid-flight or when the evaluator ran on only a subset of results.
    """
    scored = [e for e in entries if e.get("tcr_pass") is not None]
    out: dict[str, dict[str, dict[str, int | float]]] = defaultdict(lambda: defaultdict(lambda: {"passes": 0, "total": 0}))
    for e in scored:
        tier = _tier_of(e)
        cell = out[e["model"]][tier]
        cell["total"] += 1
        if e.get("tcr_pass"):
            cell["passes"] += 1
    # Add overall per model and fill rate
    for model, by_tier in out.items():
        total_pass = sum(c["passes"] for c in by_tier.values())
        total_all = sum(c["total"] for c in by_tier.values())
        by_tier["overall"] = {"passes": total_pass, "total": total_all}
        for cell in by_tier.values():
            cell["rate"] = (cell["passes"] / cell["total"]) if cell["total"] else 0.0
    return {m: dict(v) for m, v in out.items()}


def bootstrap_ci(
    values: list[bool],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a binary pass rate.

    `values` is a list of True/False outcomes. Returns (ci_low, ci_high)
    at the (1-alpha) confidence level. Deterministic under fixed seed.

    For cells with <5 observations, returns (0.0, 1.0) — the interval is
    uninformative but honest about the uncertainty.
    """
    n = len(values)
    if n < 5:
        return (0.0, 1.0)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1 - alpha / 2)) - 1
    return (means[lo_idx], means[min(hi_idx, len(means) - 1)])


def add_cis_to_tcr_matrix(
    tcr_matrix: dict,
    entries: list[dict],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> None:
    """Mutate tcr_matrix cells to include ci_low, ci_high fields.

    Bootstraps over individual runs within each (model, tier) cell.
    """
    by_model_tier: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e.get("tcr_pass") is None:
            continue
        tier = _tier_of(e)
        by_model_tier[e["model"]][tier].append(bool(e["tcr_pass"]))
        by_model_tier[e["model"]]["overall"].append(bool(e["tcr_pass"]))

    for model, tiers in tcr_matrix.items():
        for tier, cell in tiers.items():
            values = by_model_tier.get(model, {}).get(tier, [])
            lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
            cell["ci_low"] = lo
            cell["ci_high"] = hi


def count_unscored(entries: list[dict]) -> int:
    """Return the number of entries that lack a .score.json companion."""
    return sum(1 for e in entries if e.get("tcr_pass") is None)


# ----------------------------------------------------------------------
# Diagnostic-metric bootstrap CIs (SDR / THI / LSR / ERR / ERT)
# ----------------------------------------------------------------------
#
# The diagnostic metrics in compute_diagnostics() are ratios of sums across
# runs, not means of per-run binary outcomes — so bootstrap_ci() doesn't
# apply directly. The two helpers below split each run into a small dict of
# raw counts ("ingredients"); the bootstrap then resamples those count-dicts
# and recomputes the ratio per resample. Resampling unit = run (matches TCR
# convention).
#
# The math in _diagnostics_from_ingredients() mirrors compute_diagnostics()
# exactly. test_diagnostics_from_ingredients_matches_compute_diagnostics in
# the smoke suite asserts the two paths agree.


_DIAG_METRICS = ("sdr", "thi", "lsr", "err", "ert")


def _diagnostic_ingredients(entry: dict) -> dict[str, int]:
    """Reduce a single run entry to the count fields the diagnostic metrics need.

    Mirrors the per-run loop body in compute_diagnostics(). Keeping the
    extraction in one place lets the bootstrap operate on cheap dicts
    instead of replaying the full call_log per resample.
    """
    call_log = entry.get("call_log") or []
    n_malformed = 0
    n_hallucinated = 0
    n_ok = 0
    n_recovered = 0
    malformed_indices: list[int] = []
    for i, c in enumerate(call_log):
        if c.get("is_malformed"):
            n_malformed += 1
            malformed_indices.append(i)
        if c.get("is_hallucinated"):
            n_hallucinated += 1
        if c.get("response_status") == "ok":
            n_ok += 1
    for idx in malformed_indices:
        mal_tool = call_log[idx].get("tool_name")
        for nxt in call_log[idx + 1:]:
            if nxt.get("tool_name") != mal_tool:
                continue
            if (nxt.get("response_status") == "ok"
                    and not nxt.get("is_malformed")):
                n_recovered += 1
            break
    term = entry.get("termination")
    ert_flag = (term in ("final_answer", "malformed_terminal_call")
                and not entry.get("tcr_pass"))
    return {
        "n_calls": len(call_log),
        "n_malformed": n_malformed,
        "n_hallucinated": n_hallucinated,
        "n_ok": n_ok,
        "n_recovered": n_recovered,
        "ert_flag": int(bool(ert_flag)),
    }


def _diagnostics_from_ingredients(rows: list[dict]) -> dict[str, float | None]:
    """Compute SDR/THI/LSR/ERR/ERT from a list of per-run ingredient dicts.

    Ratio formulas mirror compute_diagnostics() — keep these in sync.

    Empty cells (no runs, no calls, no errors) return None for the
    metrics whose denominator is undefined, instead of a zero or "1.0"
    that would visually flatter the model. ERT (numerator over n_runs)
    is None when n_runs==0 and 0.0 otherwise.
    """
    if not rows:
        return {"sdr": None, "thi": None, "lsr": None, "err": None, "ert": None}
    total_calls = sum(r["n_calls"] for r in rows)
    total_malformed = sum(r["n_malformed"] for r in rows)
    total_hallucinated = sum(r["n_hallucinated"] for r in rows)
    total_ok = sum(r["n_ok"] for r in rows)
    total_recovered = sum(r["n_recovered"] for r in rows)
    early_resign = sum(r["ert_flag"] for r in rows)
    return {
        # Per-call rates: undefined when no calls were made.
        "sdr": (total_malformed / total_calls) if total_calls else None,
        "thi": (total_hallucinated / total_calls) if total_calls else None,
        # Calls per ok: inf when calls but no oks; undefined when neither.
        "lsr": (
            (total_calls / total_ok) if total_ok
            else (float("inf") if total_calls else None)
        ),
        # Recovery rate: undefined when there were no malformed calls to recover from.
        "err": (total_recovered / total_malformed) if total_malformed else None,
        # Per-run rate: well-defined whenever rows is non-empty.
        "ert": early_resign / len(rows),
    }


def add_cis_to_diagnostics(
    diagnostics: dict[str, dict[str, float]],
    entries: list[dict],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> None:
    """Mutate diagnostics to include `*_ci_low` / `*_ci_high` for each metric.

    Adds ten new fields per model: {sdr,thi,lsr,err,ert} × {ci_low, ci_high}.
    Cells with fewer than 5 scored runs get None on both ends — the
    interval would be uninformative. Bootstrap samples that produce
    LSR=inf are filtered before percentile selection, since infinity
    breaks ordering; if every sample is degenerate the CI is None/None.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("tcr_pass") is None:
            continue
        by_model[e["model"]].append(_diagnostic_ingredients(e))

    for model, rows in by_model.items():
        if model not in diagnostics:
            continue
        if len(rows) < 5:
            for m in _DIAG_METRICS:
                diagnostics[model][f"{m}_ci_low"] = None
                diagnostics[model][f"{m}_ci_high"] = None
            continue
        rng = random.Random(seed)
        samples: dict[str, list[float]] = {m: [] for m in _DIAG_METRICS}
        for _ in range(n_boot):
            sample = rng.choices(rows, k=len(rows))
            d = _diagnostics_from_ingredients(sample)
            for m in _DIAG_METRICS:
                v = d[m]
                if v == float("inf"):
                    continue
                samples[m].append(v)
        for m in _DIAG_METRICS:
            vals = samples[m]
            if not vals:
                diagnostics[model][f"{m}_ci_low"] = None
                diagnostics[model][f"{m}_ci_high"] = None
                continue
            vals.sort()
            lo_idx = int(len(vals) * alpha / 2)
            hi_idx = int(len(vals) * (1 - alpha / 2)) - 1
            diagnostics[model][f"{m}_ci_low"] = vals[lo_idx]
            diagnostics[model][f"{m}_ci_high"] = vals[min(hi_idx, len(vals) - 1)]


def _tier_of(entry: dict) -> str:
    """Extract tier letter (A0, A, B, C, D, E) from a run entry."""
    lvl = entry.get("level") or ""
    if lvl:
        return lvl
    tid = entry.get("task_id", "")
    if tid.startswith("A0"):
        return "A0"
    return tid[0] if tid and tid[0] in "ABCDE" else "?"


# ----------------------------------------------------------------------
# Capability floor
# ----------------------------------------------------------------------


def compute_capability_floor(
    tcr_matrix: dict,
    model_params: dict[str, float],
    *,
    threshold: float = 0.80,
) -> dict[str, dict[str, Any]]:
    """For each tier, find the smallest params_b model meeting `threshold` TCR.

    Returns: {tier: {model: str, params_b: float, tcr: float} | None}
    Missing tiers map to None when no model clears the threshold.
    """
    # Collect (tier, model, params_b, rate) triples
    by_tier: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    for model, tiers in tcr_matrix.items():
        params = model_params.get(model)
        if params is None:
            continue
        for tier, cell in tiers.items():
            if tier == "overall":
                continue
            if cell["total"] == 0:
                continue
            by_tier[tier].append((params, model, cell["rate"]))

    floor: dict[str, dict[str, Any] | None] = {}
    all_tiers = ["A0", "A", "B", "C", "D", "E"]
    for tier in all_tiers:
        candidates = [(p, m, r) for (p, m, r) in by_tier.get(tier, []) if r >= threshold]
        if not candidates:
            floor[tier] = None
            continue
        candidates.sort(key=lambda x: x[0])  # smallest params_b first
        p, m, r = candidates[0]
        floor[tier] = {"model": m, "params_b": p, "tcr": r}
    return floor


# ----------------------------------------------------------------------
# Diagnostic metrics (SDR, THI, LSR, ERR, ERT)
# ----------------------------------------------------------------------


def compute_diagnostics(entries: list[dict]) -> dict[str, dict[str, float]]:
    """Per-model diagnostic metrics.

    Returns: {model: {sdr, thi, lsr, err, ert, n_runs, n_calls}}

    Skips entries with `tcr_pass is None` (unscored) — ERT needs the
    pass/fail verdict to count early-resignation, and counting unscored
    runs in SDR/THI/LSR denominators understates rates on an in-flight
    sweep.
    """
    per_model: dict[str, dict[str, float]] = {}
    # Group entries by model (skip unscored — no verdict available)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("tcr_pass") is None:
            continue
        by_model[e["model"]].append(e)

    for model, runs in by_model.items():
        total_calls = 0
        total_malformed = 0
        total_hallucinated = 0
        total_ok = 0
        total_recovered = 0
        runs_early_resign_fail = 0

        for r in runs:
            call_log = r.get("call_log") or []
            total_calls += len(call_log)
            malformed_indices = []
            for i, c in enumerate(call_log):
                if c.get("is_malformed"):
                    total_malformed += 1
                    malformed_indices.append(i)
                if c.get("is_hallucinated"):
                    total_hallucinated += 1
                if c.get("response_status") == "ok":
                    total_ok += 1
            # ERR: self-correction on the NEXT call to the same tool.
            # Spec (doc/metrics.md): "Fraction of malformed calls where
            # the model self-corrected on the next turn without harness
            # intervention." We operationalize "next turn" as "the next
            # call to the same tool with no intervening successful call
            # to that tool" — this rules out the pathological case where
            # a model wanders for many steps before happening to get the
            # tool right, which shouldn't count as active recovery.
            for idx in malformed_indices:
                mal_tool = call_log[idx].get("tool_name")
                for nxt in call_log[idx + 1:]:
                    if nxt.get("tool_name") != mal_tool:
                        continue
                    # First call to same tool after the malformed one:
                    # it either recovered (ok + not malformed) or not.
                    if (nxt.get("response_status") == "ok"
                            and not nxt.get("is_malformed")):
                        total_recovered += 1
                    break
            # ERT: terminated "final_answer" (or malformed_terminal) and did not pass
            term = r.get("termination")
            if term in ("final_answer", "malformed_terminal_call") and not r.get("tcr_pass"):
                runs_early_resign_fail += 1

        per_model[model] = {
            # Per-call rates: undefined when no calls were made.
            "sdr": (total_malformed / total_calls) if total_calls else None,
            "thi": (total_hallucinated / total_calls) if total_calls else None,
            # Calls per ok: inf when calls but no oks; undefined when neither.
            "lsr": (
                (total_calls / total_ok) if total_ok
                else (float("inf") if total_calls else None)
            ),
            # Recovery rate: undefined when there were no malformed calls.
            "err": (total_recovered / total_malformed) if total_malformed else None,
            # Per-run rate: well-defined whenever runs is non-empty.
            "ert": (runs_early_resign_fail / len(runs)) if runs else None,
            "n_runs": len(runs),
            "n_calls": total_calls,
            "n_malformed": total_malformed,
            "n_hallucinated": total_hallucinated,
        }
    return per_model


# ----------------------------------------------------------------------
# Failure breakdown (F1–F7)
# ----------------------------------------------------------------------


def classify_failure(entry: dict) -> str:
    """Assign a failure code: F1-F7 (with sub-types), PASS, or INFRA_ERROR.

    Priority order (highest first):
      F4   step_budget_exhausted
      F1   any hallucinated call
      F2   malformed call (mid-trajectory, model continued)
      F2_F5 malformed call then immediate resignation
      F5   early resignation (final_answer, wrong answer, >=2 ok calls)
      F5b  plan-without-execute (agentic tiers only; resigned with 0-1 successful tool calls)
      F6   wrong tool (trajectory failed, no F1/F2)
      F7   partial (at least one checker passed)
      F3   default (context amnesia)
    """
    if entry.get("tcr_pass"):
        return "PASS"
    term = entry.get("termination") or ""
    if term == "provider_error":
        return "INFRA_ERROR"

    call_log = entry.get("call_log") or []

    # F4: ran out of steps
    if term == "step_budget_exhausted":
        return "F4"

    # F1: hallucinated tool
    if any(c.get("is_hallucinated") for c in call_log):
        return "F1"

    # F2: malformed call (includes malformed_terminal_call)
    has_malformed = term == "malformed_terminal_call" or any(
        c.get("is_malformed") for c in call_log
    )
    if has_malformed:
        if term in ("final_answer", "malformed_terminal_call"):
            return "F2_F5"
        return "F2"

    # F5: early resignation (model decided it was done but wrong)
    if term == "final_answer":
        ok_calls = sum(1 for c in call_log if c.get("response_status") == "ok")
        if _tier_of(entry) != "A0" and ok_calls <= 1:
            return "F5b"
        return "F5"

    # F6: trajectory check failed (wrong tool/args, not hallucinated/malformed)
    if not entry.get("trajectory_passed"):
        return "F6"

    # F7: some checker passed but not all
    passed_checks = sum([
        bool(entry.get("final_answer_passed")),
        bool(entry.get("submission_passed")),
        bool(entry.get("trajectory_passed")),
        bool(entry.get("forbidden_all_passed")),
    ])
    if passed_checks >= 1:
        return "F7"

    # Default: context amnesia
    return "F3"


def compute_failure_breakdown(
    entries: list[dict]
) -> dict[str, dict[str, dict[str, int]]]:
    """Failure code distribution per model x tier.

    Returns: {model: {tier: {code: count}}}

    Skips unscored entries — classify_failure needs tcr_pass + the checker
    outcomes, and assigning codes to unscored runs would emit spurious
    F5 (early resignation) for anything that happened to terminate as
    final_answer without being evaluated.
    """
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for e in entries:
        if e.get("tcr_pass") is None:
            continue
        model = e["model"]
        tier = _tier_of(e)
        code = classify_failure(e)
        out[model][tier][code] += 1
    # Convert nested defaultdicts to regular dicts for cleaner output
    return {
        m: {t: dict(codes) for t, codes in by_tier.items()}
        for m, by_tier in out.items()
    }


# ----------------------------------------------------------------------
# Cost / efficiency
# ----------------------------------------------------------------------


def compute_cost_efficiency(entries: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Per-model x per-tier token usage and latency averages.

    Returns: {model: {tier: {mean_input_tokens, mean_output_tokens,
                              mean_latency_ms, mean_wall_ms, n_runs,
                              tokens_per_pass}}}

    tokens_per_pass is total_tokens / pass_count (float('inf') if zero passes).
    """
    # Skip unscored entries — n_pass / tokens_per_pass is undefined when
    # we don't know which runs passed.
    by_mt: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e.get("tcr_pass") is None:
            continue
        by_mt[e["model"]][_tier_of(e)].append(e)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for model, tiers in by_mt.items():
        out[model] = {}
        for tier, runs in tiers.items():
            n = len(runs)
            in_tok = [(r.get("total_usage") or {}).get("input_tokens", 0) for r in runs]
            out_tok = [(r.get("total_usage") or {}).get("output_tokens", 0) for r in runs]
            lat = [r.get("total_latency_ms", 0) for r in runs]
            wall = [r.get("wall_clock_ms", 0) for r in runs]
            passes = sum(1 for r in runs if r.get("tcr_pass"))
            total_tokens = sum(in_tok) + sum(out_tok)
            out[model][tier] = {
                "n_runs": n,
                "n_pass": passes,
                "mean_input_tokens": (sum(in_tok) / n) if n else 0.0,
                "mean_output_tokens": (sum(out_tok) / n) if n else 0.0,
                "mean_latency_ms": (sum(lat) / n) if n else 0.0,
                "mean_wall_ms": (sum(wall) / n) if n else 0.0,
                "tokens_per_pass": (total_tokens / passes) if passes else float("inf"),
            }
    return out


# ----------------------------------------------------------------------
# Loaders — fuse result.json + score.json, optionally join models.yaml
# ----------------------------------------------------------------------


def load_entries(
    results_dir: Path,
    *,
    models_yaml: Path | None = None,
    require_scored: bool = False,
) -> list[dict]:
    """Walk a results directory and build one entry per (result, score) pair.

    Missing .score.json files are tolerated unless require_scored=True;
    the entry will have tcr_pass=None and all *_passed=None.

    If models_yaml is provided, entries gain params_b and tier fields.
    """
    import json
    import yaml

    model_meta: dict[str, dict] = {}
    if models_yaml:
        with open(models_yaml) as f:
            doc = yaml.safe_load(f)
        # Accept two schemas:
        #   1. Legacy models.yaml: `models: {raw_name: {slug, tier, params_b, ...}, ...}`
        #   2. Sweep config:        `models: [{name, params_b, tier, ...}, ...]`
        raw_models = doc.get("models")
        items: list[tuple[str, dict]] = []
        if isinstance(raw_models, dict):
            items = list(raw_models.items())
        elif isinstance(raw_models, list):
            items = [(m["name"], m) for m in raw_models if m.get("name")]
        for raw_name, meta in items:
            entry = {
                "params_b": meta.get("params_b"),
                "tier_bucket": meta.get("tier"),
                "raw_name": raw_name,
            }
            # Register under multiple plausible keys so result-file slugs match:
            #   1. the explicit `slug:` field from models.yaml (e.g. "qwen2.5-7b")
            #   2. the colon-and-slash-normalized raw name (matches runner._slug())
            #   3. the raw name itself
            keys = {raw_name, raw_name.replace(":", "_").replace("/", "_")}
            slug = meta.get("slug")
            if slug:
                keys.add(slug)
                keys.add(slug.replace("-", "_"))
            for key in keys:
                if key and key not in model_meta:
                    model_meta[key] = entry

    entries: list[dict] = []
    for result_path in sorted(Path(results_dir).rglob("*.json")):
        if result_path.name.endswith(".score.json"):
            continue
        # skip known non-result files (yaml/compat matrices are .yaml, but be safe)
        try:
            with open(result_path) as f:
                result = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(result, dict) or "task_id" not in result:
            continue

        score_path = result_path.with_suffix(".score.json")
        score: dict = {}
        if score_path.exists():
            try:
                with open(score_path) as f:
                    score = json.load(f)
            except json.JSONDecodeError:
                score = {}
        elif require_scored:
            continue

        # Derive variant, run_idx from filename
        stem = result_path.stem
        parts = stem.split("__")
        variant = parts[1] if len(parts) >= 2 else "v0"
        run_idx = 0
        if len(parts) >= 3 and parts[2].startswith("run"):
            try:
                run_idx = int(parts[2][3:])
            except ValueError:
                pass

        model_name = result.get("model", "?")
        model_slug = model_name.replace(":", "_").replace("/", "_").replace(" ", "_")

        entry: dict = {
            "task_id": result.get("task_id"),
            "level": result.get("level") or score.get("level"),
            "model": model_slug,
            "model_raw": model_name,
            "provider": result.get("provider"),
            "variant": score.get("variant", variant),
            "run_idx": score.get("run_idx", run_idx),
            "call_log": result.get("call_log") or [],
            "termination": result.get("termination"),
            "total_turns": result.get("total_turns", 0),
            "total_usage": result.get("total_usage") or {},
            "total_latency_ms": result.get("total_latency_ms", 0),
            "wall_clock_ms": result.get("wall_clock_ms", 0),
            "result_path": str(result_path),
        }

        if score:
            entry["tcr_pass"] = score.get("tcr_pass")
            entry["summary"] = score.get("summary")
            entry["final_answer_passed"] = (score.get("final_answer") or {}).get("passed")
            entry["submission_passed"] = (score.get("submission") or {}).get("passed")
            entry["trajectory_passed"] = (score.get("trajectory") or {}).get("passed")
            fb = score.get("forbidden") or []
            entry["forbidden_all_passed"] = all(f.get("passed") for f in fb) if fb else True
            entry["stubbed_checks"] = _extract_stubbed_check_names(score)
        else:
            entry["tcr_pass"] = None
            entry["summary"] = None
            entry["final_answer_passed"] = None
            entry["submission_passed"] = None
            entry["trajectory_passed"] = None
            entry["forbidden_all_passed"] = None
            entry["stubbed_checks"] = []

        # Join model metadata if we have it
        meta = model_meta.get(model_slug)
        if meta:
            entry["params_b"] = meta.get("params_b")
            entry["tier_bucket"] = meta.get("tier_bucket")
        else:
            entry["params_b"] = None
            entry["tier_bucket"] = None

        entries.append(entry)

    return entries


def extract_model_params(entries: list[dict]) -> dict[str, float]:
    """Pull out model → params_b map from loaded entries."""
    out: dict[str, float] = {}
    for e in entries:
        if e.get("params_b") is not None:
            out[e["model"]] = float(e["params_b"])
    return out


# ----------------------------------------------------------------------
# Stubbed-check coverage (visibility for doc/gaps.md G-018)
# ----------------------------------------------------------------------


def _extract_stubbed_check_names(score: dict) -> list[str]:
    """Collect the names of every predicate that was marked ``stubbed: True``
    in its CheckResult.details. Returned list is the canonical source for
    whether a TCR=pass row leaned on a stubbed predicate.

    Looks in three places:
      * score.trajectory.details.stubbed          → whole-trajectory fallback
      * score.trajectory_subchecks.<name>.details.stubbed
      * score.forbidden[i].details.stubbed

    The per-forbidden entry uses its `type` (or `behavior_type`) so the name
    lines up with the keys in `doc/metrics.md`. Subchecks use their predicate
    name.
    """
    stubbed: list[str] = []

    trajectory = score.get("trajectory") or {}
    tdetails = trajectory.get("details") or {}
    if tdetails.get("stubbed"):
        stubbed.append("trajectory:fallback")

    subs = score.get("trajectory_subchecks") or {}
    for name, cr in subs.items():
        details = (cr or {}).get("details") or {}
        if details.get("stubbed"):
            stubbed.append(f"trajectory:{name}")

    for fb in score.get("forbidden") or []:
        details = (fb or {}).get("details") or {}
        if details.get("stubbed"):
            t = details.get("behavior_type") or fb.get("type") or "?"
            stubbed.append(f"forbidden:{t}")

    return stubbed


def compute_stubbed_coverage(
    entries: list[dict],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-model x per-tier, how many TCR-pass runs leaned on ≥1 stubbed check.

    Returns ``{model: {tier: {passes, passes_with_stubs, rate, stubs_set}}}``
    where ``stubs_set`` is the sorted list of distinct stubbed predicate
    names observed across passing runs in that cell. Non-passing and
    unscored runs are excluded — the goal is to flag when a TCR ``pass``
    was partially propped up by a stub.

    The metrics CLI surfaces this alongside the TCR matrix so a reader can
    see which cells are trustworthy vs. which ones lean on stubbed
    predicates. Addresses doc/gaps.md G-018's P2 concern.
    """
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "passes": 0,
            "passes_with_stubs": 0,
            "stubs_set": set(),
        })
    )
    for e in entries:
        if not e.get("tcr_pass"):
            continue
        tier = _tier_of(e)
        cell = out[e["model"]][tier]
        cell["passes"] += 1
        stubs = e.get("stubbed_checks") or []
        if stubs:
            cell["passes_with_stubs"] += 1
            cell["stubs_set"].update(stubs)

    # Add overall per model + fill rate + convert set → sorted list
    for model, by_tier in out.items():
        total_pass = sum(c["passes"] for c in by_tier.values())
        total_stub = sum(c["passes_with_stubs"] for c in by_tier.values())
        all_stubs: set[str] = set()
        for c in by_tier.values():
            all_stubs.update(c["stubs_set"])
        by_tier["overall"] = {
            "passes": total_pass,
            "passes_with_stubs": total_stub,
            "stubs_set": all_stubs,
        }
        for cell in by_tier.values():
            cell["rate"] = (
                cell["passes_with_stubs"] / cell["passes"]
                if cell["passes"] else 0.0
            )
            cell["stubs_set"] = sorted(cell["stubs_set"])
    return {m: dict(v) for m, v in out.items()}
