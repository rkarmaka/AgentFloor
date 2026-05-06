#!/usr/bin/env python3
"""Pre-flight validator for the input-instance ablation (Step 4 of guide).

For each (task, instance_id) pair declared via <task>.instance_variants.yaml,
drive an oracle solver against the live FixtureDB + ToolRouter and assert the
evaluator returns TCR=PASS. Catches the failure modes that would otherwise
silently zero the cell across every model in the sweep:

  * Wrong override fields (e.g. target_value="active" but fixture has
    "deprecated").
  * Mistyped record IDs / search queries that don't resolve in the fixture.
  * Infeasible E1 instance constraints (no valid bundle satisfies the
    instance-specific constraint AND the fixture's bundle_validator).
  * Submission validator rejections (e.g. a C1 instance whose
    target_record_id has the wrong polarity vs the chosen action).

Run before kicking off the full instance sweep:

    python runs/run_instance_preflight.py

Exit 0 if every instance solves; exit 1 with a per-instance summary otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.db import FixtureDB                                # noqa: E402
from harness.evaluator import evaluate_run                       # noqa: E402
from harness.runner import load_task                             # noqa: E402
from harness.tools import ToolRouter                             # noqa: E402
from harness.variants.loader import list_instance_variant_ids    # noqa: E402


# ----------------------------------------------------------------------
# Per-task oracle solvers.
#
# Each solver receives the fully-overridden TaskSpec and returns
# (final_text, termination, call_log). The call_log is built by driving
# real ToolRouter calls so envelope shapes match what the runner produces.
# ----------------------------------------------------------------------


def _new_router(task) -> ToolRouter:
    db = FixtureDB(task.fixture_path) if task.fixture_path else FixtureDB(None)
    if db.is_empty:
        db.data["task_id"] = task.task_id
    return ToolRouter(db, task.tools_available)


def solve_a1(task) -> tuple[str, str, list[dict]]:
    gold = task.raw["gold_state"]
    rid = gold["target_record_id"]
    field = gold.get("target_field", "status")
    expected_value = gold["target_value"]

    router = _new_router(task)
    env = router.call("lookup_record", {"record_id": rid})
    actual = (env.get("result") or {}).get("fields", {}).get(field)
    if actual != expected_value:
        raise RuntimeError(
            f"fixture mismatch: lookup_record({rid}).{field}={actual!r}, "
            f"expected {expected_value!r}"
        )
    final_text = (
        f"The current {field} of product record {rid} is {expected_value}."
    )
    return final_text, "final_answer", router.call_log


def solve_b1(task) -> tuple[str, str, list[dict]]:
    gold = task.raw["gold_state"]
    query = gold["query"]
    target_rid = gold["target_record_id"]
    expected_value = gold["target_value"]

    router = _new_router(task)
    env = router.call("search_records", {"query": query})
    matches = (env.get("result") or {}).get("matches") or []
    rids = [m.get("id") for m in matches]
    if target_rid not in rids:
        raise RuntimeError(
            f"search '{query}' returned {rids}, expected to include {target_rid}"
        )

    env = router.call("lookup_record", {"record_id": target_rid})
    actual = (env.get("result") or {}).get("fields", {}).get("status")
    if actual != expected_value:
        raise RuntimeError(
            f"fixture mismatch: lookup_record({target_rid}).status={actual!r}, "
            f"expected {expected_value!r}"
        )
    final_text = (
        f"The product record matching '{query}' is {target_rid} "
        f"with current status {expected_value}."
    )
    return final_text, "final_answer", router.call_log


def solve_c1(task) -> tuple[str, str, list[dict]]:
    gold = task.raw["gold_state"]
    rid = gold["target_record_id"]
    branch_field = gold.get("branch_field", "status")
    primary_value = gold.get("primary_branch_value", "active")
    primary_action = gold.get("primary_submission", "assign_primary")
    fallback_action = gold.get("fallback_submission", "assign_fallback")

    router = _new_router(task)
    env = router.call("lookup_record", {"record_id": rid})
    actual_value = (env.get("result") or {}).get("fields", {}).get(branch_field)
    chosen_action = (
        primary_action if actual_value == primary_value else fallback_action
    )

    # Submit with record_id so the validator can scope branch_rules correctly
    # when multiple records have rules in the fixture (instance-ablation path).
    env = router.call(
        "submit_decision",
        {"record_id": rid, "action": chosen_action},
    )
    if env.get("status") != "ok":
        raise RuntimeError(
            f"submit_decision({rid}, {chosen_action}) rejected: "
            f"{env.get('error')}"
        )
    final_text = (
        f"Component {rid} has {branch_field}={actual_value!r}, so I am "
        f"submitting action={chosen_action!r}. branch_taken={chosen_action}."
    )
    return final_text, "submitted", router.call_log


# E1: enumerate all valid bundles for the fixture and pick the first one that
# satisfies the instance's expected_fields constraint. This makes the solver
# instance-agnostic — it works for any future E1 instance without code
# changes, as long as at least one valid bundle satisfies the constraint.
_E1_VALID_BUNDLES: list[tuple[str, str, str]] = [
    ("V-Approved-1", "CH-4", "GPU-7"),
    ("V-Approved-1", "CH-5", "GPU-9"),
    ("V-Approved-2", "CH-4", "GPU-7"),
    ("V-Approved-2", "CH-5", "GPU-9"),
    ("V-Approved-3", "CH-4", "GPU-7"),
    ("V-Approved-3", "CH-5", "GPU-9"),
]


def _bundle_satisfies(bundle: tuple[str, str, str], expected: dict) -> bool:
    vendor, chassis, gpu = bundle
    by_field = {"vendor_id": vendor, "chassis_id": chassis, "gpu_id": gpu}
    for field, want in expected.items():
        if by_field.get(field) != want:
            return False
    return True


def solve_e1(task) -> tuple[str, str, list[dict]]:
    fa_check = (task.raw.get("oracle_evaluator") or {}).get(
        "final_answer_check", {}
    )
    expected_fields = fa_check.get("expected_fields") or {}

    # Pick the first valid bundle satisfying the instance constraint
    chosen = next(
        (b for b in _E1_VALID_BUNDLES if _bundle_satisfies(b, expected_fields)),
        None,
    )
    if chosen is None:
        raise RuntimeError(
            f"no valid bundle satisfies expected_fields={expected_fields}"
        )
    vendor, chassis, gpu = chosen

    router = _new_router(task)

    # 1. Survey the option space (satisfies must_call_tools[list_options]).
    # The schema names the parameter `option_set`, not `option_category`.
    router.call("list_options", {"option_set": "vendors"})
    router.call("list_options", {"option_set": "chassis"})
    router.call("list_options", {"option_set": "gpus"})

    # 2. Verify vendor approval — required by trajectory_check.
    #    require_constraint_check_called[vendor_must_be_approved].
    router.call(
        "check_constraint",
        {
            "record_id": vendor,
            "constraint_name": "vendor_must_be_approved",
        },
    )

    # 3. Inspect chosen records (satisfies must_call_tools[lookup_record])
    router.call("lookup_record", {"record_id": vendor})
    router.call("lookup_record", {"record_id": chassis})
    router.call("lookup_record", {"record_id": gpu})

    # 4. Submit
    env = router.call(
        "submit_decision",
        {"vendor_id": vendor, "chassis_id": chassis, "gpu_id": gpu},
    )
    if env.get("status") != "ok":
        raise RuntimeError(
            f"submit_decision({vendor}/{chassis}/{gpu}) rejected: "
            f"{env.get('error')}"
        )
    final_text = (
        f"Recommended bundle: vendor_id={vendor}, chassis_id={chassis}, "
        f"gpu_id={gpu}. All constraints verified."
    )
    return final_text, "submitted", router.call_log


SOLVERS = {
    "A1": solve_a1,
    "B1": solve_b1,
    "C1": solve_c1,
    "E1": solve_e1,
}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def preflight_one(task_path: Path, variant_id: str) -> tuple[bool, str | None]:
    task = load_task(task_path, variant_id=variant_id)
    solver = SOLVERS.get(task.task_id)
    if solver is None:
        return False, f"no solver registered for task_id={task.task_id!r}"

    try:
        final_text, termination, call_log = solver(task)
    except Exception as e:
        return False, f"solver raised: {type(e).__name__}: {e}"

    result = {
        "task_id": task.task_id,
        "level": task.level,
        "call_log": call_log,
        "final_text": final_text,
        "termination": termination,
    }
    score = evaluate_run(result, task.raw, variant=variant_id, run_idx=0)
    if score.tcr_pass:
        return True, None

    # Build a precise failure diagnosis
    failed_checks = []
    if not score.final_answer.passed:
        failed_checks.append(f"final_answer({score.final_answer.reason})")
    if not score.submission.passed:
        failed_checks.append(f"submission({score.submission.reason})")
    if not score.trajectory.passed:
        failed_checks.append(f"trajectory({score.trajectory.reason})")
    for fb in score.forbidden:
        if not fb.passed:
            failed_checks.append(f"forbidden({fb.reason})")
    return False, "evaluator FAIL: " + "; ".join(failed_checks)


def main() -> int:
    task_files = [
        _REPO_ROOT / "tasks" / "A" / "A1.yaml",
        _REPO_ROOT / "tasks" / "B" / "B1.yaml",
        _REPO_ROOT / "tasks" / "C" / "C1.yaml",
        _REPO_ROOT / "tasks" / "E" / "E1.yaml",
    ]

    print("Pre-flight: verifying every instance variant solves end-to-end...")
    print()

    total = 0
    failed: list[tuple[str, str, str]] = []
    for tp in task_files:
        ids = list_instance_variant_ids(tp)
        if not ids:
            print(f"  (no instance variants for {tp.parent.name}/{tp.stem})")
            continue
        for vid in ids:
            total += 1
            ok, err = preflight_one(tp, vid)
            tag = f"{tp.parent.name}/{tp.stem}/{vid}"
            if ok:
                print(f"  PASS  {tag}")
            else:
                print(f"  FAIL  {tag}: {err}")
                failed.append((tp.stem, vid, err or ""))

    print()
    if failed:
        print(f"{len(failed)}/{total} instance(s) failed pre-flight:")
        for stem, vid, err in failed:
            print(f"  - {stem}/{vid}: {err}")
        return 1
    print(f"All {total} instances passed pre-flight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
