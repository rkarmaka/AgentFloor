"""submission_check dispatcher.

Reads `oracle_evaluator.submission_check` from a task YAML and validates
the submit_decision call_log entry against it. Three predicates:

  required: True/False                — whether a submission is required at all
  expected_submission: {key: value}   — literal compare of payload args
  require_submission_accepted: True   — the envelope must have status="ok"
  require_region_matches_recovery_path: True   — E5-specific cross-check
                                                 between submission region
                                                 and the recovery path's
                                                 valid_region

Two important wrinkles from real data:

1. **Use the LAST submit_decision call, not the first.** Models can recover:
   try a wrong action, get a structured error, retry with the right action.
   The fixture's branching validator emits a `branching_error` envelope on
   the first try and accepts the second. Looking at the first call would
   wrongly mark recovered runs as failures.

2. **"X OR Y" alternatives syntax in expected_submission values.** C1's
   expected_submission has `action: "assign_primary OR assign_fallback"` to
   say "either is acceptable as long as it matched the branch." We split
   on " OR " and accept any alternative.
"""

from __future__ import annotations

from . import CheckResult


def check_submission(result: dict, oracle_evaluator: dict) -> CheckResult:
    """Top-level dispatcher.

    Returns CheckResult.ok with reason='not required' when the task
    doesn't have a submission_check at all (the 19/30 tasks where
    requires_final_submission is false). Caller can short-circuit
    based on the absence-of-spec by testing oracle_evaluator first;
    we handle it here too for safety.
    """
    spec = oracle_evaluator.get("submission_check")
    if not spec or not spec.get("required"):
        return CheckResult.ok("not required")

    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    if not submits:
        return CheckResult.fail(
            "no submit_decision call in call_log",
            n_submits=0,
        )

    # Use the LAST submit_decision call. This is the recovery-aware
    # behavior: if the model tried, got a branching_error, then retried
    # successfully, we judge by the final attempt. See module docstring.
    last = submits[-1]
    n_submits = len(submits)

    # Predicate 1: require_submission_accepted
    if spec.get("require_submission_accepted"):
        envelope = last.get("response") or {}
        if envelope.get("status") != "ok":
            err = envelope.get("error") or {}
            return CheckResult.fail(
                f"last submit_decision returned status={envelope.get('status')}",
                n_submits=n_submits,
                envelope_status=envelope.get("status"),
                envelope_error_type=err.get("type") if isinstance(err, dict) else None,
                envelope_error_message=(err.get("message") if isinstance(err, dict) else None),
            )

    # Predicate 2: expected_submission literal compare (with OR support)
    expected = spec.get("expected_submission")
    if expected:
        actual = last.get("args") or {}
        for k, v in expected.items():
            if not _value_matches(actual.get(k), v):
                return CheckResult.fail(
                    f"submission.{k}={actual.get(k)!r} does not match expected {v!r}",
                    n_submits=n_submits,
                    failed_field=k,
                    actual=actual.get(k),
                    expected=v,
                )

    # Predicate 3: E5-specific cross-check between submission region
    # and the recovery path's valid_region. Implemented here because it
    # touches the submission payload directly. We need gold_state for
    # the recovery_paths map; pull it from the result if the orchestrator
    # already attached it (it does — see evaluator.py). If gold_state
    # isn't present, we can't enforce this and skip with a warning.
    if spec.get("require_region_matches_recovery_path"):
        gold_state = result.get("_gold_state") or {}
        recovery_paths = gold_state.get("recovery_paths") or {}
        actual_args = last.get("args") or {}
        # The submission has both `recovery_path` (path_a/path_b) and
        # `deployment_region`. Look up the path's valid_region.
        path = actual_args.get("recovery_path")
        region = actual_args.get("deployment_region")
        path_spec = recovery_paths.get(path) if isinstance(path, str) else None
        if not path_spec:
            return CheckResult.fail(
                f"submission recovery_path={path!r} not in gold recovery_paths",
                n_submits=n_submits,
            )
        valid_region = path_spec.get("valid_region")
        if region != valid_region:
            return CheckResult.fail(
                f"submission region={region!r} does not match path {path!r}'s valid_region={valid_region!r}",
                n_submits=n_submits,
            )

    reason = "submission accepted"
    if n_submits > 1:
        reason += f" (after {n_submits - 1} recovery attempt(s))"
    return CheckResult.ok(
        reason,
        n_submits=n_submits,
        recovered=(n_submits > 1),
    )


def _value_matches(actual, expected) -> bool:
    """Compare actual to expected with type-aware equality.

    Supports 'X OR Y' string alternatives (any number of OR-separated
    alternatives, tolerant of surrounding whitespace).

    Type rules:
      - None matches only None.
      - bool matches only bool with the same value (rejects 1/0).
      - int/float match each other (1 == 1.0) but reject bool and str.
      - str compares exactly.
      - other types compare by ==.
    """
    if expected is None:
        return actual is None

    if isinstance(expected, str) and " OR " in expected:
        alternatives = [a.strip() for a in expected.split("OR") if a.strip()]
        return isinstance(actual, str) and actual.strip() in alternatives

    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected

    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and actual == expected
        )

    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected

    return actual == expected
