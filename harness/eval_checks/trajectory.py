"""trajectory_check dispatcher.

Reads `oracle_evaluator.trajectory_check` (a dict of boolean predicates,
plus the parametrized `require_constraint_check_called` list) and runs
each enabled predicate against `result.call_log`. Also enforces
`trace_requirements.must_call_tools`.

The dispatcher returns:
  - one top-level CheckResult (AND of all predicates that ran)
  - a per-predicate dict {name: CheckResult} for the metrics layer

Predicate inventory (11 + must_call_tools = 12):
  require_step_budget                              universal
  require_valid_tool_arguments                     universal except A0
  require_branch_correctness                       tier C
  require_preferred_checked_first                  C4
  require_all_candidates_inspected                 D1
  require_constraint_applied                       D1
  require_recovery                                 D5, E5
  require_backtracking                             D5
  require_contradiction_detected                   D5
  require_constraint_check_called: [list of names] E1
  require_consistency_between_recovery_and_submission   E5
  must_call_tools (from trace_requirements)        universal

Branch correctness implementation: walks the call_log to find the
lookup_record matching gold_state.target_record_id, extracts
response.result.fields[branch_field], decides which branch should have
been taken (primary if observed value == primary_branch_value else
fallback), then checks the LAST submit_decision call's `action` field
against gold_state's primary_submission/fallback_submission.

Recovery and backtracking checks: walk the call_log looking for
`response_status == 'error'` envelopes followed by *different* tool
calls. Recovery = error followed by ANY subsequent successful call.
Backtracking = error followed by a different option/argument than the
one that failed.
"""

from __future__ import annotations

from typing import Any

from . import CheckResult
from .text_extract import value_in_text


# ----------------------------------------------------------------------
# Top-level dispatcher
# ----------------------------------------------------------------------


def check_trajectory(
    result: dict,
    oracle_evaluator: dict,
    gold_state: dict,
    trace_requirements: dict,
) -> tuple[CheckResult, dict[str, CheckResult]]:
    """Run all enabled trajectory predicates and AND them together.

    Returns (top_level_result, per_predicate_results).

    The per_predicate dict is consumed later by the metrics layer
    (SDR/THI/LSR/ERT/ERR all read from it).
    """
    spec = oracle_evaluator.get("trajectory_check") or {}
    per: dict[str, CheckResult] = {}

    # 1. must_call_tools (always-on, from trace_requirements)
    must_call = trace_requirements.get("must_call_tools") or []
    if must_call:
        per["must_call_tools"] = _check_must_call_tools(result, must_call)

    # 2. Boolean predicates from oracle_evaluator.trajectory_check
    # Track every truthy flag the task declared so we can surface any that
    # are not registered (previously these were silently skipped — see the
    # 2026-04-14 audit, P1 finding about missing predicate enforcement).
    _PARAMETRIZED_FLAGS = {"require_constraint_check_called"}
    for flag, value in spec.items():
        if not value:
            continue
        if flag in _PARAMETRIZED_FLAGS:
            continue
        predicate = _PREDICATES.get(flag)
        if predicate is None:
            # Register an explicit failure entry so the score surface
            # reflects the missing registration — preventing silent passes
            # on declared-but-unimplemented predicates.
            per[flag] = CheckResult.fail(
                f"trajectory predicate {flag!r} declared in task YAML but not "
                f"registered in _PREDICATES — this is a harness bug, not a "
                f"model failure",
                unregistered_predicate=True,
            )
            continue
        per[flag] = predicate(result, gold_state, spec)

    # 3. Parametrized: require_constraint_check_called (list of names)
    constraint_names = spec.get("require_constraint_check_called")
    if constraint_names:
        per["require_constraint_check_called"] = _check_constraint_check_called(
            result, constraint_names
        )

    # AND everything together for the top-level verdict
    failures = [name for name, r in per.items() if not r.passed]
    if failures:
        # Compose a one-line reason listing failed predicates with their
        # short reasons (truncated). Per-predicate detail is in the dict.
        snippets = [f"{name}: {per[name].reason}"[:120] for name in failures]
        return (
            CheckResult.fail(
                f"failed {len(failures)}/{len(per)}: " + "; ".join(snippets),
                failed_predicates=failures,
                num_predicates=len(per),
            ),
            per,
        )
    return (
        CheckResult.ok(
            f"all {len(per)} predicates passed",
            num_predicates=len(per),
        ),
        per,
    )


# ----------------------------------------------------------------------
# Individual predicates
# ----------------------------------------------------------------------


def _check_must_call_tools(result: dict, must_call: list) -> CheckResult:
    """Every name in must_call must appear in the call_log."""
    called = {c.get("tool_name") for c in (result.get("call_log") or [])}
    missing = [t for t in must_call if t not in called]
    if missing:
        return CheckResult.fail(
            f"missing required tool(s): {', '.join(missing)}",
            missing=missing,
            called=sorted(called),
        )
    return CheckResult.ok(
        f"all {len(must_call)} required tool(s) called",
        called=sorted(called),
    )


def _check_step_budget(result: dict, gold_state: dict, spec: dict) -> CheckResult:
    """Pass iff termination is not step_budget_exhausted."""
    term = result.get("termination")
    if term == "step_budget_exhausted":
        return CheckResult.fail(
            "step budget exhausted",
            termination=term,
            total_turns=result.get("total_turns"),
        )
    return CheckResult.ok("within step budget", termination=term)


def _check_valid_tool_arguments(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """Pass iff no call_log entry has is_malformed=True."""
    call_log = result.get("call_log") or []
    malformed = [c for c in call_log if c.get("is_malformed")]
    if malformed:
        names = sorted({c.get("tool_name") for c in malformed})
        return CheckResult.fail(
            f"{len(malformed)} malformed call(s): {', '.join(names)}",
            num_malformed=len(malformed),
            malformed_tools=names,
        )
    return CheckResult.ok(
        f"no malformed calls in {len(call_log)} total",
        num_calls=len(call_log),
    )


def _check_branch_correctness(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """Did the model take the right branch/selection given what it observed?

    Tier C tasks come in 4 gold_state shapes. We dispatch on shape:

      Shape 1 (C1/C4 original) — full branch-field triple
        Required: target_record_id, branch_field, primary_branch_value,
                  primary_submission, fallback_submission
        Check: model looked up branch_field, chose primary vs fallback
               based on observed value.

      Shape 2 (C3-like) — threshold / derived decision
        Required: expected_decision (plus typically target_record_id)
        Check: last submit_decision's action matches expected_decision.

      Shape 3 (C5-like) — attribute map / structured submission
        Required: expected_submission (dict)
        Check: last submit_decision's args include every key=value in
               expected_submission.

      Shape 4 (C2-like) — find-and-report
        Required: target_value (plus target_record_id)
        Check: final_text OR submission payload contains target_value.

    If none of the shapes match, fail with an informative message.
    """
    target_id = gold_state.get("target_record_id")
    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    last_submit_args = (submits[-1].get("args") or {}) if submits else {}

    # ------------------------------------------------------------------
    # Shape 1: full branch-field triple (C1/C4 original semantics)
    # ------------------------------------------------------------------
    branch_field = gold_state.get("branch_field")
    primary_value = gold_state.get("primary_branch_value")
    primary_sub = gold_state.get("primary_submission")
    fallback_sub = gold_state.get("fallback_submission")
    if all([target_id, branch_field, primary_value, primary_sub, fallback_sub]):
        return _branch_correctness_full(result, gold_state)

    # ------------------------------------------------------------------
    # Shape 2: threshold / derived decision (C3-like)
    # ------------------------------------------------------------------
    expected_decision = gold_state.get("expected_decision")
    if expected_decision is not None:
        if not submits:
            return CheckResult.fail(
                "branch_correctness: expected_decision set but no submit_decision",
                expected_decision=expected_decision,
            )
        actual = (
            last_submit_args.get("action")
            or last_submit_args.get("decision")
            or last_submit_args.get("result")
        )
        if actual is None:
            # Scan the submission payload for a matching value anywhere
            if str(expected_decision) in {str(v) for v in last_submit_args.values()}:
                return CheckResult.ok(
                    f"branch correct: submission contains {expected_decision!r}",
                    expected_decision=expected_decision,
                    submission=last_submit_args,
                )
            return CheckResult.fail(
                "branch_correctness: last submission has no action/decision/result key",
                expected_decision=expected_decision,
                submission=last_submit_args,
            )
        if str(actual) == str(expected_decision):
            return CheckResult.ok(
                f"branch correct: {actual} == expected_decision",
                expected_decision=expected_decision,
                actual=actual,
            )
        return CheckResult.fail(
            f"branch_correctness: expected {expected_decision!r}, got {actual!r}",
            expected_decision=expected_decision,
            actual=actual,
        )

    # ------------------------------------------------------------------
    # Shape 3: attribute map / structured submission (C5-like)
    # ------------------------------------------------------------------
    expected_submission = gold_state.get("expected_submission")
    if isinstance(expected_submission, dict) and expected_submission:
        if not submits:
            return CheckResult.fail(
                "branch_correctness: expected_submission set but no submit_decision",
                expected_submission=expected_submission,
            )
        missing = []
        wrong = []
        for k, v in expected_submission.items():
            if k not in last_submit_args:
                missing.append(k)
            elif str(last_submit_args[k]) != str(v):
                wrong.append((k, v, last_submit_args[k]))
        if missing or wrong:
            return CheckResult.fail(
                f"branch_correctness: submission mismatch "
                f"(missing={missing}, wrong={wrong})",
                missing_keys=missing,
                wrong_values=wrong,
                submission=last_submit_args,
            )
        return CheckResult.ok(
            f"branch correct: all {len(expected_submission)} key(s) match",
            submission=last_submit_args,
        )

    # ------------------------------------------------------------------
    # Shape 4: find-and-report (C2-like)
    # ------------------------------------------------------------------
    target_value = gold_state.get("target_value")
    if target_value is not None:
        final_text = (result.get("final_text") or "").strip()
        target_str = str(target_value)

        # Answer must appear in final_text or submission
        answer_present = False
        where = None
        if final_text and target_str.lower() in final_text.lower():
            answer_present = True
            where = "final_text"
        elif submits and target_str in {str(v) for v in last_submit_args.values()}:
            answer_present = True
            where = "submission"

        if not answer_present:
            return CheckResult.fail(
                f"branch_correctness: target_value {target_str!r} not in final_text or submission",
                target_value=target_value,
                final_text=(final_text[:100] if final_text else None),
                submission=last_submit_args,
            )

        # Tighter check (P1 audit 2): when gold_state declares that the
        # task has multiple-match disambiguation (C2-style), the model
        # must have taken a structural disambiguation step before
        # answering — otherwise a hallucinated answer that happens to
        # contain the target_value string would pass.
        #
        # Disambiguation is satisfied by any of:
        #   1. a `compare_records` call (the canonical C2 tool), or
        #   2. two or more distinct lookup_record/get_attribute calls
        #      against different record_ids (the model manually walked
        #      the matches)
        needs_disambiguation = (
            gold_state.get("disambiguation_criteria") is not None
            or gold_state.get("num_matches", 1) > 1
        )
        if needs_disambiguation:
            has_compare = any(
                c.get("tool_name") == "compare_records" for c in call_log
            )
            distinct_ids: set[str] = set()
            for c in call_log:
                if c.get("tool_name") not in ("lookup_record", "get_attribute"):
                    continue
                rid = (c.get("args") or {}).get("record_id")
                if rid:
                    distinct_ids.add(str(rid))
            if not has_compare and len(distinct_ids) < 2:
                return CheckResult.fail(
                    f"branch_correctness: target_value present in {where}, but "
                    f"gold_state requires disambiguation (compare_records or "
                    f">=2 distinct lookups) and none was observed",
                    target_value=target_value,
                    where=where,
                    has_compare=has_compare,
                    distinct_lookups=len(distinct_ids),
                )
            return CheckResult.ok(
                f"branch correct: target_value in {where}; disambiguation via "
                f"{'compare_records' if has_compare else f'{len(distinct_ids)} distinct lookups'}",
                target_value=target_value,
                where=where,
            )

        return CheckResult.ok(
            f"branch correct: target_value {target_str!r} in {where}",
            target_value=target_value,
            where=where,
        )

    # ------------------------------------------------------------------
    # No recognizable shape
    # ------------------------------------------------------------------
    return CheckResult.fail(
        "branch_correctness: gold_state does not match any known shape "
        "(expected one of: branch_field+primary_branch_value+primary_submission+"
        "fallback_submission; expected_decision; expected_submission; target_value)",
        gold_state_keys=sorted(gold_state.keys()),
    )


def _branch_correctness_full(
    result: dict, gold_state: dict
) -> CheckResult:
    """Shape 1 (C1/C4 original): full branch-field triple.

    Algorithm:
      1. Find the lookup (or get_attribute) for gold_state.target_record_id.
      2. Extract response.result.fields[branch_field] — this is what the
         model SAW.
      3. Determine the expected branch: primary if observed ==
         gold.primary_branch_value, else fallback.
      4. Find the LAST submit_decision call's action.
      5. Compare against gold.primary_submission / fallback_submission.
    """
    target_id = gold_state["target_record_id"]
    branch_field = gold_state["branch_field"]
    primary_value = gold_state["primary_branch_value"]
    primary_sub = gold_state["primary_submission"]
    fallback_sub = gold_state["fallback_submission"]

    call_log = result.get("call_log") or []

    # Find the lookup that returned the branch_field for the target record
    observed_value = None
    for c in call_log:
        if c.get("tool_name") not in ("lookup_record", "get_attribute"):
            continue
        args = c.get("args") or {}
        if args.get("record_id") != target_id:
            continue
        envelope = c.get("response") or {}
        if envelope.get("status") != "ok":
            continue
        result_data = envelope.get("result") or {}
        if c.get("tool_name") == "lookup_record":
            fields = result_data.get("fields") or {}
            if branch_field in fields:
                observed_value = fields[branch_field]
                break
        else:
            if result_data.get("attribute") == branch_field:
                observed_value = result_data.get("value")
                break

    if observed_value is None:
        return CheckResult.fail(
            f"branch_correctness: model never observed {branch_field} of {target_id}",
            target_id=target_id,
            branch_field=branch_field,
        )

    expected_action = primary_sub if observed_value == primary_value else fallback_sub

    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    if not submits:
        return CheckResult.fail(
            "branch_correctness: no submit_decision call",
        )
    last_submit = submits[-1]
    actual_action = (last_submit.get("args") or {}).get("action")

    if str(actual_action) != str(expected_action):
        return CheckResult.fail(
            f"branch_correctness: observed {branch_field}={observed_value!r}, "
            f"expected action={expected_action!r}, got {actual_action!r}",
            observed_value=observed_value,
            expected_action=expected_action,
            actual_action=actual_action,
        )
    return CheckResult.ok(
        f"branch correct: {branch_field}={observed_value} → {actual_action}",
        observed_value=observed_value,
        action=actual_action,
    )


def _check_preferred_checked_first(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """C4: the preferred vendor must be checked (via constraint or lookup)
    before any other vendor.
    """
    preferred = gold_state.get("preferred_vendor_id")
    if not preferred:
        return CheckResult.fail("preferred_vendor_id missing from gold_state")

    call_log = result.get("call_log") or []
    # Walk the log; the first record_id we see in args should be the preferred
    for c in call_log:
        args = c.get("args") or {}
        rid = args.get("record_id") or args.get("vendor_id") or args.get("option_id")
        if rid is None:
            # Maybe we see it inside a check_constraint payload
            inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else None
            if inputs:
                rid = inputs.get("vendor_id") or inputs.get("record_id")
        if rid is not None:
            if str(rid) == str(preferred):
                return CheckResult.ok(
                    f"preferred vendor {preferred} checked first",
                    preferred=preferred,
                )
            return CheckResult.fail(
                f"first checked id was {rid!r}, expected preferred {preferred!r}",
                first_checked=rid,
                preferred=preferred,
            )
    return CheckResult.fail(
        "no record_id observed in any call_log args",
        preferred=preferred,
    )


def _check_all_candidates_inspected(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D1: every candidate_id must have been looked up at least once.

    Defensive on ``record_id`` shape: models occasionally emit a list
    (observed live on mistral:7b D1, 2026-04-15) where the schema wants
    a string. String values go into the inspected set; list/tuple values
    contribute their string elements; other shapes are skipped. This
    prevents ``TypeError: unhashable type: 'list'`` from silently
    aborting the whole evaluator for one malformed tool call.
    """
    candidates = gold_state.get("candidate_ids") or []
    if not candidates:
        return CheckResult.fail("candidate_ids missing from gold_state")
    inspected: set = set()
    for c in (result.get("call_log") or []):
        if c.get("tool_name") not in ("lookup_record", "get_attribute"):
            continue
        rid = (c.get("args") or {}).get("record_id")
        if isinstance(rid, str) and rid:
            inspected.add(rid)
        elif isinstance(rid, (list, tuple)):
            for sub in rid:
                if isinstance(sub, str) and sub:
                    inspected.add(sub)
        # else: silently skip — unhashable or non-string shapes don't
        # count as evidence that any candidate was inspected.
    missing = [cid for cid in candidates if cid not in inspected]
    if missing:
        return CheckResult.fail(
            f"{len(missing)} of {len(candidates)} candidates not inspected: {', '.join(missing)}",
            missing=missing,
            inspected=sorted(inspected),
        )
    return CheckResult.ok(
        f"all {len(candidates)} candidates inspected",
        inspected=sorted(inspected),
    )


def _check_constraint_applied(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D1: at least one check_constraint call against the hard_constraint."""
    hard = gold_state.get("hard_constraint") or {}
    if not hard:
        return CheckResult.fail("hard_constraint missing from gold_state")

    call_log = result.get("call_log") or []
    constraint_calls = [c for c in call_log if c.get("tool_name") == "check_constraint"]
    if not constraint_calls:
        # Alternative: model could enforce the constraint by inspecting the
        # field via lookup_record/get_attribute and comparing manually.
        # We'll accept that path too: was the constraint's field looked up?
        constraint_keys = list(hard.keys())
        looked_up_fields: set[str] = set()
        for c in call_log:
            if c.get("tool_name") not in ("lookup_record", "get_attribute"):
                continue
            envelope = c.get("response") or {}
            if envelope.get("status") != "ok":
                continue
            result_data = envelope.get("result") or {}
            if c.get("tool_name") == "lookup_record":
                fields = result_data.get("fields") or {}
                looked_up_fields.update(fields.keys())
            else:
                attr = result_data.get("attribute")
                if attr:
                    looked_up_fields.add(attr)
        # Strip _max/_min/_limit comparison suffixes from constraint keys
        # so "lead_time_days_max" matches a field named "lead_time_days".
        # Don't strip anything from lookup field names — those are real
        # field identifiers.
        normalized_constraint_keys = {
            k.removesuffix("_max").removesuffix("_min").removesuffix("_limit")
            for k in constraint_keys
        }
        if normalized_constraint_keys & looked_up_fields:
            return CheckResult.ok(
                "constraint enforced via field lookup (no explicit check_constraint call)",
                method="implicit",
                matched_keys=sorted(normalized_constraint_keys & looked_up_fields),
            )
        return CheckResult.fail(
            "no check_constraint call and constraint field not looked up",
            constraint_keys=constraint_keys,
        )

    return CheckResult.ok(
        f"{len(constraint_calls)} check_constraint call(s)",
        method="explicit",
    )


def _check_recovery(result: dict, gold_state: dict, spec: dict) -> CheckResult:
    """D5/E5: model encountered an error envelope and made another call after."""
    call_log = result.get("call_log") or []
    error_indices = [
        i for i, c in enumerate(call_log) if c.get("response_status") == "error"
    ]
    if not error_indices:
        # No error → vacuously the model didn't need to recover, but the
        # benchmark requires recovery to have *happened*. Fail.
        return CheckResult.fail(
            "require_recovery: no error envelope in call_log",
        )
    # Did any post-error call land successfully?
    last_err = error_indices[-1]
    post_calls = call_log[last_err + 1 :]
    successes = [c for c in post_calls if c.get("response_status") == "ok"]
    if not successes:
        return CheckResult.fail(
            f"recovery: error at call {last_err+1} not followed by any successful call",
            error_at=last_err + 1,
        )
    return CheckResult.ok(
        f"recovered after error at call {last_err+1} ({len(successes)} subsequent ok)",
        error_at=last_err + 1,
        successes_after=len(successes),
    )


def _check_backtracking(result: dict, gold_state: dict, spec: dict) -> CheckResult:
    """D5: after an error, model called a *different* option than the one that failed.

    Heuristic: find an error envelope; check that any subsequent call uses
    different args (specifically, a different `record_id` / `option_id`)
    than the failed call.
    """
    call_log = result.get("call_log") or []
    for i, c in enumerate(call_log):
        if c.get("response_status") != "error":
            continue
        failed_args = c.get("args") or {}
        failed_id = (
            failed_args.get("record_id")
            or failed_args.get("option_id")
            or failed_args.get("vendor_id")
        )
        if failed_id is None:
            continue
        for later in call_log[i + 1 :]:
            later_args = later.get("args") or {}
            later_id = (
                later_args.get("record_id")
                or later_args.get("option_id")
                or later_args.get("vendor_id")
            )
            if later_id is not None and str(later_id) != str(failed_id):
                return CheckResult.ok(
                    f"backtracked from {failed_id} to {later_id}",
                    from_id=failed_id,
                    to_id=later_id,
                )
    return CheckResult.fail("no backtracking detected after any error")


def _check_contradiction_detected(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D5: model received an error AND then made a different choice.

    Implementation: combine recovery + backtracking. If both pass, the
    model implicitly "detected" the contradiction (it changed course).
    """
    rec = _check_recovery(result, gold_state, spec)
    if not rec.passed:
        return CheckResult.fail(
            f"contradiction_detected: {rec.reason}",
            **rec.details,
        )
    back = _check_backtracking(result, gold_state, spec)
    if not back.passed:
        return CheckResult.fail(
            f"contradiction_detected: {back.reason}",
            **back.details,
        )
    return CheckResult.ok("contradiction detected and addressed")


def _check_constraint_check_called(result: dict, required_names: list) -> CheckResult:
    """E1: every constraint name in required_names was passed to a check_constraint call."""
    call_log = result.get("call_log") or []
    seen: set[str] = set()
    for c in call_log:
        if c.get("tool_name") != "check_constraint":
            continue
        args = c.get("args") or {}
        # check_constraint has a `constraint` arg (or `constraint_name`)
        cname = args.get("constraint") or args.get("constraint_name") or args.get("name")
        if isinstance(cname, str):
            seen.add(cname)
        # Some payloads stash the name inside an inputs dict
        inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else None
        if inputs:
            cname = inputs.get("constraint") or inputs.get("name")
            if isinstance(cname, str):
                seen.add(cname)
    missing = [n for n in required_names if n not in seen]
    if missing:
        return CheckResult.fail(
            f"missing constraint check(s): {', '.join(missing)}",
            missing=missing,
            seen=sorted(seen),
        )
    return CheckResult.ok(
        f"all {len(required_names)} constraint check(s) called",
        seen=sorted(seen),
    )


def _check_recovery_submission_consistency(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """E5: the chosen recovery path and the submission must be consistent.

    Three trajectory-level requirements:
      1. The submission must declare a recovery_path.
      2. That recovery_path must be a valid key in gold_state.recovery_paths.
      3. The model must have made a check_constraint call before submitting
         (so the path's constraint was actually verified, not assumed).

    The submission_check.require_region_matches_recovery_path predicate
    handles the structural region/path consistency separately; this
    trajectory predicate adds the "verify before submit" requirement.
    """
    call_log = result.get("call_log") or []
    recovery_paths = gold_state.get("recovery_paths") or {}

    # Find the last submit_decision call (recovery-aware: a model can
    # try, error, retry).
    submit_indices = [
        i for i, c in enumerate(call_log)
        if c.get("tool_name") == "submit_decision"
    ]
    if not submit_indices:
        return CheckResult.fail(
            "no submit_decision in call_log",
            details={"n_submits": 0},
        )
    last_submit_idx = submit_indices[-1]
    last_submit = call_log[last_submit_idx]
    submit_args = last_submit.get("args") or {}

    # 1. submission declares a recovery_path
    chosen_path = submit_args.get("recovery_path")
    if not chosen_path:
        return CheckResult.fail(
            "submission has no recovery_path field",
            details={"submit_args_keys": sorted(submit_args.keys())},
        )

    # 2. the chosen path is a known one
    if chosen_path not in recovery_paths:
        return CheckResult.fail(
            f"recovery_path={chosen_path!r} not in gold_state.recovery_paths "
            f"({sorted(recovery_paths.keys())})",
            details={
                "chosen_path": chosen_path,
                "valid_paths": sorted(recovery_paths.keys()),
            },
        )

    # 3. at least one check_constraint call before the final submit
    pre_submit = call_log[:last_submit_idx]
    constraint_calls = [
        c for c in pre_submit if c.get("tool_name") == "check_constraint"
    ]
    if not constraint_calls:
        return CheckResult.fail(
            "no check_constraint call before submission — path constraint not verified",
            details={
                "chosen_path": chosen_path,
                "pre_submit_calls": len(pre_submit),
            },
        )

    return CheckResult.ok(
        f"recovery path {chosen_path!r} verified via "
        f"{len(constraint_calls)} check_constraint call(s) before submit",
        details={
            "chosen_path": chosen_path,
            "n_constraint_checks": len(constraint_calls),
        },
    )


# ----------------------------------------------------------------------
# Predicates added 2026-04-14 audit (P1): previously declared on task
# YAMLs but not registered in _PREDICATES, so they were silently skipped.
# ----------------------------------------------------------------------


def _check_conflict_detected(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D2/E4: model inspected both sides of a conflicting-values situation.

    Uses gold_state.conflicting_fields (dict of field_name → value pairs)
    to know which fields the conflict spans. The model must have emitted
    at least 2 inspection calls (lookup_record / get_attribute /
    compare_records) that together cover both sides.

    For E4 (no conflicting_fields key) we fall back to: at least one
    compare_records call OR at least 2 lookup_record calls against the
    target_option_id / target_record_id.
    """
    conflicting = gold_state.get("conflicting_fields") or {}
    call_log = result.get("call_log") or []
    inspection_calls = [
        c for c in call_log
        if c.get("tool_name") in ("lookup_record", "get_attribute", "compare_records")
    ]

    if conflicting:
        # Require that at least the fields named in conflicting_fields
        # were looked up (across all inspection calls).
        looked_up_fields: set[str] = set()
        for c in inspection_calls:
            envelope = c.get("response") or {}
            if envelope.get("status") != "ok":
                continue
            data = envelope.get("result") or {}
            if c.get("tool_name") == "lookup_record":
                looked_up_fields.update((data.get("fields") or {}).keys())
            elif c.get("tool_name") == "get_attribute":
                attr = data.get("attribute")
                if attr:
                    looked_up_fields.add(attr)
            elif c.get("tool_name") == "compare_records":
                # compare_records typically inspects multiple fields
                fields_arg = (c.get("args") or {}).get("fields") or []
                if isinstance(fields_arg, list):
                    looked_up_fields.update(fields_arg)
        required = set(conflicting.keys())
        missing = required - looked_up_fields
        if missing:
            return CheckResult.fail(
                f"conflict_detected: conflicting fields {sorted(missing)} never looked up",
                missing=sorted(missing),
                looked_up=sorted(looked_up_fields),
            )
        return CheckResult.ok(
            "all conflicting fields inspected",
            looked_up=sorted(looked_up_fields & required),
        )

    # Fallback: at least one compare_records OR >=2 lookup calls
    compares = [c for c in inspection_calls if c.get("tool_name") == "compare_records"]
    if compares:
        return CheckResult.ok(f"compare_records used ({len(compares)}x)")
    lookups = [c for c in inspection_calls if c.get("tool_name") == "lookup_record"]
    if len(lookups) >= 2:
        return CheckResult.ok(f"multiple lookups ({len(lookups)}) — conflict inspection plausible")
    return CheckResult.fail(
        "conflict_detected: neither compare_records nor multiple lookups observed",
        inspection_call_count=len(inspection_calls),
    )


def _check_precedence_applied(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D2/E4: the final answer / submission uses the precedence-winning value.

    Uses gold_state.target_value (D2) or gold_state.target_option_id
    (E4). Matches against final_text OR last submission payload.
    """
    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    last_submit_args = (submits[-1].get("args") or {}) if submits else {}
    final_text = (result.get("final_text") or "").lower()

    target = gold_state.get("target_value")
    target_id = gold_state.get("target_option_id") or gold_state.get("target_record_id")

    needle = target if target is not None else target_id
    if needle is None:
        return CheckResult.fail(
            "precedence_applied: gold_state missing target_value / target_option_id",
            gold_state_keys=sorted(gold_state.keys()),
        )

    # Check submission payload values
    needle_str = str(needle).lower()
    if any(str(v).lower() == needle_str for v in last_submit_args.values()):
        return CheckResult.ok(
            f"precedence applied: submission contains {needle!r}",
            target=needle,
        )
    # Check final_text
    if final_text and needle_str in final_text:
        return CheckResult.ok(
            f"precedence applied: final_text contains {needle!r}",
            target=needle,
        )
    return CheckResult.fail(
        f"precedence_applied: target value {needle!r} not found in submission or final_text",
        target=needle,
        submission=last_submit_args,
        final_text_sample=final_text[:120] if final_text else None,
    )


def _check_all_constraints_checked(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D3/E3/E4: every key in gold_state.hard_constraints must have been
    checked, either via check_constraint or via a lookup/get_attribute
    that retrieved the corresponding field (name prefix match, stripping
    _max/_min/_limit/_required suffixes).
    """
    hard = gold_state.get("hard_constraints") or gold_state.get("hard_constraint") or {}
    if not hard:
        return CheckResult.fail(
            "all_constraints_checked: gold_state missing hard_constraints",
        )

    required_keys = set(hard.keys())
    normalized = {
        k: (k.removesuffix("_max").removesuffix("_min")
             .removesuffix("_limit").removesuffix("_required"))
        for k in required_keys
    }
    call_log = result.get("call_log") or []

    # Track which constraint keys have been confirmed
    confirmed: set[str] = set()

    for c in call_log:
        tool = c.get("tool_name")
        args = c.get("args") or {}
        if tool == "check_constraint":
            # Accept either a 'constraint' name or inputs.constraint
            cname = args.get("constraint") or args.get("constraint_name") or args.get("name")
            inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else None
            if inputs and not cname:
                cname = inputs.get("constraint") or inputs.get("name")
            if isinstance(cname, str):
                # Match on exact name OR normalized name
                for k, norm in normalized.items():
                    if cname == k or cname == norm:
                        confirmed.add(k)
            # If the payload embeds any of the field names, also confirm
            for k, norm in normalized.items():
                if k in args or norm in args or (inputs and (k in inputs or norm in inputs)):
                    confirmed.add(k)
        elif tool in ("lookup_record", "get_attribute"):
            envelope = c.get("response") or {}
            if envelope.get("status") != "ok":
                continue
            data = envelope.get("result") or {}
            if tool == "lookup_record":
                fields = set((data.get("fields") or {}).keys())
            else:
                attr = data.get("attribute")
                fields = {attr} if attr else set()
            for k, norm in normalized.items():
                if k in fields or norm in fields:
                    confirmed.add(k)

    missing = sorted(required_keys - confirmed)
    if missing:
        return CheckResult.fail(
            f"all_constraints_checked: {len(missing)}/{len(required_keys)} not "
            f"checked: {', '.join(missing)}",
            missing=missing,
            confirmed=sorted(confirmed),
        )
    return CheckResult.ok(
        f"all {len(required_keys)} constraint(s) checked",
        confirmed=sorted(confirmed),
    )


def _check_query_correction(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """D4: the model corrected a noisy query. Requires gold_state.noisy_query
    and gold_state.corrected_query. Pass iff any search_records call used
    a query matching the corrected form (case-insensitive substring), and
    iff the noisy form was either not used OR was followed by the
    corrected form.
    """
    noisy = gold_state.get("noisy_query")
    corrected = gold_state.get("corrected_query")
    if not corrected:
        return CheckResult.fail(
            "query_correction: gold_state missing corrected_query",
        )
    call_log = result.get("call_log") or []
    searches = [c for c in call_log if c.get("tool_name") == "search_records"]
    if not searches:
        return CheckResult.fail(
            "query_correction: no search_records call in trajectory",
        )

    # Defensive on `query` arg shape: models occasionally emit dict/list
    # where the schema wants a string (observed live on llama3.2:3b D4,
    # 2026-04-15 Bug D). Non-string queries are coerced to "" so the
    # case-insensitive substring check doesn't crash; such malformed
    # queries cannot match `corrected_query` and the model will fail
    # the check cleanly.
    queries_raw = [(c.get("args") or {}).get("query") for c in searches]
    queries = [q if isinstance(q, str) else "" for q in queries_raw]
    corrected_lc = corrected.lower()
    used_corrected = any(corrected_lc in q.lower() for q in queries)
    if not used_corrected:
        return CheckResult.fail(
            f"query_correction: corrected_query {corrected!r} never used",
            noisy=noisy,
            corrected=corrected,
            queries=queries,
        )
    return CheckResult.ok(
        f"query corrected to {corrected!r}",
        corrected=corrected,
        num_searches=len(searches),
    )


def _check_future_dependency_considered(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """E2: did the model inspect the specific `future_requirement` named
    in gold_state beyond just the current requirement?

    The C2 audit found this was previously a no-op — a trace that touched
    compare_records for any reason passed. The stronger check uses
    gold_state.future_requirement (a domain token like
    'reporting_plugin_support') and verifies the model's trajectory
    actually inspected that property:

      - any check_constraint/compare_records call whose args reference
        the future_requirement by name, OR
      - any lookup_record/get_attribute that returned a field matching
        the future_requirement (exact, normalized, or substring).

    If gold_state has no future_requirement field, fall back to the
    structural proxy (stubbed pass with a warning flag).
    """
    future_req = gold_state.get("future_requirement")
    if not future_req:
        # Fallback: the task declares the predicate but not the target
        # requirement. Structural proxy only.
        call_log = result.get("call_log") or []
        multi_dim = [
            c for c in call_log
            if c.get("tool_name") in ("check_constraint", "compare_records")
        ]
        if multi_dim:
            return CheckResult.ok(
                f"stubbed-pass: {len(multi_dim)} check_constraint/compare_records call(s) "
                f"observed, but gold_state has no future_requirement to verify against",
                stubbed=True,
                weak_implementation=True,
                proxy_calls=len(multi_dim),
            )
        return CheckResult.fail(
            "future_dependency_considered: no check_constraint or compare_records "
            "observed (and gold_state has no future_requirement to match against)",
            stubbed=True,
            weak_implementation=True,
        )

    req_lc = str(future_req).lower()
    req_tokens = {t for t in req_lc.replace("-", "_").split("_") if len(t) >= 4}
    call_log = result.get("call_log") or []

    def _args_reference_requirement(args: dict) -> bool:
        """Any args key or string value mentions the future_requirement."""
        if not isinstance(args, dict):
            return False
        for k, v in args.items():
            if isinstance(k, str) and (req_lc in k.lower() or k.lower() in req_lc):
                return True
            if isinstance(v, str) and req_lc in v.lower():
                return True
            if isinstance(v, dict) and _args_reference_requirement(v):
                return True
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and req_lc in item.lower():
                        return True
        return False

    def _field_matches_requirement(field_name: str) -> bool:
        f_lc = field_name.lower()
        if req_lc in f_lc or f_lc in req_lc:
            return True
        f_tokens = {t for t in f_lc.replace("-", "_").split("_") if len(t) >= 4}
        return bool(req_tokens & f_tokens)

    for c in call_log:
        tool = c.get("tool_name")
        args = c.get("args") or {}
        if tool in ("check_constraint", "compare_records"):
            if _args_reference_requirement(args):
                return CheckResult.ok(
                    f"future_dependency verified: {tool} referenced {future_req!r}",
                    future_requirement=future_req,
                    via=tool,
                )
        if tool in ("lookup_record", "get_attribute"):
            envelope = c.get("response") or {}
            if envelope.get("status") != "ok":
                continue
            data = envelope.get("result") or {}
            if tool == "lookup_record":
                fields = (data.get("fields") or {}).keys()
            else:
                attr = data.get("attribute") or ""
                fields = [attr] if attr else []
            for f in fields:
                if _field_matches_requirement(str(f)):
                    return CheckResult.ok(
                        f"future_dependency verified: {tool} returned field {f!r} "
                        f"matching {future_req!r}",
                        future_requirement=future_req,
                        via=tool,
                        matching_field=f,
                    )
    return CheckResult.fail(
        f"future_dependency_considered: no call inspected {future_req!r}",
        future_requirement=future_req,
    )


def _check_greedy_option_avoided(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """E2: final submission must NOT reference tempting_local_optimum_id.

    Related to the forbidden-behavior check `greedy_local_selection`, but
    this one reads gold_state's `tempting_local_optimum_id` and checks
    the final submission directly. Also verifies the submission contains
    the globally_valid_target_id if present.
    """
    greedy_id = gold_state.get("tempting_local_optimum_id")
    valid_id = gold_state.get("globally_valid_target_id")
    if not greedy_id:
        return CheckResult.fail(
            "greedy_option_avoided: gold_state missing tempting_local_optimum_id",
        )
    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    if not submits:
        return CheckResult.fail("greedy_option_avoided: no submit_decision call")
    last_args = submits[-1].get("args") or {}
    greedy_str = str(greedy_id)
    # Check any submission value references the greedy id → fail
    for v in last_args.values():
        if str(v) == greedy_str:
            return CheckResult.fail(
                f"greedy_option_avoided: submitted tempting option {greedy_str!r}",
                greedy_id=greedy_id,
                submission=last_args,
            )
    # If valid_id is specified, verify it's present
    if valid_id:
        valid_str = str(valid_id)
        if not any(str(v) == valid_str for v in last_args.values()):
            return CheckResult.fail(
                f"greedy_option_avoided: expected {valid_str!r} in submission",
                valid_id=valid_id,
                submission=last_args,
            )
        return CheckResult.ok(
            f"avoided greedy {greedy_id}, chose valid {valid_id}",
            valid_id=valid_id,
        )
    return CheckResult.ok(
        f"greedy option {greedy_id} not in submission",
        submission=last_args,
    )


def _check_distractor_rejected(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """E3-style: final answer must NOT reference tempting_invalid_option_id;
    should reference valid_target_option_id if specified.

    Accepts the answer from either a submit_decision call OR — when the
    task doesn't require submission (E3 declares
    `requires_final_submission: false`) — the model's final_text prose.
    Pre-fix, this predicate unconditionally failed without submit, making
    E3 unreachable by its own declared contract.
    """
    distractor = gold_state.get("tempting_invalid_option_id")
    valid_id = gold_state.get("valid_target_option_id")
    if not distractor:
        return CheckResult.fail(
            "distractor_rejected: gold_state missing tempting_invalid_option_id",
        )

    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    dist_str = str(distractor)
    valid_str = str(valid_id) if valid_id else None

    # Primary path: if a submit exists, check its payload. Mirrors
    # C1/C3/C5 branch_correctness semantics — a submit is the
    # authoritative answer when present.
    if submits:
        last_args = submits[-1].get("args") or {}
        for v in last_args.values():
            if str(v) == dist_str:
                return CheckResult.fail(
                    f"distractor_rejected: submitted distractor {dist_str!r}",
                    distractor=distractor,
                    submission=last_args,
                    answer_source="submission",
                )
        if valid_str:
            if not any(str(v) == valid_str for v in last_args.values()):
                return CheckResult.fail(
                    f"distractor_rejected: expected {valid_str!r} in submission",
                    valid_id=valid_id,
                    answer_source="submission",
                )
            return CheckResult.ok(
                f"rejected distractor {distractor}, chose valid {valid_id}",
                answer_source="submission",
            )
        return CheckResult.ok(
            f"distractor {distractor} not submitted",
            answer_source="submission",
        )

    # Fallback: no submit. Check final_text prose for the valid option
    # and absence of the distractor. Uses word-bounded matching to
    # avoid partial-substring collisions (e.g. CI-4 matching CI-40).
    final_text = (result.get("final_text") or "").strip()
    if not final_text:
        return CheckResult.fail(
            "distractor_rejected: no submit_decision call and no final_text",
            distractor=distractor,
            valid_id=valid_id,
        )
    if value_in_text(dist_str, final_text):
        return CheckResult.fail(
            f"distractor_rejected: final_text references distractor {dist_str!r}",
            distractor=distractor,
            answer_source="final_text",
        )
    if valid_str and not value_in_text(valid_str, final_text):
        return CheckResult.fail(
            f"distractor_rejected: final_text missing expected valid option {valid_str!r}",
            valid_id=valid_id,
            answer_source="final_text",
        )
    return CheckResult.ok(
        f"rejected distractor {distractor}"
        + (f", chose valid {valid_id}" if valid_id else "")
        + " (via final_text)",
        answer_source="final_text",
    )


_STAGE_TO_TOOL: dict[str, tuple[str, ...]] = {
    # Map each declared pipeline-stage name to the minimum-evidence tool(s)
    # in the call_log that must appear for the stage to be considered
    # completed. The left-hand names are drawn from the task YAML (e.g.
    # tasks/E/E4.yaml).
    "candidate_search":  ("search_records",),
    "search":            ("search_records",),       # hint-style synonym
    "record_verification": ("lookup_record", "get_attribute"),
    "score_computation": ("compute_value",),
    "score":             ("compute_value",),        # hint-style synonym
    "final_submission":  ("submit_decision",),
    "submit":            ("submit_decision",),      # hint-style synonym
}


def _check_all_pipeline_stages_completed(
    result: dict, gold_state: dict, spec: dict
) -> CheckResult:
    """E4: the long-horizon pipeline must have traversed all its stages.

    Two modes:

      1. **Declared-stages mode** (preferred). If the YAML passes a list —
         e.g. ``require_all_pipeline_stages_completed: [candidate_search,
         record_verification, conflict_resolution, score_computation,
         compliance_check, final_submission]`` — we enforce every named
         stage. Named stages are mapped to concrete call_log evidence via
         ``_STAGE_TO_TOOL`` for the "single-tool" stages, and dispatched
         to ``_check_conflict_detected`` / ``_check_all_constraints_checked``
         for the semantic ones (``conflict_resolution`` / ``compliance_check``).
         An unmapped stage name fails explicitly rather than stubbing
         (same philosophy as the 2026-04-14 predicate-registration audit).

      2. **Hint-based fallback** (legacy). If the predicate is declared as a
         bare ``True`` with no explicit stage list, we anchor on gold_state
         hints (query → SEARCH, scoring_rule → SCORE, target_* → SUBMIT).
         Stubs as a pass if no hints are present.

    The declared-stages mode lets tasks like E4 fail realistic pipeline
    shortcuts (e.g. "skeletal one-call-per-tool" traces) that the hint-mode
    checker would have passed. See doc/gaps.md G-018 for the motivation.
    """
    declared = spec.get("require_all_pipeline_stages_completed")
    if isinstance(declared, list) and declared:
        return _check_declared_pipeline_stages(result, gold_state, spec, declared)
    return _check_hint_based_pipeline_stages(result, gold_state)


def _check_declared_pipeline_stages(
    result: dict,
    gold_state: dict,
    spec: dict,
    declared: list[str],
) -> CheckResult:
    call_log = result.get("call_log") or []
    tool_counts: dict[str, int] = {}
    for c in call_log:
        tool_counts[c.get("tool_name")] = tool_counts.get(c.get("tool_name"), 0) + 1

    required_stages: list[tuple[str, bool, str]] = []
    unmapped: list[str] = []

    for stage in declared:
        if not isinstance(stage, str):
            unmapped.append(str(stage))
            continue

        # Semantic stages delegate to existing predicate logic so we
        # reuse the same evidence rules applied elsewhere in the sweep.
        if stage == "conflict_resolution":
            sub = _check_conflict_detected(result, gold_state, spec)
            required_stages.append((
                stage, sub.passed,
                f"conflict_resolution → {sub.reason}",
            ))
            continue
        if stage == "compliance_check":
            # If gold_state declares hard_constraints, require every
            # constraint key to have been inspected. Otherwise accept
            # any check_constraint call as evidence that compliance
            # was at least touched.
            hard = gold_state.get("hard_constraints") or gold_state.get("hard_constraint") or {}
            if hard:
                sub = _check_all_constraints_checked(result, gold_state, spec)
                required_stages.append((
                    stage, sub.passed,
                    f"compliance_check → {sub.reason}",
                ))
            else:
                ok = tool_counts.get("check_constraint", 0) > 0
                required_stages.append((
                    stage, ok,
                    "check_constraint was never called"
                    if not ok else "check_constraint observed",
                ))
            continue

        # Tool-anchored stages look up the call_log evidence directly.
        tools = _STAGE_TO_TOOL.get(stage)
        if tools is None:
            unmapped.append(stage)
            continue
        ok = any(tool_counts.get(t, 0) > 0 for t in tools)
        required_stages.append((
            stage, ok,
            f"{stage} missing — none of {list(tools)} in call_log"
            if not ok else f"{stage} observed via {[t for t in tools if tool_counts.get(t, 0) > 0]}",
        ))

    if unmapped:
        # Fail explicitly — prevents silent stage passes the way the
        # 2026-04-14 audit prevented silent predicate passes.
        return CheckResult.fail(
            f"all_pipeline_stages_completed: unmapped stage name(s): {unmapped} — "
            f"extend _STAGE_TO_TOOL or the semantic dispatch in trajectory.py",
            unmapped_stages=unmapped,
            declared_stages=list(declared),
        )

    missing = [name for (name, ok, _) in required_stages if not ok]
    if missing:
        reasons = [msg for (name, ok, msg) in required_stages if not ok]
        return CheckResult.fail(
            f"all_pipeline_stages_completed: {len(missing)}/{len(required_stages)} "
            f"stage(s) missing: {', '.join(missing)} — {'; '.join(reasons)}",
            missing_stages=missing,
            required_stages=[n for n, _, _ in required_stages],
        )
    return CheckResult.ok(
        f"all {len(required_stages)} declared pipeline stage(s) completed: "
        f"{', '.join(n for n, _, _ in required_stages)}",
        stages=[n for n, _, _ in required_stages],
    )


def _check_hint_based_pipeline_stages(
    result: dict, gold_state: dict
) -> CheckResult:
    call_log = result.get("call_log") or []
    tool_counts: dict[str, int] = {}
    for c in call_log:
        tool_counts[c.get("tool_name")] = tool_counts.get(c.get("tool_name"), 0) + 1

    required_stages: list[tuple[str, bool, str]] = []
    if gold_state.get("query"):
        required_stages.append((
            "search",
            tool_counts.get("search_records", 0) > 0,
            "search_records was never called",
        ))
    if gold_state.get("scoring_rule"):
        required_stages.append((
            "score",
            tool_counts.get("compute_value", 0) > 0,
            "compute_value was never called despite scoring_rule in gold_state",
        ))
    target = gold_state.get("target_option_id") or gold_state.get("target_record_id")
    if target:
        submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
        required_stages.append((
            "submit",
            bool(submits),
            "submit_decision was never called",
        ))

    if not required_stages:
        return CheckResult.ok(
            "stubbed: gold_state has no query/scoring_rule/target to anchor stages",
            stubbed=True,
            weak_implementation=True,
        )

    missing = [name for (name, ok, _) in required_stages if not ok]
    if missing:
        reasons = [msg for (name, ok, msg) in required_stages if not ok]
        return CheckResult.fail(
            f"all_pipeline_stages_completed: {len(missing)}/{len(required_stages)} "
            f"stage(s) missing: {', '.join(missing)} — {'; '.join(reasons)}",
            missing_stages=missing,
            required_stages=[n for n, _, _ in required_stages],
        )
    return CheckResult.ok(
        f"all {len(required_stages)} pipeline stage(s) completed: "
        f"{', '.join(n for n, _, _ in required_stages)}",
        stages=[n for n, _, _ in required_stages],
    )


# ----------------------------------------------------------------------
# Predicate dispatch table
# ----------------------------------------------------------------------


_PREDICATES = {
    "require_step_budget": _check_step_budget,
    "require_valid_tool_arguments": _check_valid_tool_arguments,
    "require_branch_correctness": _check_branch_correctness,
    "require_preferred_checked_first": _check_preferred_checked_first,
    "require_all_candidates_inspected": _check_all_candidates_inspected,
    "require_constraint_applied": _check_constraint_applied,
    "require_recovery": _check_recovery,
    "require_backtracking": _check_backtracking,
    "require_contradiction_detected": _check_contradiction_detected,
    "require_consistency_between_recovery_and_submission": _check_recovery_submission_consistency,
    # Added 2026-04-14 audit — previously declared on D2/D3/D4/E2/E3/E4 task
    # YAMLs but silently skipped because not registered in _PREDICATES.
    "require_conflict_detected": _check_conflict_detected,
    "require_precedence_applied": _check_precedence_applied,
    "require_all_constraints_checked": _check_all_constraints_checked,
    "require_query_correction": _check_query_correction,
    "require_future_dependency_considered": _check_future_dependency_considered,
    "require_greedy_option_avoided": _check_greedy_option_avoided,
    "require_distractor_rejected": _check_distractor_rejected,
    "require_all_pipeline_stages_completed": _check_all_pipeline_stages_completed,
}
