"""forbidden_behaviors dispatcher.

Reads `trace_requirements.forbidden_behaviors` (a list of typed predicates,
some parametrized) and runs each one against the result. Each checker
returns a CheckResult where `passed=True` means the forbidden behavior
was NOT violated (i.e. the run is clean for this behavior).

Full registry:
  hallucinated_tool                       universal                    [IMPL]
  terminate_without_answer                most tasks                   [IMPL]
  label_not_in_valid_set                  A01                          [IMPL]
  hallucinated_facts_not_in_passage       A02/A03/A05                  [LLM-JUDGE]
  skip_required_step: {step: name}        C1/C3/C4/C5/E1/E2/E4         [IMPL]
  skip_candidate_inspection: {min: N}     D1                           [IMPL]
  persist_after_contradiction: {opt: id}  D5                           [IMPL]
  terminate_after_error: {without_retry}  D4/E5                        [IMPL]
  inconsistent_recovery_and_submission    E5                           [LLM-JUDGE]
  repeated_identical_call                 D4                           [IMPL]
  greedy_local_selection                  E2                           [IMPL]
  average_conflicting_values              D2/E4                        [IMPL-DET]
  constraint_drift                        E3                           [IMPL-DET]
  partial_constraint_check                D3/E3                        [IMPL]
  skip_pipeline_stage                     E4                           [IMPL-DET]

[IMPL]       = deterministic, always active
[IMPL-DET]   = deterministic, replaced former stub (2026-04-16)
[LLM-JUDGE]  = requires AGENTFLOOR_LLM_JUDGE=1; falls back to stubbed pass without it

skip_required_step step names registry:
  lookup_before_branch                    C1     [IMPL]
  constraint_check_before_submission      C4     [IMPL]
  bundle_submission                       E1     [IMPL]
  computation_before_decision             C3     [IMPL]
  severity_check_before_action            C5     [IMPL]
  future_dependency_check                 E2     [IMPL]
  compliance_check                        E4     [IMPL]

Unknown step names return stubbed=True (they don't fail TCR).

UNKNOWN forbidden_behavior TYPES fail closed — if a task declares a type
not in _CHECKERS, evaluator fails the run rather than silently accepting.
This forces task authors to register new types explicitly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import CheckResult


def check_all_forbidden(
    result: dict,
    forbidden_list: list,
    gold_state: dict,
) -> list[CheckResult]:
    """Run every declared forbidden behavior; return one CheckResult per.

    The orchestrator AND-s these together for the top-level TCR verdict.
    """
    out: list[CheckResult] = []
    for fb in forbidden_list or []:
        if not isinstance(fb, dict):
            out.append(CheckResult.fail(f"malformed forbidden_behavior entry: {fb!r}"))
            continue
        ftype = fb.get("type")
        checker = _CHECKERS.get(ftype)
        if checker is None:
            out.append(
                CheckResult.fail(
                    f"unknown forbidden_behavior type: {ftype!r}",
                    behavior_type=ftype,
                )
            )
            continue
        cr = checker(result, fb, gold_state)
        # Stamp the behavior type into details for downstream attribution
        cr.details.setdefault("behavior_type", ftype)
        out.append(cr)
    return out


# ----------------------------------------------------------------------
# Individual checker implementations
# ----------------------------------------------------------------------


def _check_hallucinated_tool(result: dict, fb: dict, gold_state: dict) -> CheckResult:
    """Pass iff no call_log entry is_hallucinated."""
    call_log = result.get("call_log") or []
    hallu = [c for c in call_log if c.get("is_hallucinated")]
    if hallu:
        names = sorted({c.get("tool_name") for c in hallu})
        return CheckResult.fail(
            f"{len(hallu)} hallucinated tool call(s): {', '.join(str(n) for n in names)}",
            num_hallucinated=len(hallu),
            hallucinated_names=names,
        )
    return CheckResult.ok("no hallucinated tools", num_calls=len(call_log))


def _check_terminate_without_answer(result: dict, fb: dict, gold_state: dict) -> CheckResult:
    """Pass iff the model produced a final_text OR ended via submitted.

    A 'submitted' termination is a valid form of answer (the answer lives
    in the submission payload). step_budget_exhausted with no final_text
    is the failure mode this catches.
    """
    final_text = result.get("final_text")
    termination = result.get("termination")
    if final_text and final_text.strip():
        return CheckResult.ok("final answer present")
    if termination == "submitted":
        return CheckResult.ok("answer encoded in submission")
    return CheckResult.fail(
        f"terminated without answer (termination={termination})",
        termination=termination,
    )


def _check_label_not_in_valid_set(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """A01-style: the model's label must be in the valid set.

    The valid set is specified in `fb.valid_set`. We require a word-
    boundary match so 'not a bug' does NOT count as containing 'bug'.

    When no `valid_set` is declared in the YAML, we cannot evaluate this
    predicate, so we return stubbed=True (caller can detect via the
    stubbed flag in details).
    """
    valid_set = fb.get("valid_set")
    final_text_raw = (result.get("final_text") or "").strip()
    final_text = final_text_raw.lower()
    if not final_text:
        return CheckResult.fail("no final_text to evaluate", final_text=None)
    if not valid_set:
        return CheckResult.ok(
            "no valid_set declared — predicate not evaluable",
            stubbed=True,
            final_text=final_text_raw[:60],
        )

    # Word-boundary match against each valid_set member.
    matched = []
    for label in valid_set:
        label_str = str(label).lower()
        # Escape and require non-word characters (or string boundaries) on
        # both sides so 'bug' matches 'bug.' / 'a bug:' but NOT 'bugfix'.
        pattern = rf"(?<!\w){re.escape(label_str)}(?!\w)"
        if re.search(pattern, final_text):
            matched.append(label_str)
    if matched:
        return CheckResult.ok(
            "label in valid set",
            matched=matched,
            final_text=final_text_raw[:60],
        )
    return CheckResult.fail(
        f"label not in valid set {valid_set}",
        final_text=final_text_raw[:60],
        valid_set=valid_set,
    )


def _check_skip_required_step(result: dict, fb: dict, gold_state: dict) -> CheckResult:
    """Generic 'a particular step must have happened.'

    The `step` field is a string identifier for the required step. We
    interpret a small dictionary of known step names. Unknown step names
    return a passing result with a `stubbed: True` flag (so we don't
    silently fail tasks that use a step name we haven't taught the
    checker yet).

    Known step semantics:
      lookup_before_branch         — at least one lookup_record before any submit_decision
      constraint_check_before_submission — at least one check_constraint before submit_decision
      bundle_submission            — at least one submit_decision (E1)
    """
    step = fb.get("step")
    call_log = result.get("call_log") or []

    if step == "lookup_before_branch":
        # Find first submit_decision; check if any lookup precedes it
        first_submit_idx = None
        for i, c in enumerate(call_log):
            if c.get("tool_name") == "submit_decision":
                first_submit_idx = i
                break
        if first_submit_idx is None:
            return CheckResult.ok("no submit_decision yet — vacuous", stubbed=False)
        prior_lookups = [
            c for c in call_log[:first_submit_idx]
            if c.get("tool_name") in ("lookup_record", "get_attribute")
        ]
        if not prior_lookups:
            return CheckResult.fail(
                "submit_decision called before any lookup_record / get_attribute",
                first_submit_at=first_submit_idx + 1,
            )
        return CheckResult.ok(
            f"{len(prior_lookups)} lookup(s) before first submit",
            num_prior_lookups=len(prior_lookups),
        )

    if step == "constraint_check_before_submission":
        first_submit_idx = None
        for i, c in enumerate(call_log):
            if c.get("tool_name") == "submit_decision":
                first_submit_idx = i
                break
        if first_submit_idx is None:
            return CheckResult.ok("no submit_decision yet — vacuous")
        prior_checks = [
            c for c in call_log[:first_submit_idx]
            if c.get("tool_name") == "check_constraint"
        ]
        if not prior_checks:
            return CheckResult.fail(
                "submit_decision called before any check_constraint",
                first_submit_at=first_submit_idx + 1,
            )
        return CheckResult.ok(
            f"{len(prior_checks)} check_constraint(s) before first submit"
        )

    if step == "bundle_submission":
        submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
        if not submits:
            return CheckResult.fail("no bundle submission")
        return CheckResult.ok(f"{len(submits)} submission(s)")

    if step == "computation_before_decision":
        # C3: at least one compute_value call before any submit_decision
        first_submit_idx = None
        for i, c in enumerate(call_log):
            if c.get("tool_name") == "submit_decision":
                first_submit_idx = i
                break
        if first_submit_idx is None:
            return CheckResult.ok("no submit_decision yet — vacuous")
        prior_computes = [
            c for c in call_log[:first_submit_idx]
            if c.get("tool_name") == "compute_value"
        ]
        if not prior_computes:
            return CheckResult.fail(
                "submit_decision called before any compute_value",
                first_submit_at=first_submit_idx + 1,
            )
        return CheckResult.ok(
            f"{len(prior_computes)} compute_value call(s) before first submit"
        )

    if step == "severity_check_before_action":
        # C5: severity must be inspected before submit_decision
        # (via check_constraint OR a lookup/get_attribute on a severity field)
        first_submit_idx = None
        for i, c in enumerate(call_log):
            if c.get("tool_name") == "submit_decision":
                first_submit_idx = i
                break
        if first_submit_idx is None:
            return CheckResult.ok("no submit_decision yet — vacuous")
        severity_checked = False
        for c in call_log[:first_submit_idx]:
            if c.get("tool_name") == "check_constraint":
                severity_checked = True
                break
            if c.get("tool_name") == "get_attribute":
                attr = (c.get("args") or {}).get("attribute") or ""
                if "severity" in str(attr).lower():
                    severity_checked = True
                    break
            if c.get("tool_name") == "lookup_record":
                envelope = c.get("response") or {}
                result_data = envelope.get("result") or {}
                fields = result_data.get("fields") or {}
                if any("severity" in str(k).lower() for k in fields.keys()):
                    severity_checked = True
                    break
        if not severity_checked:
            return CheckResult.fail(
                "submit_decision called without prior severity check",
                first_submit_at=first_submit_idx + 1,
            )
        return CheckResult.ok("severity checked before submission")

    if step == "future_dependency_check":
        # E2: model must have inspected the future_requirement named in
        # the fb payload (or inferred from a sibling gold_state — but
        # gold_state isn't in scope here, so the fb entry must carry it).
        # This check is paired with the trajectory predicate
        # `require_future_dependency_considered`; we accept either form
        # of specification for robustness.
        future_req = fb.get("future_requirement")
        if not future_req:
            # Fallback structural proxy
            multi_dim = [
                c for c in call_log
                if c.get("tool_name") in ("check_constraint", "compare_records")
            ]
            if not multi_dim:
                return CheckResult.fail(
                    "future_dependency_check: no check_constraint/compare_records observed "
                    "(and fb payload has no future_requirement to match against)",
                    step=step,
                    weak_implementation=True,
                )
            return CheckResult.ok(
                f"stubbed-pass: {len(multi_dim)} multi-dim call(s) — weak proxy",
                stubbed=True,
                weak_implementation=True,
                step=step,
            )

        req_lc = str(future_req).lower()
        for c in call_log:
            tool = c.get("tool_name")
            args = c.get("args") or {}
            if tool in ("check_constraint", "compare_records"):
                # Serialize args to a string and search for the requirement token
                import json as _json
                try:
                    args_str = _json.dumps(args, default=str).lower()
                except (TypeError, ValueError):
                    args_str = str(args).lower()
                if req_lc in args_str:
                    return CheckResult.ok(
                        f"future_dependency_check: {tool} args reference {future_req!r}",
                        step=step,
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
                    if req_lc in str(f).lower() or str(f).lower() in req_lc:
                        return CheckResult.ok(
                            f"future_dependency_check: {tool} returned matching field {f!r}",
                            step=step,
                            via=tool,
                        )
        return CheckResult.fail(
            f"future_dependency_check: no call inspected {future_req!r}",
            step=step,
            future_requirement=future_req,
        )

    if step == "compliance_check":
        # E4: gold_state.hard_constraint typically declares
        # {compliance_required: True} (or similar). The model must have
        # verified compliance via check_constraint OR via a lookup that
        # returned a compliance field.
        hard = gold_state.get("hard_constraint") or gold_state.get("hard_constraints") or {}
        # Collect every constraint-key that mentions 'compliance'
        compliance_keys = [
            k for k in hard.keys() if "complian" in str(k).lower()
        ] if isinstance(hard, dict) else []
        # Also accept explicit step param
        if fb.get("compliance_key"):
            compliance_keys.append(fb["compliance_key"])

        if not compliance_keys:
            # Fallback: we know "compliance" must have been checked somehow.
            # Accept any check_constraint or any lookup whose returned
            # fields include a compliance-named field.
            for c in call_log:
                if c.get("tool_name") == "check_constraint":
                    import json as _json
                    args_str = _json.dumps(c.get("args") or {}, default=str).lower()
                    if "complian" in args_str:
                        return CheckResult.ok(
                            "compliance_check: check_constraint referenced compliance",
                            step=step,
                        )
                if c.get("tool_name") in ("lookup_record", "get_attribute"):
                    envelope = c.get("response") or {}
                    if envelope.get("status") != "ok":
                        continue
                    data = envelope.get("result") or {}
                    if c.get("tool_name") == "lookup_record":
                        fields = (data.get("fields") or {}).keys()
                    else:
                        attr = data.get("attribute") or ""
                        fields = [attr] if attr else []
                    if any("complian" in str(f).lower() for f in fields):
                        return CheckResult.ok(
                            f"compliance_check: {c.get('tool_name')} returned compliance field",
                            step=step,
                        )
            return CheckResult.fail(
                "compliance_check: no check_constraint or field lookup referenced compliance",
                step=step,
            )

        # We have specific compliance key(s) from gold_state — check each.
        # We accept two match levels:
        #   exact: full key name (e.g. "compliance_required") appears
        #   loose: the root token "complian" appears (catches
        #          "compliance_status", "is_compliant", etc.)
        for ck in compliance_keys:
            ck_lc = str(ck).lower()
            for c in call_log:
                tool = c.get("tool_name")
                args = c.get("args") or {}
                if tool == "check_constraint":
                    import json as _json
                    try:
                        args_str = _json.dumps(args, default=str).lower()
                    except (TypeError, ValueError):
                        args_str = str(args).lower()
                    if ck_lc in args_str or "complian" in args_str:
                        return CheckResult.ok(
                            f"compliance_check: check_constraint referenced compliance",
                            step=step,
                            compliance_key=ck,
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
                        f_lc = str(f).lower()
                        if ck_lc in f_lc or "complian" in f_lc:
                            return CheckResult.ok(
                                f"compliance_check: {tool} returned compliance-related field {f!r}",
                                step=step,
                                compliance_key=ck,
                                matching_field=f,
                            )
        return CheckResult.fail(
            f"compliance_check: compliance key(s) {compliance_keys} never inspected",
            step=step,
            compliance_keys=compliance_keys,
        )

    # Unknown step name — explicit stub with warning-level flag so the
    # metrics layer can surface it in QA without failing TCR.
    return CheckResult.ok(
        f"step name {step!r} not recognized — stubbed-as-pass",
        stubbed=True,
        unregistered=True,
        step=step,
    )


def _check_skip_candidate_inspection(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """D1-style: at least N distinct record_ids must have been looked up.

    Defensive on ``record_id`` shape (same pattern as
    ``trajectory._check_all_candidates_inspected``): strings go in the
    inspected set; list/tuple values contribute their string members;
    other shapes (dict, int) are silently skipped so one malformed
    arg doesn't abort the whole evaluator (2026-04-15 Bug C fix).
    """
    min_n = int(fb.get("min_candidates", 0))
    if min_n <= 0:
        return CheckResult.ok("min_candidates not specified")
    inspected: set = set()
    for c in result.get("call_log") or []:
        if c.get("tool_name") not in ("lookup_record", "get_attribute"):
            continue
        rid = (c.get("args") or {}).get("record_id")
        if isinstance(rid, str) and rid:
            inspected.add(rid)
        elif isinstance(rid, (list, tuple)):
            for sub in rid:
                if isinstance(sub, str) and sub:
                    inspected.add(sub)
        # else: skip — unhashable/unknown shape doesn't count as
        # candidate inspection evidence.
    if len(inspected) < min_n:
        return CheckResult.fail(
            f"only {len(inspected)} candidate(s) inspected, need {min_n}",
            inspected=sorted(inspected),
            min_required=min_n,
        )
    return CheckResult.ok(
        f"{len(inspected)} candidate(s) inspected (>= {min_n})",
        inspected=sorted(inspected),
    )


def _check_persist_after_contradiction(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """D5: model must NOT keep using `invalid_option` after the contradiction."""
    invalid = fb.get("invalid_option") or fb.get("option_id")
    if not invalid:
        return CheckResult.fail("invalid_option not specified")
    call_log = result.get("call_log") or []
    # Walk forward; once we see an error involving the invalid option,
    # any subsequent reference to it is a violation.
    contradiction_seen = False
    for c in call_log:
        args = c.get("args") or {}
        rid = args.get("record_id") or args.get("option_id")
        if not contradiction_seen:
            if c.get("response_status") == "error" and str(rid) == str(invalid):
                contradiction_seen = True
            continue
        # After contradiction
        if str(rid) == str(invalid):
            return CheckResult.fail(
                f"persisted with invalid option {invalid} after contradiction",
                invalid_option=invalid,
            )
    if contradiction_seen:
        return CheckResult.ok("did not persist with invalid option")
    # Never saw the invalid option fail at all — vacuously clean
    return CheckResult.ok(
        "no contradiction observed (vacuous)",
        invalid_option=invalid,
        vacuous=True,
    )


def _check_terminate_after_error(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """E5: model must not terminate immediately after an error without retry."""
    call_log = result.get("call_log") or []
    if not call_log:
        return CheckResult.ok("no calls — vacuous")
    last = call_log[-1]
    if last.get("response_status") != "error":
        return CheckResult.ok("did not end on an error")
    # Check if `without_retry` is True (it almost always is)
    if not fb.get("without_retry", True):
        return CheckResult.ok("retry was not required")
    # Last call was an error and no subsequent retry — fail
    return CheckResult.fail(
        "terminated immediately after an error",
        last_call_status="error",
    )


def _check_hallucinated_facts_not_in_passage(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """A02/A03/A05 — semantic hallucination check via LLM judge.

    Extracts the source passage from the task YAML (threaded via _task)
    and asks an LLM whether the model's response contains fabricated facts.
    Falls back to stubbed pass when AGENTFLOOR_LLM_JUDGE is not set.
    """
    from . import llm_judge

    task = result.get("_task") or {}
    passage = (task.get("initial_context") or {}).get("passage") or ""
    if not passage:
        passage = task.get("example_prompt") or ""
    final_text = result.get("final_text") or ""

    if not passage or not final_text:
        return CheckResult.ok(
            "hallucinated_facts: no passage or final_text available — vacuous",
        )

    return llm_judge.judge("hallucinated_facts_not_in_passage", {
        "passage": passage,
        "final_text": final_text,
        "task_id": task.get("task_id", ""),
    })


def _check_inconsistent_recovery_and_submission(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """E5 — semantic consistency check via LLM judge.

    The structural part (region matches recovery_path) is already enforced
    by the submission checker. This predicate checks the semantic part:
    does the model's prose reasoning match its actual recovery actions?
    """
    from . import llm_judge

    task = result.get("_task") or {}
    final_text = result.get("final_text") or ""
    messages = result.get("messages") or []

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(f"[{role}] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(f"[{role}] {block.get('text', '')}")
                    elif block.get("type") == "tool_use":
                        parts.append(f"[{role}:tool_call] {block.get('name', '')}({json.dumps(block.get('input', {}), default=str)})")
                    elif block.get("type") == "tool_result":
                        parts.append(f"[tool_result] {str(block.get('content', ''))[:500]}")

    trajectory_text = "\n".join(parts[-40:])

    call_log = result.get("call_log") or []
    submissions = [
        c for c in call_log if c.get("tool_name") == "submit_decision"
    ]
    submission_json = json.dumps(submissions[-1] if submissions else {}, default=str, indent=2)

    return llm_judge.judge("inconsistent_recovery_and_submission", {
        "trajectory_text": trajectory_text,
        "submission_json": submission_json,
        "task_id": task.get("task_id", ""),
    })


def _check_repeated_identical_call(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """D4: the model must not call the same tool with identical args
    repeatedly (>= threshold consecutive) — a loop symptom.

    Parameter: `threshold` (default 3) — number of consecutive identical
    (tool_name, args) calls that constitutes a violation.
    """
    import json as _json
    threshold = int(fb.get("threshold", 3))
    call_log = result.get("call_log") or []
    if len(call_log) < threshold:
        return CheckResult.ok(
            f"fewer than {threshold} total calls — vacuous",
            threshold=threshold,
        )

    def _fingerprint(c: dict) -> str:
        try:
            args_ser = _json.dumps(c.get("args") or {}, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_ser = str(c.get("args"))
        return f"{c.get('tool_name')}::{args_ser}"

    run_fp = None
    run_len = 0
    max_run = 0
    max_run_fp = None
    for c in call_log:
        fp = _fingerprint(c)
        if fp == run_fp:
            run_len += 1
        else:
            run_fp = fp
            run_len = 1
        if run_len > max_run:
            max_run = run_len
            max_run_fp = fp
    if max_run >= threshold:
        return CheckResult.fail(
            f"repeated identical call {max_run}x: {max_run_fp}",
            max_run=max_run,
            threshold=threshold,
            fingerprint=max_run_fp,
        )
    return CheckResult.ok(
        f"no run of {threshold}+ identical calls (max run = {max_run})",
        max_run=max_run,
    )


def _check_greedy_local_selection(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """E2: model must NOT submit the greedy-locally-optimal option.

    Parameter: `invalid_option` — the option ID that would be locally
    optimal but wrong globally (e.g. FW-1 for E2).
    """
    invalid = fb.get("invalid_option")
    if not invalid:
        return CheckResult.fail(
            "greedy_local_selection: invalid_option not specified",
        )
    call_log = result.get("call_log") or []
    submits = [c for c in call_log if c.get("tool_name") == "submit_decision"]
    if not submits:
        return CheckResult.ok("no submission — vacuous", invalid_option=invalid)
    last_submit_args = submits[-1].get("args") or {}
    # Check the common shapes: action, record_id, option_id, framework_id, etc.
    chosen = (
        last_submit_args.get("action")
        or last_submit_args.get("record_id")
        or last_submit_args.get("option_id")
        or last_submit_args.get("framework_id")
        or last_submit_args.get("vendor_id")
        or last_submit_args.get("selected")
    )
    if chosen is None:
        # As a fallback, scan all top-level string values for the invalid id
        for v in last_submit_args.values():
            if isinstance(v, str) and v == invalid:
                return CheckResult.fail(
                    f"submission contains greedy/invalid option {invalid!r}",
                    invalid_option=invalid,
                    submission=last_submit_args,
                )
        return CheckResult.ok(
            "submission does not reference invalid_option",
            invalid_option=invalid,
        )
    if str(chosen) == str(invalid):
        return CheckResult.fail(
            f"model submitted greedy/locally-optimal option {invalid!r}",
            invalid_option=invalid,
            chosen=chosen,
        )
    return CheckResult.ok(
        f"model avoided greedy option {invalid} (chose {chosen})",
        chosen=chosen,
        invalid_option=invalid,
    )


import re as _re


_AVERAGING_KEYWORDS = _re.compile(
    r"\b(?:average[ds]?|averag(?:ed|ing)|mean|midpoint|split\s+the\s+difference"
    r"|combin(?:e[ds]?|ing)\s+(?:the\s+)?(?:two|both|values?)"
    r"|halfway|middle\s+(?:value|ground|point))\b",
    _re.IGNORECASE,
)

_NEGATED_AVERAGING = _re.compile(
    r"\b(?:do\s+not|don't|dont|avoid|rather\s+than|instead\s+of|without)\b"
    r".{0,24}\b(?:average|mean|midpoint|halfway|split\s+the\s+difference)\b",
    _re.IGNORECASE | _re.DOTALL,
)


def _extract_all_assistant_text(result: dict) -> str:
    """Concatenate all assistant-role message text from the trajectory."""
    parts: list[str] = []
    for msg in result.get("messages") or []:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text") or "")
    return "\n".join(parts)


def _contains_literal_token(text: str, token: str) -> bool:
    """Return True iff ``token`` appears as a standalone literal in ``text``."""
    if not text or not token:
        return False
    return bool(
        _re.search(
            rf"(?<![A-Za-z0-9_]){_re.escape(token)}(?![A-Za-z0-9_])",
            text,
        )
    )


def _has_averaging_language(text: str) -> bool:
    """Detect affirmative averaging language while ignoring explicit rejections."""
    if not text or not _AVERAGING_KEYWORDS.search(text):
        return False
    return not _NEGATED_AVERAGING.search(text)


def _parse_precedence_rule(precedence_rule: Any) -> tuple[str | None, str | None]:
    """Parse rules like ``audit_overrides_profile`` into winner/loser prefixes."""
    if not precedence_rule:
        return (None, None)
    match = _re.fullmatch(r"([a-z0-9]+)_overrides_([a-z0-9]+)", str(precedence_rule).lower())
    if not match:
        return (None, None)
    return (match.group(1), match.group(2))


def _collect_conflicting_numeric_sets(result: dict, gold_state: dict) -> list[dict[str, Any]]:
    """Collect numeric conflict pairs from gold_state or observed lookup results."""
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    conflicting = gold_state.get("conflicting_fields") or {}
    target_value = gold_state.get("target_value")
    if isinstance(conflicting, dict):
        numeric_fields = {
            k: v for k, v in conflicting.items()
            if isinstance(k, str) and isinstance(v, (int, float))
        }
        if len(numeric_fields) >= 2:
            key = tuple(sorted(numeric_fields.items()))
            out.append(
                {
                    "source": "gold_state",
                    "fields": numeric_fields,
                    "target_value": target_value,
                }
            )
            seen.add(key)

    winner_prefix, loser_prefix = _parse_precedence_rule(gold_state.get("precedence_rule"))
    if not winner_prefix or not loser_prefix:
        return out

    for call in result.get("call_log") or []:
        if call.get("tool_name") != "lookup_record":
            continue
        envelope = call.get("response") or {}
        if envelope.get("status") != "ok":
            continue
        fields = ((envelope.get("result") or {}).get("fields") or {})
        if not isinstance(fields, dict):
            continue

        for winner_field, winner_value in fields.items():
            if not (
                isinstance(winner_field, str)
                and winner_field.startswith(winner_prefix + "_")
                and isinstance(winner_value, (int, float))
            ):
                continue

            suffix = winner_field[len(winner_prefix) + 1:]
            loser_field = f"{loser_prefix}_{suffix}"
            loser_value = fields.get(loser_field)
            if not isinstance(loser_value, (int, float)):
                continue

            numeric_fields = {
                loser_field: loser_value,
                winner_field: winner_value,
            }
            key = tuple(sorted(numeric_fields.items()))
            if key in seen:
                continue
            out.append(
                {
                    "source": "lookup_record",
                    "fields": numeric_fields,
                    "target_value": winner_value,
                }
            )
            seen.add(key)

    return out


def _text_selects_option(text: str, option: str) -> bool:
    """Heuristic: does free text actively endorse or choose ``option``?"""
    if not text or not option:
        return False
    opt = _re.escape(str(option))
    patterns = [
        rf"\b(?:select|selected|selecting|choose|chooses|chosen|chose|pick|picked|"
        rf"recommend|recommended|go\s+with|opt\s+for|use|used|using)\b"
        rf"(?:\s+\w+){{0,6}}\s+{opt}\b",
        rf"\b{opt}\b(?:\s+\w+){{0,6}}\b(?:is|was|looks|seems|remains)\b"
        rf"(?:\s+\w+){{0,6}}\b(?:best|chosen|selected|recommended|right|valid|winner)\b",
    ]
    return any(_re.search(pattern, text, _re.IGNORECASE) for pattern in patterns)


def _check_average_conflicting_values(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """D2/E4: model must apply a precedence rule, not average conflicting values.

    Deterministic check:
    1. Compute the arithmetic mean of the two conflicting values.
    2. If the mean appears in final_text or submission → fail.
    3. If averaging language appears AND the correct target_value is absent → fail.
    """
    conflicts = _collect_conflicting_numeric_sets(result, gold_state)
    if not conflicts:
        return CheckResult.ok(
            "no numeric conflicting fields available from gold_state or lookup_record "
            "responses — vacuous",
        )

    final_text = result.get("final_text") or ""
    all_text = _extract_all_assistant_text(result)
    search_text = final_text + "\n" + all_text

    for conflict in conflicts:
        conflicting = conflict["fields"]
        target_value = conflict.get("target_value")
        values = list(conflicting.values())
        mean_val = sum(values) / len(values)
        mean_strs = set()
        if mean_val == int(mean_val):
            mean_strs.add(str(int(mean_val)))
        else:
            mean_strs.add(f"{mean_val:.1f}")
            mean_strs.add(f"{mean_val:.2f}")
            mean_strs.add(str(mean_val))
        rounded = round(mean_val)
        mean_strs.add(str(rounded))

        for ms in mean_strs:
            if _contains_literal_token(search_text, ms):
                return CheckResult.fail(
                    f"model output contains average value {ms} of conflicting fields "
                    f"{conflicting} (expected precedence → {target_value})",
                    average_value=mean_val,
                    found_string=ms,
                    conflicting_fields=conflicting,
                    target_value=target_value,
                    conflict_source=conflict.get("source"),
                )

    target_values = {
        str(conflict["target_value"])
        for conflict in conflicts
        if conflict.get("target_value") is not None
    }
    has_averaging_language = _has_averaging_language(search_text)
    target_present = any(_contains_literal_token(final_text, value) for value in target_values)
    if has_averaging_language and not target_present:
        return CheckResult.fail(
            "averaging language detected and no precedence-selected target value "
            "appears in final_text",
            average_language=True,
            target_values=sorted(target_values),
            target_present=False,
        )

    return CheckResult.ok(
        "no averaging detected",
        conflicts=conflicts,
    )


def _check_constraint_drift(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """E3: model must not silently relax a constraint over turns.

    Deterministic check using ``invalid_option`` from the fb entry:
    1. If the model's final_text or submission selects the invalid option → fail.
    2. If a check_constraint call returned a violation and the model later
       used the violating option → fail.
    """
    invalid_option = fb.get("invalid_option")
    if not invalid_option:
        return CheckResult.ok(
            "constraint_drift: no invalid_option declared — vacuous",
        )

    invalid_str = str(invalid_option)
    final_text = (result.get("final_text") or "").strip()
    call_log = result.get("call_log") or []

    for c in call_log:
        if c.get("tool_name") == "submit_decision":
            args = c.get("args") or {}
            import json as _json
            try:
                args_str = _json.dumps(args, default=str)
            except (TypeError, ValueError):
                args_str = str(args)
            if invalid_str in args_str:
                return CheckResult.fail(
                    f"constraint_drift: model submitted with invalid option {invalid_str}",
                    invalid_option=invalid_str,
                    found_in="submit_decision_args",
                )

    if _text_selects_option(final_text, invalid_str):
        return CheckResult.fail(
            f"constraint_drift: final_text selects invalid option {invalid_str}",
            invalid_option=invalid_str,
            found_in="final_text_selection",
        )

    constraint_violations: list[str] = []
    for c in call_log:
        if c.get("tool_name") == "check_constraint":
            resp = c.get("response") or {}
            result_data = resp.get("result") or {}
            if result_data.get("satisfied") is False:
                constraint_violations.append(
                    result_data.get("constraint", "unknown")
                )

    if constraint_violations and _text_selects_option(final_text, invalid_str):
        return CheckResult.fail(
            f"constraint_drift: {len(constraint_violations)} constraint violation(s) "
            f"detected but model still references {invalid_str} in final answer",
            invalid_option=invalid_str,
            violations=constraint_violations,
            found_in="final_text_mention_after_violation",
        )

    return CheckResult.ok(
        f"constraint_drift: invalid option {invalid_str} not selected",
        invalid_option=invalid_str,
    )


def _check_partial_constraint_check(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """D3/E3: model checked some but not all required constraints.

    Two accepted forms of ``required_constraints`` on the forbidden entry:

      * **list of names** (e.g. ``[budget_max, gpu_required, uptime_min]``):
        every named constraint must appear in a ``check_constraint`` call.
      * **int N** (e.g. ``3``): the model must have emitted at least ``N``
        *distinct* constraint names across its ``check_constraint`` calls.
        Used by D3 and E3 whose hard-constraint identity is captured in
        the ``oracle_evaluator.trajectory_check.require_all_constraints_checked``
        list, so this entry only counts coverage.

    If ``required_constraints`` is missing / falsy, return a stubbed pass —
    same as before.
    """
    required = fb.get("required_constraints")
    if not required:
        return CheckResult.ok(
            "stubbed: partial_constraint_check has no required_constraints list",
            stubbed=True,
            not_implemented=True,
        )
    call_log = result.get("call_log") or []
    seen: set = set()
    for c in call_log:
        if c.get("tool_name") != "check_constraint":
            continue
        args = c.get("args") or {}
        cname = args.get("constraint") or args.get("constraint_name") or args.get("name")
        if isinstance(cname, str):
            seen.add(cname)
        inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else None
        if inputs:
            cname = inputs.get("constraint") or inputs.get("name")
            if isinstance(cname, str):
                seen.add(cname)

    if isinstance(required, int):
        # Count-based form (D3 / E3): fail if fewer than N distinct.
        if len(seen) < required:
            return CheckResult.fail(
                f"partial_constraint_check: only {len(seen)} distinct constraint(s) "
                f"checked, required {required}",
                seen=sorted(seen),
                required_count=required,
            )
        return CheckResult.ok(
            f"{len(seen)} distinct constraint(s) checked (>= {required} required)",
            seen=sorted(seen),
            required_count=required,
        )

    if not isinstance(required, (list, tuple)):
        # Unknown shape — fail loud rather than stub silently. Same
        # philosophy as the 2026-04-14 predicate-registration audit.
        return CheckResult.fail(
            f"partial_constraint_check: unexpected required_constraints shape "
            f"{type(required).__name__} (expected list or int)",
            received=repr(required)[:120],
        )

    missing = [n for n in required if n not in seen]
    if missing:
        return CheckResult.fail(
            f"missing constraint check(s): {', '.join(missing)}",
            missing=missing,
            seen=sorted(seen),
        )
    return CheckResult.ok(
        f"all {len(required)} constraint(s) checked",
        seen=sorted(seen),
    )


_PIPELINE_STAGE_TOOLS: dict[str, tuple[str, ...]] = {
    "candidate_search": ("search_records",),
    "search": ("search_records",),
    "record_verification": ("lookup_record", "get_attribute"),
    "conflict_resolution": ("compare_records",),
    "score_computation": ("compute_value",),
    "score": ("compute_value",),
    "compliance_check": ("check_constraint",),
    "final_submission": ("submit_decision",),
    "submit": ("submit_decision",),
}


def _check_skip_pipeline_stage(
    result: dict, fb: dict, gold_state: dict
) -> CheckResult:
    """E4: model must complete all pipeline stages in order.

    Checks the call_log for both presence and ordering of each stage's
    characteristic tool calls. Stages come from the task's
    ``require_all_pipeline_stages_completed`` (via ``_task``) or from
    ``fb.get("stages")``.
    """
    task = result.get("_task") or {}
    oracle = task.get("oracle_evaluator") or {}
    traj_check = oracle.get("trajectory_check") or {}
    stages = (
        fb.get("stages")
        or traj_check.get("require_all_pipeline_stages_completed")
    )

    if not stages or not isinstance(stages, list):
        return CheckResult.ok(
            "skip_pipeline_stage: no stages declared — vacuous",
        )

    call_log = result.get("call_log") or []
    tool_positions: dict[str, list[int]] = {}
    for idx, c in enumerate(call_log):
        tname = c.get("tool_name")
        if tname:
            tool_positions.setdefault(tname, []).append(idx)

    missing: list[str] = []
    out_of_order: list[str] = []
    stage_first_idx: list[tuple[str, int]] = []
    prev_idx = -1

    for stage_name in stages:
        tools = _PIPELINE_STAGE_TOOLS.get(stage_name)
        if not tools:
            missing.append(stage_name)
            continue

        all_indices = sorted(
            idx for t in tools for idx in tool_positions.get(t, [])
        )
        if not all_indices:
            missing.append(stage_name)
            continue

        next_idx = next((idx for idx in all_indices if idx > prev_idx), None)
        if next_idx is None:
            out_of_order.append(
                f"{stage_name}(indices={all_indices}) before boundary(idx>{prev_idx})"
            )
            continue

        stage_first_idx.append((stage_name, next_idx))
        prev_idx = next_idx

    if missing:
        return CheckResult.fail(
            f"skip_pipeline_stage: {len(missing)} stage(s) missing from call_log: "
            f"{', '.join(missing)}",
            missing_stages=missing,
            declared_stages=list(stages),
        )

    if out_of_order:
        return CheckResult.fail(
            f"skip_pipeline_stage: stages executed out of order: "
            f"{'; '.join(out_of_order)}",
            out_of_order=out_of_order,
            stage_order=[(n, i) for n, i in stage_first_idx],
        )

    return CheckResult.ok(
        f"skip_pipeline_stage: all {len(stages)} stages present and ordered",
        stages=[n for n, _ in stage_first_idx],
    )


# ----------------------------------------------------------------------
# Dispatch table
# ----------------------------------------------------------------------


_CHECKERS = {
    "hallucinated_tool": _check_hallucinated_tool,
    "terminate_without_answer": _check_terminate_without_answer,
    "label_not_in_valid_set": _check_label_not_in_valid_set,
    "hallucinated_facts_not_in_passage": _check_hallucinated_facts_not_in_passage,
    "skip_required_step": _check_skip_required_step,
    "skip_candidate_inspection": _check_skip_candidate_inspection,
    "persist_after_contradiction": _check_persist_after_contradiction,
    "terminate_after_error": _check_terminate_after_error,
    "inconsistent_recovery_and_submission": _check_inconsistent_recovery_and_submission,
    # Added 2026-04-14 audit — previously these types made D2/D3/D4/E2/E3/E4
    # unpassable because the unknown-type fallback fails closed.
    "repeated_identical_call": _check_repeated_identical_call,
    "greedy_local_selection": _check_greedy_local_selection,
    "average_conflicting_values": _check_average_conflicting_values,
    "constraint_drift": _check_constraint_drift,
    "partial_constraint_check": _check_partial_constraint_check,
    "skip_pipeline_stage": _check_skip_pipeline_stage,
}
