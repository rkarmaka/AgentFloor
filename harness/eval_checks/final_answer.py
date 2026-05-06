"""final_answer_check dispatcher.

Reads `oracle_evaluator.final_answer_check` from a task YAML and runs the
appropriate checker against `result.final_text`.

Four checker types (collapsed from five — `structured_match` and
`exact_field_match` merge into `field_match` since they only differ in
the optional tolerance dict shape):

  field_match        — every expected_field present via STRICT-then-LENIENT
                       (covers exact_field_match AND structured_match)
  numeric_match      — every expected_field's value appears as a number,
                       within absolute tolerance
  element_coverage   — every required_element appears as a substring;
                       max_sentences enforced if specified
  any_valid_solution — multi-answer; either defers to submission validity
                       (E1) or iterates a list of valid solutions (E5)

Special cases:
  - If final_text is None and the run terminated as `submitted`, the
    answer "lives" entirely in the submission payload — final_answer_check
    is automatically a pass with reason="answer in submission".
  - If final_text is None and termination is anything else, fail.
"""

from __future__ import annotations

from typing import Any

from . import CheckResult
from .text_extract import (
    extract_numbers,
    field_match_one,
    numeric_match_one,
    value_in_text,
)


# Tag words that imply a quantitative fact. If an element_coverage tag
# contains any of these, we accept it as covered when the model's text
# contains at least one numeric token, even if the literal tag word is
# absent. Covers cases like Qwen saying "47-minute outage" for a tag
# named "duration_or_impact".
_QUANTITY_TAG_WORDS = frozenset(
    {
        "duration",
        "impact",
        "count",
        "time",
        "rate",
        "quantity",
        "number",
        "amount",
        "percent",
        "size",
        "length",
        "age",
        "score",
        "value",
        "total",
        "cost",
        "price",
    }
)


def check_final_answer(
    result: dict,
    spec: dict,
    *,
    submission_passed: bool | None = None,
) -> CheckResult:
    """Top-level dispatcher.

    `submission_passed` is the outcome of `check_submission` for this
    same run; needed for the `any_valid_solution` E1 case which defers
    to the fixture's submission validator. Pass None if not relevant.
    """
    final_text = result.get("final_text")
    termination = result.get("termination")

    # Special-case: terminated via submitted means the answer lives in the
    # submission payload, not the prose. Defer to the submission checker.
    # This fires regardless of whether the assistant turn also produced
    # explanatory text (e.g. "Submitted approval.") — a model explaining
    # what it just submitted must not be failed for prose that doesn't
    # happen to contain the expected_fields literally.
    if termination == "submitted":
        if submission_passed is True:
            return CheckResult.ok(
                "answer encoded in submission",
                deferred_to="submission",
                had_final_text=bool(final_text),
            )
        if submission_passed is False:
            return CheckResult.fail(
                "submission was rejected",
                deferred_to="submission",
                had_final_text=bool(final_text),
            )
        # submission_passed is None — task does not require a submission
        # but the model terminated by calling submit_decision anyway. The
        # gold answer is in the prose, so fall through to normal text
        # checking rather than auto-passing on a submission we never
        # validated.

    if not final_text:
        return CheckResult.fail(
            "no final answer text",
            termination=termination,
        )

    t = spec.get("type")
    if t == "field_match" or t == "exact_field_match" or t == "structured_match":
        return _check_field_match(final_text, spec)
    if t == "numeric_match":
        return _check_numeric_match(final_text, spec)
    if t == "element_coverage":
        # Pass `result` through so the checker can access initial_context.passage
        # via _task. Required for the LLM-judge fallback when keyword matching
        # cannot resolve a required concept tag against paraphrased vocabulary
        # (see F-009: gpt-5 A02 vocabulary mismatch).
        return _check_element_coverage(final_text, spec, result=result)
    if t == "any_valid_solution":
        return _check_any_valid_solution(final_text, spec, submission_passed)
    return CheckResult.fail(
        f"unknown final_answer_check type: {t!r}",
        spec_type=t,
    )


# ----------------------------------------------------------------------
# Individual checker implementations
# ----------------------------------------------------------------------


def _match_nested_spec(key: str, spec: dict, text: str) -> tuple[bool, dict]:
    """Handle a nested `{type, expected}` field spec.

    Currently supports:
      - set_match: all items in `expected` must appear (value_in_text)
                   anywhere in the text, in any order.

    Returns (matched, details). Unknown `type` → returns (False, ...)
    so mis-specified tasks fail loud rather than silently passing.
    """
    t = spec.get("type")
    expected = spec.get("expected")

    if t == "set_match":
        if not isinstance(expected, (list, tuple)):
            return False, {
                "matched_field": key,
                "match_mode": "set_match_invalid_spec",
                "note": f"expected must be a list; got {type(expected).__name__}",
            }
        missing = [e for e in expected if not value_in_text(e, text)]
        if missing:
            return False, {
                "matched_field": key,
                "match_mode": "set_match_missing",
                "missing": missing,
                "expected": list(expected),
            }
        return True, {
            "matched_field": key,
            "match_mode": "set_match",
            "n_matched": len(expected),
        }

    return False, {
        "matched_field": key,
        "match_mode": "unknown_nested_spec",
        "type": t,
    }


def _check_field_match(text: str, spec: dict) -> CheckResult:
    """Strategy B (strict) with Strategy A fallback (lenient).

    All fields in `expected_fields` must match (each via field_match_one).
    Per-field tolerance is honored when both spec.tolerance is a dict and
    the field name appears in it.
    """
    expected = spec.get("expected_fields") or {}
    if not expected:
        return CheckResult.fail("no expected_fields in spec")

    tolerance = spec.get("tolerance")
    per_field: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    weak_matches: list[str] = []

    for key, value in expected.items():
        # Nested structured-spec form: `{type: <match_mode>, expected: ...}`.
        # Used by A4 (`ids: {type: set_match, expected: [SP-1, SP-2, ...]}`).
        # Without this branch, the dict is passed to field_match_one which
        # looks for the stringified dict literally — always fails.
        if isinstance(value, dict) and "type" in value and "expected" in value:
            ok, det = _match_nested_spec(key, value, text)
            per_field[key] = det
            if not ok:
                failures.append(key)
            continue

        # Numeric tolerance handling: if tolerance is a dict and this
        # key has a numeric tolerance, use numeric_match_one instead of
        # plain field_match_one. This covers structured_match cases.
        per_field_tol = None
        if isinstance(tolerance, dict):
            per_field_tol = tolerance.get(key)
        elif isinstance(tolerance, (int, float)):
            per_field_tol = float(tolerance)

        if per_field_tol is not None and isinstance(value, (int, float)):
            ok, det = numeric_match_one(key, float(value), text, tolerance=float(per_field_tol))
            per_field[key] = {"mode": "numeric", **det}
            if not ok:
                failures.append(key)
        else:
            ok, mode, det = field_match_one(key, value, text)
            per_field[key] = det
            if not ok:
                failures.append(key)
            elif mode == "lenient":
                weak_matches.append(key)

    if failures:
        return CheckResult.fail(
            f"missing field(s): {', '.join(failures)}",
            per_field=per_field,
            failed_fields=failures,
        )
    reason = "all fields matched"
    if weak_matches:
        reason += f" ({len(weak_matches)} weak)"
    return CheckResult.ok(
        reason,
        per_field=per_field,
        weak_match_fields=weak_matches,
    )


def _check_numeric_match(text: str, spec: dict) -> CheckResult:
    """Each expected field's value must appear in the text as a number."""
    expected = spec.get("expected_fields") or {}
    if not expected:
        return CheckResult.fail("no expected_fields in numeric_match spec")
    tolerance = float(spec.get("tolerance", 0))

    failures: list[str] = []
    per_field: dict[str, dict[str, Any]] = {}
    for key, value in expected.items():
        try:
            ok, det = numeric_match_one(key, float(value), text, tolerance=tolerance)
        except (TypeError, ValueError):
            return CheckResult.fail(
                f"non-numeric expected value for field {key!r}: {value!r}"
            )
        per_field[key] = det
        if not ok:
            failures.append(key)

    if failures:
        return CheckResult.fail(
            f"missing numeric value(s) for: {', '.join(failures)}",
            per_field=per_field,
        )
    return CheckResult.ok("all numeric fields matched", per_field=per_field)


def _check_element_coverage(
    text: str,
    spec: dict,
    result: dict | None = None,
) -> CheckResult:
    """All required_elements must appear (literally or semantically).

    Two-stage matching:

    1. Keyword-based (cheap, deterministic): try the raw token, the
       humanized form, partial-token match (any underscore-separated
       part >= 4 chars as a standalone word), and a quantity-tag
       fallback (numeric content in text + a quantity word in the tag).

    2. LLM-judge fallback (only when result with passage context is
       provided AND keyword matching missed an element): per missing
       element, ask the judge whether the model's response semantically
       covers the concept given the source passage. Cached via
       results/llm_judge_cache.jsonl. Stubbed (default-pass with
       stubbed=True) when AGENTFLOOR_LLM_JUDGE != 1, so the canonical
       v0-v5 baseline runs are never silently inflated by the fallback.

    Motivating case: gpt-5 A02 traces produce
    technically-correct summaries using HTTP-layer vocabulary
    ("checkouts", "503", "connection pool") that don't trigger
    keyword matching against `payment_service_failure`. With the LLM
    judge enabled the matcher recognizes the paraphrase.
    """
    elements = spec.get("required_elements") or []
    if not elements:
        return CheckResult.fail("no required_elements in spec")
    max_sentences = spec.get("max_sentences")

    # Stage 1: keyword-based matching (existing logic)
    missing: list[str] = []
    weak: list[str] = []
    text_has_numbers = bool(extract_numbers(text))
    for el in elements:
        # Try the raw token first, then a humanized variant
        if value_in_text(el, text):
            continue
        humanized = str(el).replace("_", " ")
        if humanized != el and value_in_text(humanized, text):
            continue
        # Partial-token matching: any of the underscore-separated parts
        # being a standalone word in the text counts as coverage.
        parts = [p for p in str(el).split("_") if len(p) >= 4]
        if any(value_in_text(p, text) for p in parts):
            continue
        # Quantity-tag fallback: if the tag contains a quantity word
        # AND the text has at least one numeric token, accept as a
        # weak match.
        tag_parts_lower = {p.lower() for p in str(el).split("_")}
        if text_has_numbers and (tag_parts_lower & _QUANTITY_TAG_WORDS):
            weak.append(el)
            continue
        missing.append(el)

    # Stage 2: LLM-judge fallback for missing elements.
    # Only fires when (a) caller threaded the result through and (b) the
    # task carries a passage in initial_context. Caching ensures repeat
    # calls on the same (passage, response, element) are free.
    llm_judged: list[str] = []
    llm_unparseable: list[str] = []
    if missing and result is not None:
        task = result.get("_task") or {}
        passage = (task.get("initial_context") or {}).get("passage", "")
        task_id = task.get("task_id", "")
        if passage:
            from .llm_judge import judge
            still_missing: list[str] = []
            for el in missing:
                cr = judge("element_coverage", {
                    "passage": passage,
                    "final_text": text,
                    "element": el,
                    "task_id": task_id,
                })
                # cr.passed=True with details.stubbed=True means the env
                # var is off (default-pass stub). That MUST NOT close
                # the gap silently — keep the element in the missing list
                # so v0-v5 baseline scores aren't inflated.
                if cr.passed and cr.details.get("stubbed"):
                    still_missing.append(el)
                    if cr.details.get("unparseable"):
                        llm_unparseable.append(el)
                elif cr.passed:
                    llm_judged.append(el)
                else:
                    still_missing.append(el)
            missing = still_missing

    sentence_violation = None
    if max_sentences:
        n_sent = sum(1 for ch in text if ch in ".!?")
        if n_sent > max_sentences:
            sentence_violation = f"{n_sent} sentences > max {max_sentences}"

    if missing:
        return CheckResult.fail(
            f"missing element(s): {', '.join(missing)}",
            missing=missing,
            weak_elements=weak,
            llm_judged_elements=llm_judged,
            llm_unparseable_elements=llm_unparseable,
            sentence_violation=sentence_violation,
        )
    if sentence_violation:
        return CheckResult.fail(
            sentence_violation,
            missing=[],
            weak_elements=weak,
            llm_judged_elements=llm_judged,
            llm_unparseable_elements=llm_unparseable,
            sentence_violation=sentence_violation,
        )
    reason = "all elements present"
    bits = []
    if weak:
        bits.append(f"{len(weak)} via quantity-tag fallback")
    if llm_judged:
        bits.append(f"{len(llm_judged)} via LLM judge")
    if bits:
        reason += " (" + "; ".join(bits) + ")"
    return CheckResult.ok(
        reason,
        n_elements=len(elements),
        weak_elements=weak,
        llm_judged_elements=llm_judged,
    )


def _check_any_valid_solution(
    text: str, spec: dict, submission_passed: bool | None
) -> CheckResult:
    """Multi-answer: either defers to submission, or iterates valid_solutions.

    Two sub-shapes:
      E1: spec has `canonical_solution` + `equivalence_rule: bundle_validator_passes`.
          → if submission_passed is True, the bundle satisfied the fixture
            validator and we accept any payload it accepted.
      E5: spec has `valid_solutions: [list of dicts]`.
          → check that all key/value pairs of at least one alternative
            appear in the text.
    """
    eq_rule = spec.get("equivalence_rule")
    if eq_rule == "bundle_validator_passes":
        if submission_passed is True:
            return CheckResult.ok(
                "deferred to submission validator (bundle accepted)",
                equivalence_rule=eq_rule,
            )
        return CheckResult.fail(
            "bundle_validator_passes required but submission did not pass",
            equivalence_rule=eq_rule,
        )

    valid = spec.get("valid_solutions") or []
    if valid:
        for i, sol in enumerate(valid):
            if not isinstance(sol, dict):
                continue
            ok = True
            for k, v in sol.items():
                if not value_in_text(v, text):
                    ok = False
                    break
            if ok:
                return CheckResult.ok(
                    f"matched valid_solutions[{i}]",
                    matched_index=i,
                )
        return CheckResult.fail(
            f"none of {len(valid)} valid solutions matched",
            num_alternatives=len(valid),
        )

    # Has neither equivalence_rule nor valid_solutions — malformed spec
    return CheckResult.fail(
        "any_valid_solution spec missing both equivalence_rule and valid_solutions"
    )
