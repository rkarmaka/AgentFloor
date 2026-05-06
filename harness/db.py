"""In-memory database layer for the SLM agentic capability benchmark.

Loads a single fixture YAML file into memory and exposes query methods
that the 8 tools call into. One instance per task run; no shared state
across runs.

The DB returns raw structured responses (status / result / error) — the
tool layer (built later) wraps these in the full envelope with
schema_version, tool_name, and call_id.

Supported tools (8):
    search_records, lookup_record, get_attribute, list_options,
    check_constraint, compare_records, compute_value, submit_decision

Requires Python 3.10+ (uses match statements and X | Y type hints).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Operator dispatch table for constraint evaluation
_OPERATORS = {
    "lt":     lambda a, b: a < b,
    "lte":    lambda a, b: a <= b,
    "gt":     lambda a, b: a > b,
    "gte":    lambda a, b: a >= b,
    "eq":     lambda a, b: a == b,
    "neq":    lambda a, b: a != b,
    "in":     lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}


def _ok(result: dict) -> dict:
    return {"status": "ok", "result": result, "error": None}


def _not_found(error_type: str, message: str, hint: str | None = None) -> dict:
    err = {"type": error_type, "message": message, "recoverable": True}
    if hint:
        err["hint"] = hint
    return {"status": "not_found", "result": None, "error": err}


def _error(
    error_type: str,
    message: str,
    *,
    recoverable: bool = False,
    hint: str | None = None,
) -> dict:
    err = {"type": error_type, "message": message, "recoverable": recoverable}
    if hint:
        err["hint"] = hint
    return {"status": "error", "result": None, "error": err}


class FixtureDB:
    """In-memory database loaded from a single fixture YAML file.

    One instance per task run. Construct with the fixture path (or None
    for tasks that don't need a fixture, like A0 and A3).
    """

    def __init__(self, fixture_path: str | Path | None):
        if fixture_path is None:
            self.fixture_path: Path | None = None
            self.data: dict = {}
        else:
            self.fixture_path = Path(fixture_path)
            self.data = self._load(self.fixture_path)
            self.validate_fixture(self.data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_id(self) -> str | None:
        return self.data.get("task_id")

    @property
    def domain(self) -> str | None:
        return self.data.get("domain")

    @property
    def is_empty(self) -> bool:
        return self.fixture_path is None

    # ------------------------------------------------------------------
    # File loading & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Fixture {path} did not parse as a dict")
        return data

    @staticmethod
    def validate_fixture(data: dict) -> None:
        """Lightweight schema check. Raises ValueError on structural problems."""
        for key in ("task_id", "domain"):
            if key not in data:
                raise ValueError(f"Fixture missing required field: {key}")

        constraints = data.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be a dict")
        for name, c in constraints.items():
            if not isinstance(c, dict):
                continue
            # Constraints with an explicit operator must use a known one.
            # Constraints with a 'rule' field instead are complex and skipped.
            if "operator" in c and c["operator"] not in _OPERATORS:
                raise ValueError(
                    f"Constraint {name} uses unsupported operator: {c['operator']}"
                )

        for failure in data.get("scripted_failures") or []:
            if "tool" not in failure:
                raise ValueError("scripted_failure missing 'tool' field")
            if "response" not in failure:
                raise ValueError(
                    f"scripted_failure for {failure['tool']} missing 'response'"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_db_error(self, tool: str) -> dict:
        return _error(
            "no_fixture_loaded",
            f"Cannot call {tool} — this task has no fixture loaded",
            recoverable=False,
        )

    def _find_scripted_failure(
        self, tool_name: str, args: dict
    ) -> dict | None:
        for failure in self.data.get("scripted_failures") or []:
            if failure.get("tool") != tool_name:
                continue
            when = failure.get("when_args") or {}
            if all(args.get(k) == v for k, v in when.items()):
                return failure
        return None

    def _evaluate_constraint(
        self, record: dict, constraint: dict
    ) -> tuple[bool, str | None, Any]:
        """Returns (passes, violation_label_or_None, actual_value)."""
        field = constraint.get("field")
        op = constraint.get("operator")
        if op is None:
            rule = constraint.get("rule")
            if rule == "current_component_version_supported":
                version = record.get("current_component_version")
                if version is None:
                    return False, "missing_field:current_component_version", None
                versions = self.data.get("component_versions") or {}
                version_info = versions.get(version)
                if version_info is None:
                    return False, f"unknown_component_version:{version}", version
                compatible_regions = version_info.get("compatible_regions") or []
                passes = len(compatible_regions) > 0
                return (
                    passes,
                    None if passes else f"incompatible_version:{version}",
                    version,
                )
            # Complex rule-based constraint without a DB implementation
            return False, f"unsupported_constraint_type:{rule}", None
        if field is None:
            return False, "constraint_definition_invalid", None
        expected = constraint.get("value")
        if field not in record:
            return False, f"missing_field:{field}", None
        actual = record[field]
        if actual is None:
            return False, f"null_value:{field}", None
        if op not in _OPERATORS:
            return False, f"unsupported_operator:{op}", actual
        passes = _OPERATORS[op](actual, expected)
        return passes, None if passes else f"{field}_violation", actual

    # ------------------------------------------------------------------
    # Tool query methods
    # ------------------------------------------------------------------

    def lookup_record(
        self, record_id: str, fields: list[str] | None = None
    ) -> dict:
        if self.is_empty:
            return self._empty_db_error("lookup_record")
        records = self.data.get("records") or {}
        record = records.get(record_id)
        if record is None:
            return _not_found(
                "record_not_found",
                f"No record matches ID '{record_id}'",
                hint="Use search_records to find valid IDs",
            )
        if fields:
            filtered = {k: record[k] for k in fields if k in record}
            missing = [k for k in fields if k not in record]
            result = {"record_id": record_id, "fields": filtered}
            if missing:
                result["missing_fields"] = missing
            return _ok(result)
        return _ok({"record_id": record_id, "fields": dict(record)})

    def search_records(self, query: str) -> dict:
        if self.is_empty:
            return self._empty_db_error("search_records")
        # Scripted failures take precedence over the search index
        failure = self._find_scripted_failure("search_records", {"query": query})
        if failure:
            return failure["response"]
        index = self.data.get("search_index") or {}
        record_ids = index.get(query, [])
        records = self.data.get("records") or {}
        matches = [
            {"id": rid, "name": records.get(rid, {}).get("name", "")}
            for rid in record_ids
        ]
        return _ok({"matches": matches})

    def get_attribute(self, record_id: str, attribute: str) -> dict:
        if self.is_empty:
            return self._empty_db_error("get_attribute")
        records = self.data.get("records") or {}
        record = records.get(record_id)
        if record is None:
            return _not_found(
                "record_not_found",
                f"No record matches ID '{record_id}'",
                hint="Use search_records to find valid IDs",
            )
        if attribute not in record:
            return _not_found(
                "missing_field",
                f"Attribute '{attribute}' not present on record '{record_id}'",
                hint=f"Available fields: {sorted(record.keys())}",
            )
        return _ok(
            {
                "record_id": record_id,
                "attribute": attribute,
                "value": record[attribute],
            }
        )

    def list_options(self, option_set: str | None = None) -> dict:
        if self.is_empty:
            return self._empty_db_error("list_options")
        options = self.data.get("options") or {}
        if not options:
            return _ok({"options": []})
        if option_set:
            chosen = options.get(option_set)
            if chosen is None:
                return _not_found(
                    "option_set_not_found",
                    f"No option set '{option_set}'",
                    hint=f"Available option sets: {sorted(options.keys())}",
                )
            return _ok({"options": list(chosen)})
        # Default: return the first option set
        first_key = next(iter(options))
        return _ok({"options": list(options[first_key])})

    def check_constraint(self, record_id: str, constraint_name: str) -> dict:
        if self.is_empty:
            return self._empty_db_error("check_constraint")
        records = self.data.get("records") or {}
        record = records.get(record_id)
        if record is None:
            return _not_found(
                "record_not_found",
                f"No record matches ID '{record_id}'",
            )
        failure = self._find_scripted_failure(
            "check_constraint",
            {
                "record_id": record_id,
                "constraint_name": constraint_name,
            },
        )
        if failure:
            return failure["response"]
        constraints = self.data.get("constraints") or {}
        constraint = constraints.get(constraint_name)
        if constraint is None:
            return _not_found(
                "constraint_not_found",
                f"No constraint named '{constraint_name}'",
                hint=f"Available constraints: {sorted(constraints.keys())}",
            )
        passes, violation, actual = self._evaluate_constraint(record, constraint)
        return _ok(
            {
                "satisfied": passes,
                "violations": [] if passes else [violation],
                "actual_value": actual,
                "expected_value": constraint.get("value"),
                "operator": constraint.get("operator"),
            }
        )

    def compare_records(
        self,
        record_ids: list[str],
        criterion: str,
        direction: str = "desc",
    ) -> dict:
        if self.is_empty:
            return self._empty_db_error("compare_records")
        records = self.data.get("records") or {}
        found = [(rid, records[rid]) for rid in record_ids if rid in records]
        if not found:
            return _error(
                "no_records_found",
                f"None of the IDs {record_ids} exist",
                recoverable=False,
            )
        comparison = [
            {"id": rid, "criterion_value": rec.get(criterion)}
            for rid, rec in found
        ]
        sortable = [c for c in comparison if c["criterion_value"] is not None]
        sortable.sort(
            key=lambda c: c["criterion_value"], reverse=(direction == "desc")
        )
        best_id = sortable[0]["id"] if sortable else None
        return _ok(
            {
                "comparison": comparison,
                "best_id": best_id,
                "criterion": criterion,
                "direction": direction,
            }
        )

    def compute_value(self, operation: str, inputs: dict) -> dict:
        # Stateless — does not read from self.data
        try:
            match operation:
                case "multiply":
                    value = inputs["a"] * inputs["b"]
                case "add":
                    value = inputs["a"] + inputs["b"]
                case "subtract":
                    value = inputs["a"] - inputs["b"]
                case "divide":
                    if inputs["b"] == 0:
                        return _error(
                            "division_by_zero",
                            "Cannot divide by zero",
                        )
                    value = inputs["a"] / inputs["b"]
                case "weighted_score":
                    values = inputs.get("values") or {}
                    weights = inputs.get("weights") or {}
                    if not values or not weights:
                        return _error(
                            "missing_inputs",
                            "weighted_score requires both 'values' and 'weights'",
                            recoverable=True,
                        )
                    value = sum(
                        values[k] * weights.get(k, 0) for k in values
                    )
                case _:
                    return _error(
                        "unknown_operation",
                        f"compute_value does not support operation '{operation}'",
                        recoverable=True,
                        hint="Supported: multiply, add, subtract, divide, weighted_score",
                    )
        except KeyError as e:
            return _error(
                "missing_input",
                f"compute_value missing required input: {e.args[0]}",
                recoverable=True,
            )
        except (TypeError, ValueError) as e:
            return _error("invalid_input", str(e), recoverable=True)

        return _ok(
            {"value": value, "operation": operation, "inputs_used": inputs}
        )

    def submit_decision(self, **submission) -> dict:
        if self.is_empty:
            return self._empty_db_error("submit_decision")
        validator = self.data.get("submission_validator")
        if not validator:
            return _error(
                "no_validator_defined",
                "This fixture has no submission_validator",
            )
        matched, mismatch_reason = self._validate_submission(submission, validator)
        if matched:
            on_match = validator.get("on_match") or {"accepted": True}
            return _ok(on_match)
        on_mismatch = validator.get("on_mismatch") or {}
        return _error(
            on_mismatch.get("error_type", "invalid_submission"),
            mismatch_reason or "Submission did not match validator",
            recoverable=True,
        )

    # ------------------------------------------------------------------
    # Submission validators (4 shapes)
    # ------------------------------------------------------------------

    def _validate_submission(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        if "branch_rules" in validator:
            return self._validate_branch_submission(submission, validator)
        if "expected_bundle" in validator:
            return self._validate_bundle_submission(submission, validator)
        if "consistency_rules" in validator:
            return self._validate_consistency_submission(submission, validator)
        return self._validate_simple_submission(submission, validator)

    def _check_required_fields(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        """Verify every name in `required_fields` is present and non-null."""
        required = validator.get("required_fields") or []
        for field in required:
            if field not in submission or submission[field] is None:
                return False, f"missing required field: '{field}'"
        return True, None

    def _validate_simple_submission(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        # Step 1: required_fields enforcement
        ok, reason = self._check_required_fields(submission, validator)
        if not ok:
            return False, reason

        # Step 2: every expected_* key in the validator must match the
        # corresponding submission field. The mapping is mechanical:
        #   expected_action      → submission["action"]
        #   expected_record_id   → submission["record_id"]
        #   expected_partner_id  → submission["partner_id"]
        #   expected_framework_id → submission["framework_id"]
        # Reserved keys (handled by other validators) are skipped.
        reserved = {"expected_bundle"}
        for key, expected_value in validator.items():
            if not key.startswith("expected_") or key in reserved:
                continue
            submission_field = key[len("expected_"):]
            actual = submission.get(submission_field)
            if actual != expected_value:
                return False, (
                    f"{submission_field} mismatch: expected "
                    f"'{expected_value}', got '{actual}'"
                )
        return True, None

    def _validate_branch_submission(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        """Evaluate branch_rules against runtime state and check the submission.

        branch_rules is a list of {when: <predicate>, expected_*: ...} blocks.
        The first branch whose `when` predicate matches the underlying record
        state determines which expected_* fields the submission must match.

        If the submission carries a ``record_id`` field, branch matching is
        scoped to rules whose ``when.record_id`` equals that value. This lets
        a fixture host branch_rules for multiple records simultaneously
        without the first-match-wins behavior collapsing onto whichever rule
        happens to be listed first. When the submission omits ``record_id``
        the original first-match-wins behavior is preserved (so existing
        v0-v5 prompts that submit only ``action`` keep working unchanged).
        """
        ok, reason = self._check_required_fields(submission, validator)
        if not ok:
            return False, reason

        branch_rules = validator.get("branch_rules") or []
        if not isinstance(branch_rules, list):
            return False, (
                "branch_rules must be a list of {when, expected_*} blocks "
                "(legacy dict form is no longer supported)"
            )

        submission_record_id = submission.get("record_id")
        matched_branch = None
        for branch in branch_rules:
            if not isinstance(branch, dict):
                continue
            when = branch.get("when") or {}
            if (
                submission_record_id is not None
                and when.get("record_id") != submission_record_id
            ):
                continue
            if self._branch_predicate_matches(when):
                matched_branch = branch
                break

        if matched_branch is None:
            return False, "no branch_rule matched runtime state"

        # Validate the submission against the matched branch's expected_* fields
        for key, expected_value in matched_branch.items():
            if not key.startswith("expected_"):
                continue
            submission_field = key[len("expected_"):]
            actual = submission.get(submission_field)
            if actual != expected_value:
                return False, (
                    f"branch matched {matched_branch.get('when')}: expected "
                    f"{submission_field}='{expected_value}', got '{actual}'"
                )
        return True, None

    def _branch_predicate_matches(self, when: dict) -> bool:
        """Evaluate a branch `when` predicate against fixture state.

        Supported predicate types:

        - **Field equality**: `{record_id, field, equals: <value>}` or
          `{record_id, field, not_equals: <value>}`. Looks up the record
          and checks the named field against the expected value.
        - **Constraint**: `{record_id, constraint: <name>, satisfied: bool}`.
          Evaluates the named constraint against the record and matches if
          the result equals `satisfied`.
        - **Computed score**: `{record_id, computed: <formula>, operator,
          value}`. Computes the named scoring formula on the record and
          applies the operator against the value.
        """
        record_id = when.get("record_id")
        records = self.data.get("records") or {}
        record = records.get(record_id)
        if record is None:
            return False

        # Field equality predicate
        if "field" in when:
            field = when["field"]
            if field not in record:
                return False
            actual = record[field]
            if "equals" in when:
                return actual == when["equals"]
            if "not_equals" in when:
                return actual != when["not_equals"]
            return False

        # Constraint predicate
        if "constraint" in when:
            constraint_name = when["constraint"]
            constraints = self.data.get("constraints") or {}
            constraint = constraints.get(constraint_name)
            if constraint is None:
                return False
            passes, _, _ = self._evaluate_constraint(record, constraint)
            return passes == when.get("satisfied", True)

        # Computed score predicate (e.g., weighted_score against threshold)
        if "computed" in when:
            scoring_rule = self.data.get("scoring_rule") or {}
            if scoring_rule.get("formula") != when["computed"]:
                return False
            weights = scoring_rule.get("weights") or {}
            try:
                score = sum(record.get(k, 0) * w for k, w in weights.items())
            except (TypeError, KeyError):
                return False
            op = when.get("operator", "gt")
            value = when.get("value")
            if op not in _OPERATORS:
                return False
            return _OPERATORS[op](score, value)

        return False

    def _validate_bundle_submission(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        # Step 1: required_fields must be present and non-null
        ok, reason = self._check_required_fields(submission, validator)
        if not ok:
            return False, reason

        # Step 2: bundle_validator checks (compute + checks). Optional —
        # fixtures without a bundle_validator block fall through to step 3.
        bundle_validator = validator.get("bundle_validator")
        if bundle_validator:
            ok, reason = self._validate_bundle_with_compute(
                submission, bundle_validator
            )
            if not ok:
                return False, reason

        # Step 3: equivalence handling. If `allow_any_valid_bundle` is set,
        # any bundle that has passed all bundle_validator checks above is
        # accepted (the checks ARE the definition of validity). Otherwise,
        # require strict equality with `expected_bundle`. The expected_bundle
        # block is preserved for documentation in either case.
        if validator.get("allow_any_valid_bundle"):
            return True, None

        expected = validator.get("expected_bundle") or {}
        for key, val in expected.items():
            if submission.get(key) != val:
                return False, (
                    f"bundle field '{key}' mismatch: "
                    f"expected '{val}', got '{submission.get(key)}'"
                )
        return True, None

    def _validate_bundle_with_compute(
        self, submission: dict, bundle_validator: dict
    ) -> tuple[bool, str | None]:
        """Run derived computations and bundle-level checks.

        bundle_validator format:
            compute:
              <name>:
                formula: sum
                inputs:
                  - {record_id_from: <submission_field>, field: <field_name>}
            checks:
              - {computed: <name>, operator, value}             # derived value
              - {lookup_record_from: <submission_field>,        # cross-record
                 field, operator, value}
              - {rule: gpu_in_chassis_compatible_list,          # rule-based
                 gpu_from, chassis_from}
        """
        # Step 1: compute derived values from the submission
        computed_values: dict[str, Any] = {}
        compute_block = bundle_validator.get("compute") or {}
        for name, spec in compute_block.items():
            formula = spec.get("formula")
            inputs = spec.get("inputs") or []
            if formula == "sum":
                total: float = 0
                for inp in inputs:
                    record_id_field = inp.get("record_id_from")
                    field = inp.get("field")
                    record_id = submission.get(record_id_field)
                    if record_id is None:
                        return False, (
                            f"compute {name}: submission missing field "
                            f"'{record_id_field}'"
                        )
                    record = (self.data.get("records") or {}).get(record_id)
                    if record is None:
                        return False, (
                            f"compute {name}: record '{record_id}' not found"
                        )
                    if field not in record:
                        return False, (
                            f"compute {name}: field '{field}' missing on "
                            f"record '{record_id}'"
                        )
                    total += record[field]
                computed_values[name] = total
            else:
                return False, (
                    f"compute {name}: unsupported formula '{formula}'"
                )

        # Step 2: run all checks; collect violations rather than short-circuit
        checks = bundle_validator.get("checks") or []
        violations: list[str] = []
        for check in checks:
            ok, reason = self._evaluate_bundle_check(
                check, submission, computed_values
            )
            if not ok:
                violations.append(reason or "unspecified violation")

        if violations:
            return False, "; ".join(violations)
        return True, None

    def _evaluate_bundle_check(
        self,
        check: dict,
        submission: dict,
        computed_values: dict,
    ) -> tuple[bool, str | None]:
        """Evaluate a single bundle-level check."""
        # Computed-value check (e.g., total_cost <= 4000)
        if "computed" in check:
            name = check["computed"]
            if name not in computed_values:
                return False, f"check references unknown computed value: {name}"
            actual = computed_values[name]
            op = check.get("operator")
            expected = check.get("value")
            if op not in _OPERATORS:
                return False, f"check uses unsupported operator: {op}"
            if not _OPERATORS[op](actual, expected):
                return False, (
                    f"{name} violation (actual={actual}, expected {op} {expected})"
                )
            return True, None

        # Cross-record field check (e.g., vendor.approved == true)
        if "lookup_record_from" in check:
            record_id_field = check["lookup_record_from"]
            record_id = submission.get(record_id_field)
            if record_id is None:
                return False, (
                    f"submission missing field '{record_id_field}'"
                )
            record = (self.data.get("records") or {}).get(record_id)
            if record is None:
                return False, f"record '{record_id}' not found"
            field = check.get("field")
            if field not in record:
                return False, (
                    f"field '{field}' missing on record '{record_id}'"
                )
            actual = record[field]
            op = check.get("operator")
            expected = check.get("value")
            if op not in _OPERATORS:
                return False, f"unsupported operator: {op}"
            if not _OPERATORS[op](actual, expected):
                return False, (
                    f"{record_id}.{field} {op} {expected} failed "
                    f"(actual={actual})"
                )
            return True, None

        # Rule-based check
        if "rule" in check:
            rule = check["rule"]
            if rule == "gpu_in_chassis_compatible_list":
                gpu_id = submission.get(check.get("gpu_from", "gpu_id"))
                chassis_id = submission.get(
                    check.get("chassis_from", "chassis_id")
                )
                compat = self.data.get("compatibility") or {}
                chassis_compat = compat.get(chassis_id) or {}
                allowed = chassis_compat.get("compatible_gpus") or []
                if gpu_id in allowed:
                    return True, None
                return False, (
                    f"gpu '{gpu_id}' not compatible with chassis '{chassis_id}'"
                )
            return False, f"unsupported rule: {rule}"

        return False, f"check has no recognized type: {check}"

    def _validate_consistency_submission(
        self, submission: dict, validator: dict
    ) -> tuple[bool, str | None]:
        ok, reason = self._check_required_fields(submission, validator)
        if not ok:
            return False, reason
        rules = validator.get("consistency_rules") or []
        version = submission.get("component_version")
        region = submission.get("deployment_region")
        for rule in rules:
            if rule.get("if_version") == version:
                required_region = rule.get("required_region")
                if region != required_region:
                    return False, (
                        f"version '{version}' requires region "
                        f"'{required_region}', got '{region}'"
                    )
                return True, None
        return False, f"No consistency rule matched version '{version}'"
