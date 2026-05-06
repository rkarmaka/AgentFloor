"""Evaluator check primitives.

Each module exposes one or more pure functions that take a result dict
plus a per-task spec and return a CheckResult. The orchestrator in
`evaluator.py` wires them together.

The four checker families:
  final_answer  — does the model's free text contain the expected answer?
  submission    — does the submit_decision payload match the gold state?
  trajectory    — did the call_log follow the required tool sequence + flags?
  forbidden     — did the run avoid the declared forbidden behaviors?
"""

from __future__ import annotations

# Re-export the dataclass at the package level so check modules can
# import it without circular-importing the evaluator module.
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Outcome of a single evaluator check.

    `passed` is the headline boolean. `reason` is a one-line human
    explanation. `details` is a structured dict for the metrics layer
    to read later — e.g. {"matched_field": "status", "weak_match": True}.

    Helpers like `ok()` and `fail()` keep call sites short.
    """

    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, reason: str = "ok", **details: Any) -> "CheckResult":
        return cls(passed=True, reason=reason, details=dict(details))

    @classmethod
    def fail(cls, reason: str, **details: Any) -> "CheckResult":
        return cls(passed=False, reason=reason, details=dict(details))


__all__ = ["CheckResult"]
