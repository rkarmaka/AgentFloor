"""Tool dispatch and envelope wrapping for the benchmark.

The ToolRouter is the bridge between the model's tool calls and the
FixtureDB. It owns three responsibilities:

1. **Filtering** — only tools in the task's `tools_available` list can
   be called. Calls to other tools are rejected as F1 (schema
   hallucination).

2. **Validation** — every call's arguments are validated against the
   tool's JSONSchema. Failures are rejected as F2 (malformed call).

3. **Logging** — every call (valid or not) is appended to a structured
   trace that the runner / evaluator can read after the run.

Successful and failed calls are both wrapped in the standard envelope:

    {
        "schema_version": "0.2",
        "tool_name":      "lookup_record",
        "call_id":        "B3_lookup_record_1",
        "status":         "ok" | "not_found" | "error",
        "result":         <dict | None>,
        "error":          <dict | None>
    }
"""

from __future__ import annotations

from typing import Any

import jsonschema

from .db import FixtureDB
from .schemas import TOOL_SCHEMAS, get_schemas_for_task
from .tool_registry import DISPATCH_TABLE


SCHEMA_VERSION = "0.2"


class ToolRouter:
    """Routes model tool calls to the FixtureDB and logs every call.

    One instance per task run. Holds the call log for the entire run so
    the runner can extract it for metrics computation.
    """

    def __init__(self, db: FixtureDB, allowed_tools: list[str]):
        self.db = db
        self.allowed_tools: set[str] = set(allowed_tools)
        self.call_log: list[dict] = []
        self._counter = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[dict]:
        """Return JSONSchema definitions for tools available in this task.

        Used by the runner to construct the LLM tool-use payload.
        """
        return get_schemas_for_task(sorted(self.allowed_tools))

    def call(
        self,
        tool_name: str,
        args: dict | None = None,
        *,
        malformed_override: str | None = None,
    ) -> dict:
        """Main entry point. Always returns a fully-formed envelope.

        The envelope is identical in shape whether the call succeeds,
        is rejected as hallucinated, or fails schema validation.
        """
        args = args or {}
        self._counter += 1
        call_id = self._make_call_id(tool_name)

        # Step 1: Reject if the tool isn't in the allowed list (F1)
        if tool_name not in self.allowed_tools:
            envelope = self._envelope(
                tool_name,
                call_id,
                status="error",
                result=None,
                error={
                    "type": "tool_not_available",
                    "message": f"Tool '{tool_name}' is not available for this task",
                    "recoverable": False,
                    "hint": f"Available tools: {sorted(self.allowed_tools)}",
                },
            )
            self._log(call_id, tool_name, args, envelope, hallucinated=True)
            return envelope

        # Step 2: Preserve adapter-reported parse failures as F2 even for
        # polymorphic tools like submit_decision, whose schema may otherwise
        # accept arbitrary object keys.
        if malformed_override is not None:
            envelope = self._envelope(
                tool_name,
                call_id,
                status="error",
                result=None,
                error={
                    "type": "schema_validation_error",
                    "message": malformed_override,
                    "recoverable": True,
                },
            )
            self._log(call_id, tool_name, args, envelope, malformed=True)
            return envelope

        # Step 2: Validate arguments against the tool's schema (F2)
        valid, validation_error = self._validate(tool_name, args)
        if not valid:
            envelope = self._envelope(
                tool_name,
                call_id,
                status="error",
                result=None,
                error={
                    "type": "schema_validation_error",
                    "message": validation_error,
                    "recoverable": True,
                },
            )
            self._log(call_id, tool_name, args, envelope, malformed=True)
            return envelope

        # Step 3: Dispatch to the DB
        try:
            raw = DISPATCH_TABLE[tool_name](self.db, args)
        except Exception as e:
            envelope = self._envelope(
                tool_name,
                call_id,
                status="error",
                result=None,
                error={
                    "type": "dispatch_error",
                    "message": f"{type(e).__name__}: {e}",
                    "recoverable": False,
                },
            )
            self._log(call_id, tool_name, args, envelope, dispatch_error=True)
            return envelope

        # Step 4: Wrap and log
        try:
            envelope = self._envelope(
                tool_name,
                call_id,
                status=raw["status"],
                result=raw["result"],
                error=raw["error"],
            )
        except KeyError as e:
            envelope = self._envelope(
                tool_name,
                call_id,
                status="error",
                result=None,
                error={
                    "type": "dispatch_error",
                    "message": (
                        "Dispatcher returned malformed response missing key: "
                        f"{e.args[0]}"
                    ),
                    "recoverable": False,
                },
            )
            self._log(call_id, tool_name, args, envelope, dispatch_error=True)
            return envelope
        self._log(call_id, tool_name, args, envelope)
        return envelope

    # ------------------------------------------------------------------
    # Metrics-friendly accessors
    # ------------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        return len(self.call_log)

    @property
    def hallucinated_calls(self) -> int:
        return sum(1 for c in self.call_log if c["is_hallucinated"])

    @property
    def malformed_calls(self) -> int:
        return sum(1 for c in self.call_log if c["is_malformed"])

    @property
    def successful_calls(self) -> int:
        return sum(1 for c in self.call_log if c["response_status"] == "ok")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_call_id(self, tool_name: str) -> str:
        prefix = self.db.task_id or "NOTASK"
        return f"{prefix}_{tool_name}_{self._counter}"

    def _validate(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
        schema = TOOL_SCHEMAS[tool_name]["parameters"]
        try:
            jsonschema.validate(instance=args, schema=schema)
            return True, None
        except jsonschema.ValidationError as e:
            # e.message is the leaf error; e.json_path locates it within args
            location = e.json_path if hasattr(e, "json_path") else "$"
            return False, f"{location}: {e.message}"

    def _envelope(
        self,
        tool_name: str,
        call_id: str,
        *,
        status: str,
        result: Any,
        error: Any,
    ) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_name": tool_name,
            "call_id": call_id,
            "status": status,
            "result": result,
            "error": error,
        }

    def _log(
        self,
        call_id: str,
        tool_name: str,
        args: dict,
        envelope: dict,
        *,
        hallucinated: bool = False,
        malformed: bool = False,
        dispatch_error: bool = False,
    ) -> None:
        self.call_log.append(
            {
                "call_id": call_id,
                "call_index": self._counter,
                "tool_name": tool_name,
                "args": args,
                "is_hallucinated": hallucinated,
                "is_malformed": malformed,
                "is_dispatch_error": dispatch_error,
                "valid_schema": not malformed,
                "response_status": envelope["status"],
                "response": envelope,
            }
        )
