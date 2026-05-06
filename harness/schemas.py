"""JSONSchema definitions for all 8 benchmark tools.

Each schema follows the OpenAI / Anthropic function-calling format:

    {
        "name": "tool_name",
        "description": "Human-readable explanation for the model",
        "parameters": <JSONSchema object>
    }

The `parameters` block is what `jsonschema.validate()` checks against.
The full schema dict can be exported directly to LLM provider tool-use
formats with minimal translation.

Design rules:
- All schemas use `additionalProperties: false` (strict).
- The single exception is `submit_decision`, whose payload shape is
  task-dependent and validated by the fixture's `submission_validator`.
- `compute_value` uses `oneOf` to express per-operation input shapes.
- Examples are included only for the most error-prone parameters.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Helper for compute_value branches
# ----------------------------------------------------------------------


def _arith_branch(op_name: str, description: str) -> dict:
    """Schema branch for binary arithmetic operations (multiply/add/sub/div)."""
    return {
        "type": "object",
        "properties": {
            "operation": {"const": op_name, "description": description},
            "inputs": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
        "required": ["operation", "inputs"],
        "additionalProperties": False,
    }


_WEIGHTED_SCORE_BRANCH = {
    "type": "object",
    "properties": {
        "operation": {
            "const": "weighted_score",
            "description": "Compute Σ(values[k] * weights[k]) for matching keys.",
        },
        "inputs": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "object",
                    "description": (
                        "Dictionary of named numeric values, e.g. "
                        '{"skill": 70, "experience": 98}.'
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "weights": {
                    "type": "object",
                    "description": (
                        "Dictionary of weights matching the keys in `values`, e.g. "
                        '{"skill": 0.6, "experience": 0.4}.'
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["values", "weights"],
            "additionalProperties": False,
        },
    },
    "required": ["operation", "inputs"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------
# Tool schemas
# ----------------------------------------------------------------------


TOOL_SCHEMAS: dict[str, dict] = {
    "search_records": {
        "name": "search_records",
        "description": (
            "Search for records in the current domain by an exact name or known "
            "alias. Returns matching record previews containing only id and name. "
            "Use lookup_record to fetch full details after finding a match. "
            "Returns an empty matches list if no records are found — try a "
            "different exact name or alias before giving up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The exact name or known alias of the record to search "
                        "for. Search uses exact-key lookup against the registered "
                        "index, not substring matching — use the full registered "
                        "name."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },

    "lookup_record": {
        "name": "lookup_record",
        "description": (
            "Retrieve a record by its exact ID. Optionally filter the returned "
            "fields. If the record is not found, returns status=not_found with a "
            "recovery hint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The exact record ID to look up (e.g. 'P104', 'SUP-12').",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of field names to return. If omitted, all "
                        "fields are returned."
                    ),
                },
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
    },

    "get_attribute": {
        "name": "get_attribute",
        "description": (
            "Retrieve a single named attribute from a specific record. Use this "
            "when you only need one field and want to keep responses small. "
            "Returns status=not_found if the record or attribute doesn't exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID to query.",
                },
                "attribute": {
                    "type": "string",
                    "description": "The exact field name to extract from the record.",
                },
            },
            "required": ["record_id", "attribute"],
            "additionalProperties": False,
        },
    },

    "list_options": {
        "name": "list_options",
        "description": (
            "Enumerate the available options in the current domain. Returns "
            "previews containing id and name only — use lookup_record to fetch "
            "details on a specific option."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "option_set": {
                    "type": "string",
                    "description": (
                        "Optional name of a specific option set. If omitted, "
                        "returns the default option set for this task."
                    ),
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },

    "check_constraint": {
        "name": "check_constraint",
        "description": (
            "Validate whether a record satisfies a named constraint defined in "
            "the current task. Returns satisfied=true/false plus the actual and "
            "expected values. Constraints must be referenced by their exact name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record to check the constraint against.",
                },
                "constraint_name": {
                    "type": "string",
                    "description": (
                        "The name of a constraint defined for this task "
                        "(e.g. 'budget_max', 'region_required')."
                    ),
                },
            },
            "required": ["record_id", "constraint_name"],
            "additionalProperties": False,
        },
    },

    "compare_records": {
        "name": "compare_records",
        "description": (
            "Compare multiple records by a single criterion field and return "
            "them ranked. Use this to find the best record by some metric. "
            "Note: comparison is by criterion only — apply constraint filtering "
            "separately with check_constraint if needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "List of record IDs to compare.",
                },
                "criterion": {
                    "type": "string",
                    "description": "The field name to rank by (e.g. 'reliability').",
                },
                "direction": {
                    "type": "string",
                    "enum": ["desc", "asc"],
                    "description": "Sort direction. Default is 'desc' (highest first).",
                },
            },
            "required": ["record_ids", "criterion"],
            "additionalProperties": False,
        },
    },

    "compute_value": {
        "name": "compute_value",
        "description": (
            "Perform a deterministic computation. Supported operations:\n"
            "- multiply / add / subtract / divide: inputs={a, b}\n"
            "- weighted_score: inputs={values: {k: v, ...}, weights: {k: w, ...}}\n"
            "Example: {operation: 'multiply', inputs: {a: 12, b: 8}} → 96\n"
            "Example: {operation: 'weighted_score', inputs: {values: {skill: 70, "
            "experience: 98}, weights: {skill: 0.6, experience: 0.4}}} → 81.2"
        ),
        "parameters": {
            "oneOf": [
                _arith_branch("multiply", "Multiply two numbers (a * b)"),
                _arith_branch("add", "Add two numbers (a + b)"),
                _arith_branch("subtract", "Subtract b from a (a - b)"),
                _arith_branch("divide", "Divide a by b (a / b). Errors on b=0."),
                _WEIGHTED_SCORE_BRANCH,
            ]
        },
    },

    "submit_decision": {
        "name": "submit_decision",
        "description": (
            "Submit your final decision for the task. The required fields depend "
            "on the task — refer to the task prompt for what to include. Common "
            "shapes:\n"
            "- Simple action: {action: 'approve', record_id: 'REQ-220'}\n"
            "- Bundle: {vendor_id: 'V-2', chassis_id: 'CH-4', gpu_id: 'GPU-7'}\n"
            "- Versioned deployment: {component_version: 'v2.1', "
            "deployment_region: 'region-east'}\n"
            "Returns status=ok with submission_id on success, or status=error "
            "with a hint on mismatch (recoverable)."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": True,
        },
    },
}


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def get_schemas_for_task(allowed_tools: list[str]) -> list[dict]:
    """Return the subset of TOOL_SCHEMAS available for a specific task.

    Used by the runner to construct the LLM tool-use payload — only tools
    listed in the task's `tools_available` field are presented to the model.
    """
    return [TOOL_SCHEMAS[t] for t in allowed_tools if t in TOOL_SCHEMAS]


def all_tool_names() -> list[str]:
    """Return the canonical list of supported tool names."""
    return sorted(TOOL_SCHEMAS.keys())
