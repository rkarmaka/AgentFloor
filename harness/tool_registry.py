"""Dispatch table mapping tool names to FixtureDB query methods.

The tool layer (tools.py) calls into this table after schema validation.
Each dispatcher unpacks the validated args and calls the appropriate
DB method, returning the raw envelope from the DB.

Keeping dispatch separate from validation lets us:
- Test the dispatch layer without spinning up jsonschema
- Add per-tool argument transformation in one place
- Swap the underlying DB implementation without touching tools.py
"""

from __future__ import annotations

from typing import Callable

from .db import FixtureDB


# ----------------------------------------------------------------------
# Per-tool dispatchers
# ----------------------------------------------------------------------


def _dispatch_search_records(db: FixtureDB, args: dict) -> dict:
    return db.search_records(args["query"])


def _dispatch_lookup_record(db: FixtureDB, args: dict) -> dict:
    return db.lookup_record(args["record_id"], args.get("fields"))


def _dispatch_get_attribute(db: FixtureDB, args: dict) -> dict:
    return db.get_attribute(args["record_id"], args["attribute"])


def _dispatch_list_options(db: FixtureDB, args: dict) -> dict:
    return db.list_options(args.get("option_set"))


def _dispatch_check_constraint(db: FixtureDB, args: dict) -> dict:
    return db.check_constraint(args["record_id"], args["constraint_name"])


def _dispatch_compare_records(db: FixtureDB, args: dict) -> dict:
    return db.compare_records(
        args["record_ids"],
        args["criterion"],
        args.get("direction", "desc"),
    )


def _dispatch_compute_value(db: FixtureDB, args: dict) -> dict:
    return db.compute_value(args["operation"], args["inputs"])


def _dispatch_submit_decision(db: FixtureDB, args: dict) -> dict:
    # submit_decision is polymorphic — pass the entire args dict as kwargs
    return db.submit_decision(**args)


# ----------------------------------------------------------------------
# Public dispatch table
# ----------------------------------------------------------------------


DISPATCH_TABLE: dict[str, Callable[[FixtureDB, dict], dict]] = {
    "search_records":   _dispatch_search_records,
    "lookup_record":    _dispatch_lookup_record,
    "get_attribute":    _dispatch_get_attribute,
    "list_options":     _dispatch_list_options,
    "check_constraint": _dispatch_check_constraint,
    "compare_records":  _dispatch_compare_records,
    "compute_value":    _dispatch_compute_value,
    "submit_decision":  _dispatch_submit_decision,
}
