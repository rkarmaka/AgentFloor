"""Resolve a (task, variant_id) pair to the prompt text the runner should send.

Lookup rules:

  variant_id == "v0" or None
      → return task.example_prompt unchanged. No variants file needed.

  variant_id starts with "v" (e.g. "v1", "v6_explicit")
      → load <task_path>.parent/<task_id>.variants.yaml, find the entry
        whose `id` matches, return its `text` field.

  variant_id doesn't start with "v"
      → ValueError.

The variants file schema (one per task, written by hand):

    task_id: A1
    generator:
      by: <model id>
      generated_at: <ISO timestamp>
      notes: <free text>
    variants:
      - id: v1
        axis: paraphrase
        text: "..."
      - id: v2
        axis: distractor_text
        text: "..."
      - id: v3
        axis: ID_format_noise
        text: "..."
      - id: v4
        axis: reordered_instruction
        text: "..."
      - id: v5
        axis: mild_typo
        text: "..."

The loader does not validate that the variants file's axes match the
task's `variant_axes` declaration — that is the variant author's
responsibility. The loader only enforces shape (id present, text non-empty,
v1..v5 are the only legal ids).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _is_valid_variant_id(vid: str) -> bool:
    return vid == "v0" or (vid.startswith("v") and len(vid) > 1)


class VariantNotFoundError(FileNotFoundError):
    """Raised when a non-v0 variant is requested but no variants file exists.

    Inherits FileNotFoundError so callers that already handle missing
    files don't need a separate except clause.
    """


def variants_path_for_task(task_path: Path) -> Path:
    """Return the canonical .variants.yaml path for a given task YAML.

    Tasks live at tasks/<level>/<task_id>.yaml. Variant files live
    at tasks/<level>/<task_id>.variants.yaml — same directory, same
    stem, .variants.yaml suffix.
    """
    task_path = Path(task_path)
    return task_path.with_name(task_path.stem + ".variants.yaml")


def load_variant_text(
    task_path: Path | str,
    variant_id: str | None,
    *,
    example_prompt: str | None = None,
) -> str:
    """Resolve `variant_id` to a prompt string.

    `task_path` is the path to the task YAML. `example_prompt` is the
    fallback for v0 — the caller has usually already loaded the task and
    can pass it directly. If omitted, we re-read the task YAML to get it.

    Raises VariantNotFoundError if `variant_id` is non-trivial and the
    variants file is missing. Raises KeyError if the file exists but the
    requested id isn't in it. Raises ValueError on unknown variant_id.
    """
    if variant_id is None or variant_id == "v0":
        if example_prompt is not None:
            return example_prompt
        return _load_example_prompt(Path(task_path))

    if not _is_valid_variant_id(variant_id):
        raise ValueError(
            f"unknown variant_id {variant_id!r}; must start with 'v'"
        )

    variants_path = variants_path_for_task(Path(task_path))
    if not variants_path.exists():
        raise VariantNotFoundError(
            f"variants file does not exist: {variants_path}"
        )

    data = _load_variants_file(variants_path)
    entries = data.get("variants") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == variant_id:
            text = entry.get("text")
            if not text or not isinstance(text, str):
                raise KeyError(
                    f"variant {variant_id!r} in {variants_path} has no text"
                )
            return text
    raise KeyError(
        f"variant {variant_id!r} not found in {variants_path}; "
        f"available: {[e.get('id') for e in entries]}"
    )


def list_variant_ids(task_path: Path | str) -> list[str]:
    """Return all variant ids declared for a task, in file order.

    Always includes "v0" first (the original example_prompt), then any
    variants from the .variants.yaml file. Returns just ["v0"] if no
    variants file exists.
    """
    variants_path = variants_path_for_task(Path(task_path))
    out = ["v0"]
    if not variants_path.exists():
        return out
    data = _load_variants_file(variants_path)
    for entry in data.get("variants") or []:
        if isinstance(entry, dict):
            vid = entry.get("id")
            if isinstance(vid, str) and _is_valid_variant_id(vid) and vid != "v0":
                out.append(vid)
    return out


# ----------------------------------------------------------------------
# Instance variants — same task family, different concrete instance
# ----------------------------------------------------------------------
#
# Prompt variants (v1..v5, above) preserve the task instance and only
# change the wording the user sends. Instance variants (i1..i5) preserve
# the task *family* (same template, same fixture, same tool surface) but
# pick a different concrete record / target / parameter set, so the
# expected answer changes too. This is the external-validity ablation
# from the NeurIPS execution guide: it answers the reviewer attack
# "your TCR measures robustness to paraphrase on a fixed instance, not
# generalization across the task family."
#
# Each instance entry overrides:
#   * task.example_prompt (via the `text` field)
#   * any subset of top-level task keys (via the `overrides` block,
#     deep-merged into the task dict — typically gold_state and
#     oracle_evaluator.final_answer_check.expected_fields)


class InstanceVariantNotFoundError(FileNotFoundError):
    """Raised when an i* instance is requested but no instance file exists.

    Inherits FileNotFoundError so callers that handle missing files
    don't need a separate except clause. Mirrors VariantNotFoundError.
    """


def _is_valid_instance_id(vid: str) -> bool:
    return vid == "i0" or (vid.startswith("i") and len(vid) > 1)


def instance_variants_path_for_task(task_path: Path) -> Path:
    """Return the canonical .instance_variants.yaml path for a task YAML.

    Parallels variants_path_for_task — same directory, same stem,
    .instance_variants.yaml suffix.
    """
    task_path = Path(task_path)
    return task_path.with_name(task_path.stem + ".instance_variants.yaml")


def load_instance_overrides(
    task_path: Path | str,
    variant_id: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve an i* `variant_id` to (new_example_prompt, override_dict).

    The override dict contains top-level task keys to deep-merge into the
    loaded task data (typically gold_state, oracle_evaluator). For i0 or
    None, returns ("", {}) — caller treats as no-op.

    Raises InstanceVariantNotFoundError if the instance file is missing.
    Raises KeyError if the file exists but the requested id isn't in it,
    or if the id's text/overrides block is malformed. Raises ValueError
    on a malformed variant_id.
    """
    if variant_id is None or variant_id == "i0":
        return ("", {})

    if not _is_valid_instance_id(variant_id):
        raise ValueError(
            f"unknown instance variant_id {variant_id!r}; must start with 'i'"
        )

    instance_path = instance_variants_path_for_task(Path(task_path))
    if not instance_path.exists():
        raise InstanceVariantNotFoundError(
            f"instance variants file does not exist: {instance_path}"
        )

    data = _load_variants_file(instance_path)
    entries = data.get("instances") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == variant_id:
            text = entry.get("text")
            if not text or not isinstance(text, str):
                raise KeyError(
                    f"instance {variant_id!r} in {instance_path} has no text"
                )
            overrides = entry.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise KeyError(
                    f"instance {variant_id!r} in {instance_path} has non-dict overrides"
                )
            return (text, overrides)
    raise KeyError(
        f"instance {variant_id!r} not found in {instance_path}; "
        f"available: {[e.get('id') for e in entries]}"
    )


def list_instance_variant_ids(task_path: Path | str) -> list[str]:
    """Return all instance variant ids declared for a task, in file order.

    Returns [] if no instance file exists. Unlike list_variant_ids, this
    does not include "i0" — i0 is implicit (run the canonical instance
    via the v0 path).
    """
    instance_path = instance_variants_path_for_task(Path(task_path))
    if not instance_path.exists():
        return []
    data = _load_variants_file(instance_path)
    out: list[str] = []
    for entry in data.get("instances") or []:
        if isinstance(entry, dict):
            vid = entry.get("id")
            if isinstance(vid, str) and _is_valid_instance_id(vid) and vid != "i0":
                out.append(vid)
    return out


def deep_merge_overrides(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base.

    Dict values get merged key-by-key. Non-dict values (including lists)
    in override replace the corresponding base value — list replacement
    rather than concatenation keeps semantics unambiguous when a task
    overrides e.g. trace_requirements.must_call_tools.

    Returns a new dict; does not mutate either argument.
    """
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge_overrides(out[k], v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_example_prompt(task_path: Path) -> str:
    """Pull task.example_prompt from disk."""
    with open(task_path) as f:
        data = yaml.safe_load(f)
    prompt = data.get("example_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise KeyError(f"task {task_path} has no example_prompt")
    return prompt


def _load_variants_file(variants_path: Path) -> dict[str, Any]:
    with open(variants_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"variants file {variants_path} is not a YAML mapping")
    return data
