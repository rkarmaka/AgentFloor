"""Canonical result-subset filters for the AgentFloor paper artifact.

These helpers keep paper-facing figures and aggregate exports aligned with
the exact sweep slices described in the manuscript.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve()
_NEW_ROOT = _HERE.parent.parent

_PASS2_VARIANTS = {"v0", "v1", "v2", "v3", "v4"}
_GPT5_ANCHOR_VARIANTS = {"v0", "v1", "v2"}


def _normalize_model_dir(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


def _parse_result_path(path: str | Path) -> tuple[Path, str, str, int]:
    p = Path(path)
    name = p.name
    if name.endswith(".score.json"):
        stem = name[: -len(".score.json")]
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
    else:
        stem = p.stem
    parts = stem.split("__")
    if len(parts) < 3 or not parts[2].startswith("run"):
        raise ValueError(f"unexpected result filename: {p.name}")
    task_id = parts[0]
    variant = parts[1]
    run_idx = int(parts[2][3:])
    return p, task_id, variant, run_idx


@lru_cache(maxsize=1)
def pass2_model_dirs() -> set[str]:
    cfg = _NEW_ROOT / "sweep_configs" / "ollama_full_pass2.yaml"
    doc = yaml.safe_load(cfg.read_text())
    return {
        _normalize_model_dir(model["name"])
        for model in doc.get("models", [])
        if model.get("name")
    }


def is_paper_baseline_result(path: str | Path) -> bool:
    p, _, variant, run_idx = _parse_result_path(path)
    model_dir = p.parent.name
    provider_dir = p.parent.parent.name

    if provider_dir == "openai_compatible:ollama":
        return model_dir in pass2_model_dirs() and variant in _PASS2_VARIANTS and 0 <= run_idx <= 4

    if provider_dir == "openai":
        if model_dir == "gpt-5":
            return variant in _GPT5_ANCHOR_VARIANTS and 0 <= run_idx <= 2
        if model_dir in {"gpt-5-mini", "gpt-5-nano"}:
            return variant == "v0" and run_idx == 0

    return False


def is_gpt5_extsteps_result(path: str | Path) -> bool:
    p, _, variant, run_idx = _parse_result_path(path)
    return (
        p.parent.parent.name == "openai"
        and p.parent.name == "gpt-5-extsteps"
        and variant in _GPT5_ANCHOR_VARIANTS
        and 0 <= run_idx <= 2
    )


def entry_in_subset(entry: dict, subset: str) -> bool:
    if subset == "all":
        return True
    result_path = entry.get("result_path")
    if not result_path:
        return False
    if subset == "paper_baseline":
        return is_paper_baseline_result(result_path)
    if subset == "gpt5_extsteps":
        return is_gpt5_extsteps_result(result_path)
    raise ValueError(f"unknown subset: {subset}")
