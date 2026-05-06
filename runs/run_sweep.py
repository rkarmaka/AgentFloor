#!/usr/bin/env python3
"""Batch sweep runner for the AgentFloor benchmark.

Drives the full model x task x variant x run matrix from a YAML config.
Unlike the per-backend run_*.py scripts (run_vllm, run_gemini, run_nim),
this orchestrates many models sequentially with resume support, model
lifecycle management, and optional auto-evaluation.

Usage:

    python runs/run_sweep.py --config sweep_configs/smoke.yaml
    python runs/run_sweep.py --config sweep_configs/ollama_full.yaml --dry-run
    python runs/run_sweep.py --config sweep_configs/smoke.yaml --models qwen2.5:7b
    python runs/run_sweep.py --config sweep_configs/smoke.yaml --tasks A1,A2,B1
    python runs/run_sweep.py --config sweep_configs/smoke.yaml --eval

Sweep config format (see sweep_configs/smoke.yaml for an example):

    sweep_name: ollama_baseline
    backend: ollama                         # ollama | vllm | nim | openai | gemini | anthropic
    base_url: http://localhost:11434/v1     # for openai-compatible backends
    results_dir: results                    # default
    max_output_tokens: 1024                 # default

    models:
      - name: qwen2.5:7b
        params_b: 7.0
        tier: 7b-9b

    tasks: all                              # or [A1,B1,C1] or tier:A,B
    variants: [v0]                          # v0 only for baseline phase
    runs_per_combo: 1                       # 5 for final sweep

Resume behavior: if the result JSON already exists at
{results_dir}/{provider}/{model_slug}/{task_id}__{variant}__run{N}.json,
the combo is skipped. Re-run the same command to resume after an interruption.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_dotenv_files() -> None:
    """Load API keys from .env.local files without overriding already-set env vars.

    Checks (in order) runs/.env.local then repo-root .env.local. No external
    dependency — simple KEY=VALUE parser, ignores blank lines and '#' comments.
    """
    import os
    for candidate in (_HERE / ".env.local", _REPO_ROOT / ".env.local"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


_load_dotenv_files()

from harness.providers.base import ModelConfig, user_message  # noqa: E402
from harness.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from harness.runner import run_task, write_result, _slug  # noqa: E402
from harness.schemas import TOOL_SCHEMAS  # noqa: E402


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str                      # e.g. "qwen2.5:7b" for Ollama
    params_b: float | None = None
    tier: str | None = None
    backend_override: str | None = None      # overrides sweep-level backend
    base_url_override: str | None = None
    api_key_env: str | None = None           # e.g. "NVIDIA_API_KEY"
    max_output_tokens: int | None = None     # overrides sweep-level
    max_steps_override: int | None = None    # overrides task.max_steps;
                                             # use for ablations like the
                                             # gpt-5 D/E F4 step-budget probe
    notes: str | None = None
    system_prompt_suffix: str | None = None  # e.g. "/no_think" for Qwen3
                                             # reasoning-mode ablations; gets
                                             # appended to the default
                                             # SYSTEM_PROMPT for this model only
    extra: dict | None = None                # passed to ModelConfig.extra →
                                             # merged into API payload (e.g.
                                             # {"reasoning_effort": "low"})
    system_prompt_override: str | None = None  # fully replaces SYSTEM_PROMPT
                                               # (unlike suffix which appends)
    api_model: str | None = None             # actual model name sent to API
                                             # when different from `name` (the
                                             # result-directory slug). Use for
                                             # ablations where one API model has
                                             # multiple sweep entries.


@dataclass
class SweepConfig:
    sweep_name: str
    backend: str                   # ollama | vllm | nim | gemini | openai | anthropic
    base_url: str | None = None
    results_dir: Path = Path("results")
    max_output_tokens: int = 1024
    models: list[ModelSpec] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)    # resolved task IDs
    variants: list[str] = field(default_factory=lambda: ["v0"])
    runs_per_combo: int = 1
    max_steps_override: int | None = None   # sweep-level fallback override of
                                            # task.max_steps; per-model
                                            # ModelSpec.max_steps_override wins
                                            # if both are set
    auto_pull: bool = True                 # Ollama: pull before running
    auto_stop: bool = True                 # Ollama: stop after each model
    probe_before_run: bool = True          # send one-call probe first
    compat_out: Path | None = None         # where to write compatibility YAML

    @property
    def provider_name(self) -> str:
        """DEPRECATED: sweep-wide provider_name. Use provider_name_for(model)
        instead — in mixed-backend sweeps (e.g. api_frontier.yaml), each
        model's backend may differ from the sweep default.

        Kept for backward compatibility with code paths that assume a
        single backend (e.g. the compat_out probe YAML path).
        """
        return self.provider_name_for_backend(self.backend)

    @staticmethod
    def provider_name_for_backend(backend: str) -> str:
        """The result-path provider identifier for a given backend.

        For openai-compatible backends (ollama/vllm/together/nim), the
        provider stamps "openai_compatible:{backend}" on RunResult and
        runner.write_result() writes the result under that raw name.
        For native API providers the name is just the backend.
        """
        if backend in ("ollama", "vllm", "together", "nim"):
            return f"openai_compatible:{backend}"
        return backend

    def provider_name_for(self, model: "ModelSpec") -> str:
        """Provider name a result from this specific model will be saved under."""
        return self.provider_name_for_backend(model.backend_override or self.backend)


def load_sweep_config(path: Path) -> SweepConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    raw_models = data.get("models") or []
    models = [
        ModelSpec(
            name=m["name"],
            params_b=m.get("params_b"),
            tier=m.get("tier"),
            backend_override=m.get("backend"),
            base_url_override=m.get("base_url"),
            api_key_env=m.get("api_key_env"),
            max_output_tokens=m.get("max_output_tokens"),
            max_steps_override=m.get("max_steps_override"),
            notes=m.get("notes"),
            system_prompt_suffix=m.get("system_prompt_suffix"),
            extra=m.get("extra"),
            system_prompt_override=m.get("system_prompt_override"),
            api_model=m.get("api_model"),
        )
        for m in raw_models
    ]

    tasks = _resolve_task_list(data.get("tasks", "all"))
    variants = data.get("variants") or ["v0"]
    results_dir = Path(data.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = _REPO_ROOT / results_dir

    compat_out = data.get("compat_out")
    if compat_out:
        compat_out = Path(compat_out)
        if not compat_out.is_absolute():
            compat_out = _REPO_ROOT / compat_out

    backend = data.get("backend")
    if not backend:
        model_backends = [m.backend_override for m in models if m.backend_override]
        if not model_backends:
            raise ValueError(
                "Sweep config must define 'backend' at top-level, or provide "
                "per-model backend values under models[].backend."
            )
        # Mixed-backend sweeps still need a sweep-level default for legacy paths
        # (e.g. probe metadata output). Use the first model backend as a safe
        # fallback; model-level dispatch still uses each model override.
        backend = model_backends[0]

    return SweepConfig(
        sweep_name=data.get("sweep_name", path.stem),
        backend=backend,
        base_url=data.get("base_url"),
        results_dir=results_dir,
        max_output_tokens=int(data.get("max_output_tokens", 1024)),
        models=models,
        tasks=tasks,
        variants=variants,
        runs_per_combo=int(data.get("runs_per_combo", 1)),
        max_steps_override=data.get("max_steps_override"),
        auto_pull=bool(data.get("auto_pull", True)),
        auto_stop=bool(data.get("auto_stop", True)),
        probe_before_run=bool(data.get("probe_before_run", True)),
        compat_out=compat_out,
    )


_ALL_TASKS = [
    "A01", "A02", "A03", "A04", "A05",
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5",
]


def _resolve_task_list(spec: Any) -> list[str]:
    """Accept: 'all', list of IDs, or 'tier:A,B' shorthand."""
    if spec == "all" or spec is None:
        return list(_ALL_TASKS)
    if isinstance(spec, list):
        return [str(t) for t in spec]
    if isinstance(spec, str):
        if spec.startswith("tier:"):
            tiers = [t.strip() for t in spec[len("tier:"):].split(",")]
            out = []
            for t in _ALL_TASKS:
                if t.startswith("A0"):
                    tier = "A0"
                else:
                    tier = t[0]
                if tier in tiers:
                    out.append(t)
            return out
        # CSV fallback
        return [s.strip() for s in spec.split(",") if s.strip()]
    raise ValueError(f"cannot resolve tasks spec: {spec!r}")


def resolve_task_path(task_id: str) -> Path:
    """Map a bare task_id to its YAML file path under tasks/."""
    tasks_root = _REPO_ROOT / "tasks"
    if task_id.startswith("A0"):
        level = "A0"
    elif task_id and task_id[0] in "ABCDE":
        level = task_id[0]
    else:
        raise SystemExit(f"cannot infer tier for task_id {task_id!r}")
    cand = tasks_root / level / f"{task_id}.yaml"
    if cand.exists():
        return cand
    raise SystemExit(f"task file not found: {cand}")


# ----------------------------------------------------------------------
# Ollama lifecycle
# ----------------------------------------------------------------------


def ollama_list_local() -> set[str]:
    """Return the set of model names currently pulled locally."""
    try:
        res = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    names = set()
    for line in res.stdout.splitlines()[1:]:       # skip header
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def ollama_pull(model: str, *, timeout: float = 3600.0) -> bool:
    """Pull an Ollama model. Returns True on success.

    Default timeout is 1 hour — at a conservative ~3–4 MB/s sustained pull
    speed this accommodates models up to ~14 GB. Larger models (20–25 GB
    range) have been observed to exceed the old 600 s default even for
    partial pulls; 3600 s matches the worst observed cold-pull case for
    the current canonical sweep list (gpt-oss:20b / gemma4:26b /
    mistral-small3.2:24b).
    """
    print(f"  [ollama] pulling {model}...", flush=True)
    try:
        subprocess.run(
            ["ollama", "pull", model],
            timeout=timeout, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [ollama] pull failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def ollama_stop(model: str) -> None:
    """Best-effort: ask Ollama to unload a model to free GPU memory."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            capture_output=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def ollama_server_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Quick probe: is the Ollama (or any OpenAI-compatible) server up?"""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


# ----------------------------------------------------------------------
# Tool-support probe (reused from run_vllm.py)
# ----------------------------------------------------------------------


_PROBE_TOOL_NAME = "lookup_record"
_PROBE_PROMPT = (
    "Look up the product record with ID 'P104' using the lookup_record tool. "
    "Call the tool — do not answer from memory."
)
_JSON_SHAPE_RE = re.compile(r"[\{\[].*[\}\]]", re.DOTALL)
_PYCALL_SHAPE_RE = re.compile(rf"\b{re.escape(_PROBE_TOOL_NAME)}\s*\(")


def _classify_probe(turn, tool_name: str) -> tuple[str, str]:
    if turn.tool_calls:
        return "native", f"emitted {len(turn.tool_calls)} tool_call(s) natively"
    text = (turn.text or "").strip()
    if not text:
        return "no_tools", "empty response"
    has_name = tool_name in text
    has_json = bool(_JSON_SHAPE_RE.search(text))
    has_pycall = bool(_PYCALL_SHAPE_RE.search(text))
    if has_name and (has_json or has_pycall):
        shape = "JSON" if has_json else "Python-call"
        return "text_leaked", f"tool_calls empty + text contains tool name + {shape} shape"
    if has_name:
        return "text_leaked", "text mentions tool name but tool_calls empty"
    return "no_tools", "normal prose, no tool use"


@dataclass
class ProbeResult:
    model: str
    status: str                 # native | text_leaked | no_tools | error
    tool_calls: int
    stop_reason: str | None
    text_excerpt: str | None
    notes: str
    latency_ms: int


def probe_tool_support(provider, model_name: str, max_output_tokens: int = 256) -> ProbeResult:
    schema = TOOL_SCHEMAS[_PROBE_TOOL_NAME]
    tools_payload = provider.format_tools([schema])
    config = ModelConfig(model=model_name, max_output_tokens=max_output_tokens, temperature=0.0)
    messages = [user_message(_PROBE_PROMPT)]
    try:
        turn = provider.complete(
            system=(
                "You are an agent that completes tasks by calling tools. "
                "Call the provided tool using the native tool-calling interface."
            ),
            messages=messages,
            tools=tools_payload,
            config=config,
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            model=model_name, status="error", tool_calls=0, stop_reason=None,
            text_excerpt=None, notes=f"{type(e).__name__}: {str(e)[:200]}", latency_ms=0,
        )
    status, notes = _classify_probe(turn, _PROBE_TOOL_NAME)
    excerpt = (turn.text or "").strip().replace("\n", " ")[:140] or None
    return ProbeResult(
        model=model_name, status=status, tool_calls=len(turn.tool_calls),
        stop_reason=turn.stop_reason, text_excerpt=excerpt, notes=notes,
        latency_ms=turn.latency_ms,
    )


def append_probe_yaml(result: ProbeResult, backend: str, path: Path) -> None:
    """Append probe outcome to a running compatibility matrix YAML."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {backend} tool-calling compatibility matrix\n"
            f"# First entry written by runs/run_sweep.py at {timestamp}\n"
            f"probe_tool: {_PROBE_TOOL_NAME}\n"
            f"probe_prompt: {_PROBE_PROMPT!r}\n"
            f"backend: {backend}\n"
            "results:\n"
        )
    with open(path, "a") as f:
        f.write(f"  - model: {result.model}\n")
        f.write(f"    probed_at: {timestamp}\n")
        f.write(f"    status: {result.status}\n")
        f.write(f"    tool_calls: {result.tool_calls}\n")
        f.write(f"    stop_reason: {result.stop_reason!r}\n")
        f.write(f"    latency_ms: {result.latency_ms}\n")
        f.write(f"    notes: {result.notes!r}\n")
        if result.text_excerpt:
            f.write(f"    text_excerpt: {result.text_excerpt!r}\n")


# ----------------------------------------------------------------------
# Provider construction
# ----------------------------------------------------------------------


def make_provider(config: SweepConfig, model: ModelSpec):
    """Construct a Provider instance for the given sweep+model.

    Returns the provider; does NOT load the model. Caller is responsible
    for any model lifecycle (ollama pull, etc.).
    """
    backend = model.backend_override or config.backend
    base_url = model.base_url_override or config.base_url

    if backend == "ollama":
        return OpenAICompatibleProvider(
            backend="ollama",
            base_url=base_url or "http://localhost:11434/v1",
            api_key="ollama",
        )
    if backend == "vllm":
        return OpenAICompatibleProvider(
            backend="vllm",
            base_url=base_url or "http://localhost:8000/v1",
            api_key="dummy",
        )
    if backend == "nim":
        import os
        api_key = os.environ.get(model.api_key_env or "NVIDIA_API_KEY")
        if not api_key:
            raise SystemExit(f"NVIDIA_API_KEY env var missing for NIM backend")
        return OpenAICompatibleProvider(
            backend="nim",
            base_url=base_url or "https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
    if backend == "together":
        import os
        api_key = os.environ.get(model.api_key_env or "TOGETHER_API_KEY")
        if not api_key:
            raise SystemExit("TOGETHER_API_KEY env var missing for Together backend")
        return OpenAICompatibleProvider(
            backend="together",
            base_url=base_url or "https://api.together.xyz/v1",
            api_key=api_key,
        )
    if backend == "gemini":
        from harness.providers.gemini import GeminiProvider
        return GeminiProvider()
    if backend == "openai":
        from harness.providers.openai import OpenAIProvider
        return OpenAIProvider()
    if backend == "anthropic":
        from harness.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    raise SystemExit(f"unknown backend: {backend}")


# ----------------------------------------------------------------------
# Sweep execution
# ----------------------------------------------------------------------


@dataclass
class RunRecord:
    model: str
    task_id: str
    variant: str
    run_idx: int
    status: str              # "ran" | "skipped" | "error" | "skipped_no_tools"
    tcr_pass: bool | None = None
    termination: str | None = None
    error: str | None = None
    wall_ms: int = 0


def result_path_for(
    results_dir: Path,
    provider_name: str,
    model: str,
    task_id: str,
    variant: str,
    run_idx: int,
) -> Path:
    """Compute the canonical result JSON path for a (model, task, variant, run) combo.

    Must match runner.write_result()'s convention: the provider name is
    used *raw* (with its colon if any), only the model name is slugged.
    """
    model_slug = _slug(model)
    fname = f"{task_id}__{variant}__run{run_idx}.json"
    return results_dir / provider_name / model_slug / fname


def run_one_model(
    config: SweepConfig,
    model: ModelSpec,
    *,
    dry_run: bool = False,
    probe_only: bool = False,
) -> list[RunRecord]:
    """Execute all (task, variant, run_idx) combos for one model. Sequential.

    If probe_only=True, runs the tool-support probe and stops (no tasks).
    """
    records: list[RunRecord] = []
    backend = model.backend_override or config.backend

    # Pre-flight: ensure model is loaded (Ollama)
    real_model_name = model.api_model or model.name
    if backend == "ollama" and not dry_run:
        if config.auto_pull:
            local = ollama_list_local()
            if real_model_name not in local:
                if not ollama_pull(real_model_name):
                    print(f"  [skip] pull failed for {real_model_name}", file=sys.stderr)
                    return records
            else:
                print(f"  [ollama] {real_model_name} already local", flush=True)
        base_url = model.base_url_override or config.base_url or "http://localhost:11434/v1"
        if not ollama_server_reachable(base_url):
            print(f"  [error] Ollama server unreachable at {base_url}", file=sys.stderr)
            return records

    provider = make_provider(config, model)
    max_out = model.max_output_tokens or config.max_output_tokens
    api_model = model.api_model or model.name
    model_config = ModelConfig(model=api_model, max_output_tokens=max_out, temperature=0.0,
                               extra=model.extra or {})

    # In probe-only mode, send probe and return — don't run tasks
    if probe_only:
        if dry_run:
            print(f"  [probe] would probe {model.name}", flush=True)
            return records
        probe = probe_tool_support(provider, model.name)
        _print_probe(model.name, probe)
        if config.compat_out:
            append_probe_yaml(probe, backend, config.compat_out)
        if backend == "ollama" and config.auto_stop:
            ollama_stop(model.name)
        return records

    # Probe tool support before the full sweep
    probe_failed = False
    if config.probe_before_run and not dry_run:
        print(f"  [probe] testing tool support on {model.name}...", flush=True)
        probe = probe_tool_support(provider, model.name)
        _print_probe(model.name, probe)
        if config.compat_out:
            append_probe_yaml(probe, backend, config.compat_out)
        if probe.status in ("no_tools", "text_leaked", "error"):
            print(f"  [skip] {model.name} failed tool probe (status={probe.status}); "
                  "skipping full sweep for this model", file=sys.stderr)
            probe_failed = True

    if probe_failed:
        for task_id in config.tasks:
            for variant in config.variants:
                for run_idx in range(config.runs_per_combo):
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="skipped_no_tools",
                    ))
        if backend == "ollama" and config.auto_stop and not dry_run:
            ollama_stop(model.name)
        return records

    # The actual task x variant x run loop
    total = len(config.tasks) * len(config.variants) * config.runs_per_combo
    done = 0
    for task_id in config.tasks:
        try:
            task_path = resolve_task_path(task_id)
        except SystemExit as e:
            print(f"  [error] {e}", file=sys.stderr)
            for variant in config.variants:
                for run_idx in range(config.runs_per_combo):
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="error", error=str(e),
                    ))
                    done += 1
            continue

        for variant in config.variants:
            for run_idx in range(config.runs_per_combo):
                done += 1
                out_path = result_path_for(
                    config.results_dir, config.provider_name_for(model), model.name,
                    task_id, variant, run_idx,
                )
                if out_path.exists():
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="skipped",
                    ))
                    continue

                if dry_run:
                    print(f"  [{done}/{total}] DRY {task_id} {variant} run{run_idx}", flush=True)
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="skipped",
                    ))
                    continue

                wall_start = time.perf_counter()
                try:
                    run_kwargs: dict = {"variant_id": variant}
                    if model.system_prompt_override:
                        run_kwargs["system_prompt"] = model.system_prompt_override
                    elif model.system_prompt_suffix:
                        from harness.runner import SYSTEM_PROMPT as _DEFAULT_SYS
                        run_kwargs["system_prompt"] = (
                            _DEFAULT_SYS + "\n\n" + model.system_prompt_suffix
                        )
                    # max_steps override: per-model wins over sweep-level.
                    # Used for budget ablations (e.g. raise the cap to test
                    # whether step_budget_exhausted reflects clipping rather
                    # than a capability ceiling).
                    eff_max_steps = (
                        model.max_steps_override
                        if model.max_steps_override is not None
                        else config.max_steps_override
                    )
                    if eff_max_steps is not None:
                        run_kwargs["max_steps_override"] = eff_max_steps
                    result = run_task(
                        task_path, provider, model_config,
                        **run_kwargs,
                    )
                    # Use the sweep slug (model.name) for the result path,
                    # not the API model name — prevents ablation runs from
                    # overwriting base model results.
                    result.model = model.name
                    write_result(result, config.results_dir, variant=variant, run_idx=run_idx)
                    wall_ms = int((time.perf_counter() - wall_start) * 1000)
                    print(f"  [{done}/{total}] {task_id} {variant} run{run_idx}: "
                          f"term={result.termination} turns={result.total_turns} "
                          f"calls={len(result.call_log)} {wall_ms}ms", flush=True)
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="ran",
                        termination=result.termination, wall_ms=wall_ms,
                    ))
                except Exception as e:  # noqa: BLE001
                    wall_ms = int((time.perf_counter() - wall_start) * 1000)
                    print(f"  [{done}/{total}] {task_id} {variant} run{run_idx}: "
                          f"ERROR {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
                    records.append(RunRecord(
                        model=model.name, task_id=task_id, variant=variant,
                        run_idx=run_idx, status="error", error=f"{type(e).__name__}: {e}",
                        wall_ms=wall_ms,
                    ))

    if backend == "ollama" and config.auto_stop and not dry_run:
        print(f"  [ollama] stopping {model.name} to free memory", flush=True)
        ollama_stop(model.name)

    return records


def run_sweep(
    config: SweepConfig,
    *,
    dry_run: bool = False,
    probe_only: bool = False,
) -> list[RunRecord]:
    all_records: list[RunRecord] = []
    print()
    print(f"=== Sweep: {config.sweep_name} ===")
    print(f"backend       : {config.backend}")
    print(f"base_url      : {config.base_url}")
    print(f"models        : {len(config.models)}")
    if probe_only:
        print("MODE: PROBE ONLY (no tasks will run)")
    else:
        print(f"tasks         : {len(config.tasks)}")
        print(f"variants      : {config.variants}")
        print(f"runs_per_combo: {config.runs_per_combo}")
        total_combos = len(config.models) * len(config.tasks) * len(config.variants) * config.runs_per_combo
        print(f"total combos  : {total_combos}")
    print(f"results_dir   : {config.results_dir}")
    if config.compat_out:
        print(f"compat_out    : {config.compat_out}")
    if dry_run:
        print("MODE: DRY RUN (no actual execution)")
    print()

    for i, model in enumerate(config.models, 1):
        print(f"=== [{i}/{len(config.models)}] Model: {model.name} "
              f"(tier={model.tier}, params_b={model.params_b}) ===")
        recs = run_one_model(config, model, dry_run=dry_run, probe_only=probe_only)
        all_records.extend(recs)
        if not probe_only:
            _print_model_summary(model.name, recs)

    return all_records


def _print_probe(model_name: str, probe) -> None:
    """Print a one-liner probe summary with status annotation."""
    status_markers = {
        "native": "PASS",
        "text_leaked": "BROKEN (parser can't extract tool calls)",
        "no_tools": "BROKEN (model ignored tool schema)",
        "error": "ERROR",
    }
    marker = status_markers.get(probe.status, probe.status)
    print(f"  [probe] status={probe.status} [{marker}] "
          f"tool_calls={probe.tool_calls} latency_ms={probe.latency_ms}", flush=True)
    if probe.notes:
        print(f"  [probe] notes: {probe.notes}", flush=True)
    if probe.text_excerpt and probe.status != "native":
        print(f"  [probe] text  : {probe.text_excerpt}", flush=True)


def _print_model_summary(model: str, records: list[RunRecord]) -> None:
    counts = {"ran": 0, "skipped": 0, "error": 0, "skipped_no_tools": 0}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    total_wall = sum(r.wall_ms for r in records if r.status == "ran")
    print(f"  summary: ran={counts['ran']} skipped={counts['skipped']} "
          f"errors={counts['error']} no_tools={counts['skipped_no_tools']} "
          f"total_wall={total_wall/1000:.1f}s")
    print()


def _print_final_summary(records: list[RunRecord]) -> None:
    total = len(records)
    counts = {"ran": 0, "skipped": 0, "error": 0, "skipped_no_tools": 0}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    print()
    print("=" * 60)
    print(f"SWEEP COMPLETE: {total} combos")
    print(f"  ran              : {counts['ran']}")
    print(f"  skipped (resume) : {counts['skipped']}")
    print(f"  skipped (no tools): {counts['skipped_no_tools']}")
    print(f"  errors           : {counts['error']}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Auto-evaluate
# ----------------------------------------------------------------------


def run_evaluation(config: SweepConfig) -> None:
    """Score all results and print TCR table. Uses the existing evaluator.

    For mixed-backend sweeps (per-model backend_override), each model's
    results live under its own provider directory. We collect scores
    across every provider dir referenced by the sweep.
    """
    from harness.evaluator import (  # noqa: E402
        evaluate_directory,
        print_tcr_table,
        print_failure_breakdown,
        write_score_alongside,
    )
    print()
    print("=" * 60)
    print("EVALUATING RESULTS")
    print("=" * 60)
    tasks_dir = _HERE / "tasks"
    # Collect the unique provider dirs this sweep writes to. In a
    # single-backend sweep this is typically one; in mixed-backend
    # sweeps (api_frontier.yaml) it may be several.
    provider_names = {config.provider_name_for(m) for m in config.models}
    if not provider_names:
        provider_names = {config.provider_name}
    scopes = []
    for pname in sorted(provider_names):
        scope = config.results_dir / pname
        if scope.exists():
            scopes.append(scope)
        else:
            print(f"no results yet under {scope}")
    if not scopes:
        return
    scores = []
    for scope in scopes:
        scores.extend(evaluate_directory(scope, tasks_dir))
    if not scores:
        print("no scorable results found")
        return
    for s in scores:
        write_score_alongside(s)
    print_tcr_table(scores)
    print_failure_breakdown(scores)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to sweep YAML")
    ap.add_argument("--models", help="CSV filter: only run models with these names")
    ap.add_argument("--tasks", help="CSV filter: only run these task IDs")
    ap.add_argument("--variants", help="CSV filter: only run these variants")
    ap.add_argument("--runs", type=int, help="Override runs_per_combo")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    ap.add_argument("--eval", action="store_true", help="Run evaluator after sweep completes")
    ap.add_argument("--no-probe", action="store_true", help="Skip tool-support probe before each model")
    ap.add_argument("--no-stop", action="store_true", help="Don't ollama stop after each model")
    ap.add_argument("--probe-only", action="store_true",
                    help="Send tool-support probe to every model, then exit (no tasks run). "
                         "Use this to validate all models' tool templates before a full sweep.")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # try both CWD-relative and repo-root-relative
        if not config_path.exists():
            alt = _REPO_ROOT / args.config
            if alt.exists():
                config_path = alt
    config = load_sweep_config(config_path)

    # Apply CLI overrides
    if args.models:
        wanted = {s.strip() for s in args.models.split(",")}
        config.models = [m for m in config.models if m.name in wanted]
        if not config.models:
            raise SystemExit(f"no models matched --models {args.models}")
    if args.tasks:
        config.tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    if args.variants:
        config.variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    if args.runs is not None:
        config.runs_per_combo = args.runs
    if args.no_probe:
        config.probe_before_run = False
    if args.no_stop:
        config.auto_stop = False

    # In probe-only mode we don't need tasks; skip task resolution nicely
    if args.probe_only:
        records = run_sweep(config, dry_run=args.dry_run, probe_only=True)
        print()
        print("=" * 60)
        print(f"PROBE COMPLETE: {len(config.models)} model(s)")
        print("=" * 60)
        if config.compat_out:
            print(f"  results logged to: {config.compat_out}")
        print("Review the compat_out YAML; models with status=native pass, others need fixes.")
        return 0

    records = run_sweep(config, dry_run=args.dry_run)
    _print_final_summary(records)

    if args.eval and not args.dry_run:
        run_evaluation(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
