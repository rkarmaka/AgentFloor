# AgentFloor

A benchmark for evaluating LLM agents on multi-step tool-calling tasks across
six difficulty tiers, from pure-NLU baselines (A0) to long-horizon planning
under uncertainty (E).

The benchmark ships 30 tasks (5 per tier × 6 tiers), 5 prompt variants per
task, instance variants for a 4-task ablation subset, fixtures the tools
operate on, a runner that drives any LLM provider, and a scorer that produces
the headline TCR / SDR / THI / LSR / ERR / ERT metrics.

## Repository layout

| Path | What's there |
|---|---|
| `tasks/{A0,A,B,C,D,E}/` | Task YAMLs (`<id>.yaml`), prompt variants (`<id>.variants.yaml`), and instance variants (`<id>.instance_variants.yaml`) |
| `tasks/registry.yaml` | The list of 30 active tasks, level descriptions, defaults |
| `tasks/task_template.yaml` | Reference of the task-YAML field set |
| `fixtures/{A,B,C,D,E}/` | YAML data the tools operate on, one per fixture-using task |
| `harness/` | Runner, evaluator, metrics, providers, eval checks |
| `harness/eval_checks/` | The four scoring families: `final_answer`, `submission`, `trajectory`, `forbidden`, plus `llm_judge` for predicates needing semantic judgment |
| `harness/providers/` | Provider adapters for Anthropic, OpenAI, Gemini, and any OpenAI-compatible server (vLLM, Ollama, NIM, Together, etc.) |
| `runs/` | Entry-point scripts: `run_sweep.py` (batch), `run_metrics.py` (aggregate), and one-off scripts per backend |
| `sweep_configs/` | YAML configs for the sweeps reported in the paper |

## Requirements

- Python ≥ 3.10 (the harness uses `match` statements and `X | Y` type hints)
- Provider SDKs as needed — install only what you'll use:
  - `anthropic` for Claude
  - `openai` for GPT and any OpenAI-compatible backend (vLLM, Ollama, NIM, …)
  - `google-genai` for Gemini

```bash
pip install -r requirements.txt
```

Set API keys in `.env.local` at the repo root (or in the environment):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
NVIDIA_API_KEY=nvapi-...
```

## Quickstart

Run one task on Gemini:

```bash
python runs/run_gemini.py --task A1 --model gemini-2.5-flash --runs 1
```

Run a single task against a local Ollama server:

```bash
python runs/run_sweep.py --config sweep_configs/smoke_local.yaml --tasks A1
```

Run the full smoke sweep (a few small models on a handful of tasks, ~minutes):

```bash
python runs/run_sweep.py --config sweep_configs/smoke.yaml --eval
```

Score a results directory and print the headline tables:

```bash
python runs/run_metrics.py results/
```

## Anatomy of a task

Each task is a single YAML file with a fixed schema. From `tasks/A/A1.yaml`:

```yaml
task_id: A1
fixture: fixtures/A/A1_product_catalog.yaml
name: Look up a record by ID
level: A
goal: Look up a record by its given ID and report a specific field value.
tools_available: [lookup_record]
max_steps: 2
gold_state:
  target_record_id: P104
  target_field: status
  target_value: active
oracle_evaluator:
  final_answer_check:
    type: exact_field_match
    expected_fields: {status: active}
trace_requirements:
  must_call_tools: [lookup_record]
  forbidden_behaviors:
    - type: hallucinated_tool
    - type: terminate_without_answer
example_prompt: Look up product record P104 and tell me its current status.
```

Key fields:

- **`fixture`** — path (relative to repo root) of the YAML data the tools see.
  `null` for pure-NLU A0 tasks and a few tasks like A3 (pure compute).
- **`tools_available`** — the subset of the eight built-in tools
  (`lookup_record`, `search_records`, `get_attribute`, `list_options`,
  `check_constraint`, `compare_records`, `compute_value`, `submit_decision`)
  the model is allowed to call.
- **`gold_state`** — ground-truth answer and any intermediate values the
  scorer needs.
- **`oracle_evaluator`** — declares the `final_answer_check` shape and any
  `submission_check` and `trajectory_check` predicates.
- **`trace_requirements`** — required tool calls, optional tools, and
  declared `forbidden_behaviors` (each one is a named predicate in
  `harness/eval_checks/forbidden.py`).
- **`example_prompt`** — the user message sent to the model in the canonical
  v0 variant.

## Variants and instance variants

Two orthogonal axes of variation:

- **Prompt variants** (`<task>.variants.yaml`, `v1`–`v5`): five hand-crafted
  rephrasings of the same task along the five axes declared in
  `variant_axes` (e.g. paraphrase, distractor text, ID-format noise,
  reordered instruction, mild typo). `v0` is the original `example_prompt`.
  Every task ships variants.
- **Instance variants** (`<task>.instance_variants.yaml`, `i1`–`i5`):
  alternative `gold_state` overrides, used for the input-instance ablation.
  Only A1, B1, C1, E1 ship instance variants today.

## Scoring

`harness/evaluator.py` produces a per-run `TaskScore` by AND-ing four
checker families:

1. `final_answer` — does the free-text answer contain the expected value?
2. `submission` — does the `submit_decision` payload match `gold_state`?
3. `trajectory` — did the call log follow the required tool sequence and
   pass every declared trajectory predicate?
4. `forbidden` — did the run avoid every declared forbidden behavior?

The Boolean AND is the headline TCR pass/fail. The detailed per-check
breakdown is preserved in `TaskScore.details` so the metrics layer can
compute SDR / THI / LSR / ERT / ERR.

A few `forbidden` and `final_answer` predicates require semantic judgment
(e.g. "did the model hallucinate facts not in the source passage?"). These
delegate to an LLM judge (`harness/eval_checks/llm_judge.py`), which is
cached at `results/llm_judge_cache.jsonl` and gated by
`AGENTFLOOR_LLM_JUDGE=1`.

**The judge is disabled by default**, in which case the affected predicates
return a stubbed PASS (with `stubbed=True` recorded in the per-check
details). This is by design: canonical baseline runs are scored without
calling an external API, and the per-cell stubbed-coverage table prints
alongside the TCR table so consumers can see which cells leaned on a
default-pass. To run the LLM judge (≈ $0.50–$2 per 1M tokens on
`gpt-5-nano`), export `AGENTFLOOR_LLM_JUDGE=1` and `OPENAI_API_KEY`
before invoking the scorer. Predicates affected: `hallucinated_facts_not_in_passage`
(A02/A03/A05) and `inconsistent_recovery_and_submission` (E5), plus the
`element_coverage` keyword fallback.

## Reproducing paper numbers

Each sweep config in `sweep_configs/` corresponds to a block of paper
results:

| Config | What it produces |
|---|---|
| `frontier_anchor.yaml` | Closed-source frontier anchor (gpt-5, claude-sonnet-4-6, gemini-2.5-pro) |
| `api_frontier.yaml` | Cheaper frontier APIs |
| `ollama_full.yaml`, `ollama_full_pass2.yaml` | Open-source SLM sweep via Ollama |
| `vllm_colab.yaml`, `vllm_rescue.yaml` | vLLM-served open models |
| `nim_slm.yaml` | NVIDIA NIM-hosted open models |
| `cross_backend_vllm.yaml` | Same model on multiple backends (consistency check) |
| `instance_ablation.yaml` | Input-instance ablation (i1–i5 on 4 tasks) |
| `e_explicit_ablation.yaml` | E-tier prompt-explicitness ablation |
| `reasoning_effort_ablation.yaml` | Reasoning-effort ablation on reasoning models |
| `structured_prompt_ablation.yaml` | Structured-prompt scaffolding ablation |
| `qwen3_nothink_ablation.yaml` | Qwen3 thinking-vs-no-thinking ablation |
| `gpt5_de_max_steps.yaml` | gpt-5 D/E step-budget ablation |

Each sweep is resume-safe — re-running the same command picks up at the
first missing result file. A representative end-to-end:

```bash
python runs/run_sweep.py --config sweep_configs/ollama_full.yaml --eval
python runs/run_metrics.py results/ --subset paper_baseline
```

### Re-scoring without re-running

The scorer is decoupled from the runner. If the scoring code changes, you
can re-score existing trajectories without paying for any model calls:

```bash
# Read-only: see what would change vs the existing .score.json files.
python runs/rescore_diff.py results/

# Inspect every flipped run, not just the summary.
python runs/rescore_diff.py results/ --details

# Overwrite the .score.json files after you're satisfied with the diff.
python runs/rescore_diff.py results/ --write
```

The diff reports per-(model, tier) TCR deltas, which check leg drove each
flip, and how many runs errored vs were unchanged.

### Reproducibility caveats

LLM provider determinism is best-effort even at `temperature=0`. Reported
numbers depend on:

- Provider SDK versions, model snapshots/digests, and server build (for
  Ollama / vLLM / NIM). Pin where possible.
- Whether the `AGENTFLOOR_LLM_JUDGE` env var is set when scoring (see the
  Scoring section above).
- The bootstrap CI in `harness.metrics` is seeded (`random.Random(42)`),
  so CIs are deterministic given the same input runs.

## License

MIT — see [LICENSE](LICENSE).
