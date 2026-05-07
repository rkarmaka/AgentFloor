# AgentFloor

> **AgentFloor: How Far Up the Tool-Use Ladder Can Small Open-Weight Models Go?**
> Ranit Karmakar, Jayita Chatterjee
> arXiv preprint [2605.00334](https://arxiv.org/abs/2605.00334), May 2026.

A deterministic, capability-tiered benchmark for evaluating LLM agents on
multi-step tool-calling tasks. AgentFloor isolates progressively harder
cognitive demands across six tiers — from instruction-following without
tools (A0) through long-horizon planning under persistent constraints (E)
— inside a fixed abstract-tool environment with no filesystem, no live
APIs, and no plausible route to pretraining-corpus contamination.

## Headline finding

Across 16,542 scored runs spanning 16 open-weight models (0.27B–32B) plus
GPT-5 as the frontier anchor, the strongest open-weight model
(`gemma4:26b`) is statistically equivalent to GPT-5 in aggregate
(Δ = +0.4 pp; 90% CI [−5.1, +5.8]) at substantially lower cost and
latency. The frontier advantage concentrates almost entirely on the E
tier (long-horizon planning); on A0/A/B/C/D, small or mid-scale
open-weight models match or beat GPT-5 in our corpus. See the paper for
the full per-tier TOST equivalence analysis, capability heatmap, and
failure-mode cascade.

## What's in this repo

The full benchmark, harness, sweep configurations, and re-scoring tools.

| Path | What's there |
|---|---|
| `tasks/{A0,A,B,C,D,E}/` | 30 task YAMLs, 5 prompt-variant files (`<id>.variants.yaml`), and 4 instance-variant files (`A1`/`B1`/`C1`/`E1`) |
| `tasks/registry.yaml` | Active-task list, level descriptions, defaults |
| `tasks/task_template.yaml` | Reference for the task-YAML field set |
| `fixtures/{A,B,C,D,E}/` | YAML data the tools operate on (24 fixtures) |
| `harness/` | Runner, evaluator, metrics, providers, eval checks |
| `harness/eval_checks/` | Four scoring families (`final_answer`, `submission`, `trajectory`, `forbidden`) plus `llm_judge` for the two semantic predicates |
| `harness/providers/` | Adapters for Anthropic, OpenAI, Gemini, and any OpenAI-compatible server (vLLM, Ollama, NIM, Together, …) |
| `runs/` | Entry-point scripts: `run_sweep.py` (batch with resume), `run_metrics.py` (aggregate), `rescore_diff.py` (re-score without re-running), and per-backend launchers |
| `sweep_configs/` | Every sweep referenced in the paper, ready to launch |

## Citation

```bibtex
@article{karmakar2026agentfloor,
  title   = {{AgentFloor}: How Far Up the Tool-Use Ladder Can Small Open-Weight Models Go?},
  author  = {Karmakar, Ranit and Chatterjee, Jayita},
  journal = {arXiv preprint arXiv:2605.00334},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.00334},
}
```

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

Two orthogonal axes of variation, both checked into the repo:

- **Prompt variants** (`<task>.variants.yaml`, `v1`–`v5`): five hand-crafted
  rephrasings of the same task along the five axes declared in
  `variant_axes` (paraphrase, distractor text, ID-format noise, reordered
  instruction, mild typo). `v0` is the original `example_prompt`. Every
  task ships variants. E1 additionally ships `v6_explicit` and
  `v7_explicit` for the explicit-submission ablation (paper §5.2).
- **Instance variants** (`<task>.instance_variants.yaml`, `i1`–`i5`):
  alternative `gold_state` overrides, used for the input-instance
  ablation. A1, B1, C1, E1 ship instance variants.

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
default-pass. To run the LLM judge (≈ $0.50/$2.00 per 1M in/out tokens on
`gpt-5-nano`), export `AGENTFLOOR_LLM_JUDGE=1` and `OPENAI_API_KEY`
before invoking the scorer. Predicates affected:
`hallucinated_facts_not_in_passage` (A02/A03/A05) and
`inconsistent_recovery_and_submission` (E5), plus the `element_coverage`
keyword fallback.

## Reproducing paper numbers

Each sweep config in `sweep_configs/` corresponds to a block of paper
results.

| Config | Paper section | What it produces |
|---|---|---|
| `frontier_anchor.yaml` | Frame A anchor (§4.1, Table 1) | GPT-5 (and optional Claude / Gemini) anchor: 3 variants × 3 runs |
| `ollama_full.yaml` + `ollama_full_pass2.yaml` | Main SLM sweep (§4.2, Table 2) | 16 open-weight models × 30 tasks × 5 variants × 5 runs = 12,000 runs |
| `nim_slm.yaml` | Cross-backend probe | NVIDIA NIM-hosted open models |
| `vllm_colab.yaml`, `vllm_rescue.yaml` | Cross-backend probe | vLLM-served open models |
| `cross_backend_vllm.yaml` | §B (appendix) | Same model on multiple backends (consistency check) |
| `instance_ablation.yaml` | §5.2 instance variation | Input-instance ablation (i1–i5 on A1/B1/C1/E1) |
| `e_explicit_ablation.yaml` | §5.2 explicit-submission | E-tier prompt-explicitness ablation |
| `reasoning_effort_ablation.yaml` | §5.2 reasoning effort | Reasoning-effort ablation on reasoning models |
| `structured_prompt_ablation.yaml` | §5.2 structured prompt | Structured-prompt scaffolding ablation |
| `qwen3_nothink_ablation.yaml` | §5.2 reasoning toggle | Qwen3 thinking-vs-no-thinking |
| `gpt5_de_max_steps.yaml` | §5.2 step-budget ×2 | GPT-5 D/E step-budget ablation |
| `api_frontier.yaml` | (sanity check) | Cheaper frontier APIs |
| `smoke.yaml`, `smoke_local.yaml` | (smoke tests) | Short subsets for development |

Each sweep is resume-safe — re-running the same command picks up at the
first missing result file. The full SLM main sweep takes O(days) on a
single Ollama-capable host; reproducing the headline aggregate-equivalence
claim from §4.1 requires both `ollama_full*.yaml` and `frontier_anchor.yaml`
to be run in full. A representative end-to-end:

```bash
# Main SLM sweep (12,000 runs)
python runs/run_sweep.py --config sweep_configs/ollama_full.yaml --eval
python runs/run_sweep.py --config sweep_configs/ollama_full_pass2.yaml --eval

# Frontier anchor (270 GPT-5 runs)
python runs/run_sweep.py --config sweep_configs/frontier_anchor.yaml --eval

# Aggregate
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

LLM-provider determinism is best-effort even at `temperature=0`. Reported
numbers depend on:

- Provider SDK versions, model snapshots/digests, and server build (for
  Ollama / vLLM / NIM). Pin where possible; the paper lists the exact
  Ollama tags used in May 2026.
- Whether the `AGENTFLOOR_LLM_JUDGE` env var is set when scoring (see the
  Scoring section above).
- The bootstrap CI in `harness.metrics` is seeded (`random.Random(42)`),
  so CIs are deterministic given the same input runs.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The
benchmark tasks, fixtures, and sweep configurations are released under
CC-BY 4.0 alongside the arXiv preprint.
