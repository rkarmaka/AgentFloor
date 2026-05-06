"""Prompt variants for the AgentFloor benchmark.

Each task in `tasks/` declares 5 axes in its `variant_axes` field
(e.g. paraphrase, distractor_text, ID_format_noise, reordered_instruction,
mild_typo). For each task, a sibling file `<task_id>.variants.yaml`
holds 5 hand-crafted variant prompts (v1..v5), one per axis. v0 is
reserved for the original `example_prompt` declared in the task YAML.

The runtime loader resolves a (task, variant_id) pair to the prompt
text the runner should send to the model. Variants are part of the
benchmark — checked into the repo, audited like the task YAMLs themselves.
"""

from .loader import load_variant_text, list_variant_ids, VariantNotFoundError

__all__ = ["load_variant_text", "list_variant_ids", "VariantNotFoundError"]
