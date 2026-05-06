"""LLM-as-judge for forbidden-behavior predicates that need semantic evaluation.

Two predicates require an LLM to judge:
  hallucinated_facts_not_in_passage   — A02/A03/A05
  inconsistent_recovery_and_submission — E5

Architecture:
  judge(predicate, context) → CheckResult
    ├─ env-var guard: AGENTFLOOR_LLM_JUDGE=1 required, else stubbed pass
    ├─ cache lookup (SHA256 of predicate + inputs)
    ├─ _call_llm() → raw verdict text
    ├─ parse VERDICT: PASS|FAIL + REASON
    └─ cache write + cost accounting

Model: gpt-5-nano via OpenAI API (~$0.50/$2.00 per 1M tokens).
Cache: append-only JSONL at results/llm_judge_cache.jsonl (at the repo root).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import CheckResult


_JUDGE_MODEL = "gpt-5-nano"
# gpt-5-nano is a reasoning model: completion_tokens_details.reasoning_tokens
# typically consumes 500–800 tokens even for a one-sentence verdict, before
# any visible output is produced. A 256 cap was empirically observed to
# return empty content on every call (verdict defaulted to PASS, which made
# the env-var-enabled run indistinguishable from the disabled stub). 2048
# leaves comfortable headroom — observed a 740-token completion (704
# reasoning + 36 output) on a trivial probe; production prompts are larger
# but reasoning length is mostly task-difficulty-driven, not prompt-size.
_MAX_TOKENS = 2048
_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "llm_judge_cache.jsonl"

_cost_tracker: dict[str, Any] = {
    "calls": 0,
    "cache_hits": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def get_cost_summary() -> dict[str, Any]:
    input_per_mtok = 0.50
    output_per_mtok = 2.00
    cost = (
        _cost_tracker["input_tokens"] / 1_000_000 * input_per_mtok
        + _cost_tracker["output_tokens"] / 1_000_000 * output_per_mtok
    )
    return {**_cost_tracker, "estimated_cost_usd": round(cost, 4)}


def _is_enabled() -> bool:
    return os.environ.get("AGENTFLOOR_LLM_JUDGE", "").strip() == "1"


def _cache_key(predicate: str, **fields: str) -> str:
    blob = json.dumps({"p": predicate, **fields}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not _CACHE_PATH.exists():
        return cache
    for line in _CACHE_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            cache[entry["key"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue
    return cache


def _append_cache(entry: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _call_llm(system: str, user_prompt: str) -> tuple[str, int, int]:
    """Single-turn OpenAI completion. Returns (text, input_tokens, output_tokens).

    gpt-5-nano is a reasoning model: requires max_completion_tokens (not
    max_tokens) and rejects custom temperature/top_p.
    """
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=_JUDGE_MODEL,
        max_completion_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content or "" if resp.choices else ""
    usage = resp.usage
    return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """Extract VERDICT: PASS|FAIL and REASON from judge output."""
    verdict = None
    reason = ""
    for line in text.splitlines():
        upper = line.strip().upper()
        if upper.startswith("VERDICT:"):
            val = upper.split(":", 1)[1].strip()
            if "PASS" in val:
                verdict = True
            elif "FAIL" in val:
                verdict = False
        if line.strip().upper().startswith("REASON:"):
            reason = line.strip().split(":", 1)[1].strip()
    return verdict, reason


def judge(predicate: str, context: dict) -> CheckResult:
    """Main entry point. Dispatches to predicate-specific prompt, calls LLM, returns CheckResult."""
    if not _is_enabled():
        return CheckResult.ok(
            f"stubbed: {predicate} requires LLM judge (set AGENTFLOOR_LLM_JUDGE=1)",
            stubbed=True,
            not_implemented=True,
        )

    builder = _PROMPT_BUILDERS.get(predicate)
    if not builder:
        return CheckResult.ok(
            f"llm_judge: no prompt builder for {predicate!r}",
            stubbed=True,
        )

    system, user_prompt, cache_fields = builder(context)
    key = _cache_key(predicate, **cache_fields)

    cache = _load_cache()
    if key in cache:
        _cost_tracker["cache_hits"] += 1
        entry = cache[key]
        passed = entry.get("verdict_pass", True)
        reason = entry.get("reason", "cached")
        # Reason starting with "unparseable judge output" means the original
        # call failed to produce a parseable VERDICT line and we defaulted
        # to PASS. Surface that as stubbed so the metrics CLI flags the
        # cell — otherwise these defaults inflate TCR silently.
        if isinstance(reason, str) and reason.startswith("unparseable judge output"):
            return CheckResult.ok(reason, cached=True, stubbed=True, unparseable=True)
        return CheckResult.ok(reason, cached=True) if passed else CheckResult.fail(reason, cached=True)

    try:
        raw, in_tok, out_tok = _call_llm(system, user_prompt)
    except Exception as e:
        print(f"  WARN llm_judge({predicate}): API error {type(e).__name__}: {e}", file=sys.stderr)
        return CheckResult.ok(
            f"llm_judge: API error, defaulting to PASS — {e}",
            llm_error=True,
            stubbed=True,
        )

    _cost_tracker["calls"] += 1
    _cost_tracker["input_tokens"] += in_tok
    _cost_tracker["output_tokens"] += out_tok

    verdict, reason = _parse_verdict(raw)
    is_unparseable = verdict is None
    if is_unparseable:
        print(f"  WARN llm_judge({predicate}): unparseable verdict, defaulting to PASS", file=sys.stderr)
        verdict = True
        reason = f"unparseable judge output (defaulting PASS): {raw[:200]}"

    _append_cache({
        "key": key,
        "predicate": predicate,
        "verdict_pass": verdict,
        "reason": reason,
        "raw": raw,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "timestamp": time.time(),
        **cache_fields,
    })

    if is_unparseable:
        return CheckResult.ok(
            reason, llm_judged=True, stubbed=True, unparseable=True,
        )
    if verdict:
        return CheckResult.ok(reason or "LLM judge: PASS", llm_judged=True)
    return CheckResult.fail(reason or "LLM judge: FAIL", llm_judged=True)


# ------------------------------------------------------------------
# Predicate-specific prompt builders
# ------------------------------------------------------------------
# Each returns (system_prompt, user_prompt, cache_fields_dict).
# cache_fields are hashed into the cache key.


def _build_hallucinated_facts(ctx: dict) -> tuple[str, str, dict]:
    passage = ctx.get("passage", "")
    final_text = ctx.get("final_text", "")
    task_id = ctx.get("task_id", "")

    system = (
        "You are a factual consistency judge for an AI benchmark. "
        "Your job is to determine whether a model's response contains facts "
        "that are NOT present in or directly inferable from the source passage. "
        "Minor formatting differences (e.g. date formats, abbreviations) are acceptable. "
        "Only flag genuinely fabricated information."
    )
    user_prompt = (
        f"SOURCE PASSAGE:\n{passage}\n\n"
        f"MODEL RESPONSE:\n{final_text}\n\n"
        "Does the model response contain any hallucinated facts — information that "
        "is NOT present in or directly inferable from the source passage?\n\n"
        "Answer with:\n"
        "VERDICT: PASS (if no hallucinated facts) or FAIL (if hallucinated facts found)\n"
        "REASON: (one sentence explaining your judgment)"
    )
    text_hash = hashlib.sha256((passage + final_text).encode()).hexdigest()[:16]
    return system, user_prompt, {"task_id": task_id, "text_hash": text_hash}


def _build_inconsistent_recovery(ctx: dict) -> tuple[str, str, dict]:
    trajectory = ctx.get("trajectory_text", "")
    submission = ctx.get("submission_json", "")
    task_id = ctx.get("task_id", "")

    system = (
        "You are a consistency judge for an AI benchmark. "
        "An agent was given a deployment task where a compatibility check failed. "
        "It had two recovery options:\n"
        "- Path A: downgrade to v2.1 (requires region-east)\n"
        "- Path B: upgrade to v3.0 (requires region-west)\n\n"
        "Your job: determine if the agent's stated reasoning about which path it chose "
        "is consistent with the actions it actually took."
    )
    user_prompt = (
        f"AGENT TRAJECTORY:\n{trajectory}\n\n"
        f"FINAL SUBMISSION:\n{submission}\n\n"
        "Is the agent's stated recovery reasoning consistent with its actual "
        "tool calls and submission? A model that says 'I chose path A' but submits "
        "region-west is inconsistent.\n\n"
        "VERDICT: PASS (if consistent) or FAIL (if reasoning contradicts actions)\n"
        "REASON: (one sentence)"
    )
    text_hash = hashlib.sha256((trajectory + submission).encode()).hexdigest()[:16]
    return system, user_prompt, {"task_id": task_id, "text_hash": text_hash}


def _build_element_coverage(ctx: dict) -> tuple[str, str, dict]:
    """Per-element semantic coverage judge for the element_coverage check.

    Fires only as a fallback when the keyword-based matcher in
    final_answer._check_element_coverage cannot match a required concept
    tag (e.g. payment_service_failure) against the model's final_text.
    Caller invokes judge('element_coverage', ctx) once per missing element;
    cache key is per (task_id, element, passage+text+element hash) so
    different elements on the same trace produce distinct cache entries.

    Motivating case (F-009): gpt-5 A02 traces produced technically-correct
    summaries using HTTP-layer vocabulary ('checkouts', '503', 'connection
    pool') instead of payment-service vocabulary, even though the source
    passage explicitly says 'payment processing service'. Pure keyword
    matching marked these missing; semantic judge accepts the paraphrase.
    """
    passage = ctx.get("passage", "")
    final_text = ctx.get("final_text", "")
    element = ctx.get("element", "")
    task_id = ctx.get("task_id", "")

    system = (
        "You are a semantic coverage judge for an AI benchmark. Your job: "
        "determine whether a model's response semantically covers a required "
        "concept from a source passage. The required concept is given as a "
        "tag (e.g. 'payment_service_failure'). Models may use different but "
        "conceptually equivalent vocabulary — accept paraphrases and "
        "technically-precise terminology that refers to the same concept "
        "described in the passage; reject responses that genuinely omit the "
        "concept."
    )
    user_prompt = (
        f"SOURCE PASSAGE:\n{passage}\n\n"
        f"MODEL RESPONSE:\n{final_text}\n\n"
        f"REQUIRED CONCEPT TAG: {element}\n\n"
        "Does the model response semantically cover this concept, given the "
        "source passage? Accept paraphrases and equivalent technical vocabulary; "
        "reject if the concept is genuinely absent.\n\n"
        "Answer with:\n"
        "VERDICT: PASS (if the concept is covered) or FAIL (if the concept is absent)\n"
        "REASON: (one sentence explaining your judgment)"
    )
    text_hash = hashlib.sha256(
        (passage + "|" + final_text + "|" + str(element)).encode()
    ).hexdigest()[:16]
    return system, user_prompt, {
        "task_id": task_id,
        "element": str(element),
        "text_hash": text_hash,
    }


_PROMPT_BUILDERS = {
    "hallucinated_facts_not_in_passage": _build_hallucinated_facts,
    "inconsistent_recovery_and_submission": _build_inconsistent_recovery,
    "element_coverage": _build_element_coverage,
}
