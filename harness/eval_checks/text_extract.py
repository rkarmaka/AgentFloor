"""Text-to-field extraction helpers for the final_answer checker.

The benchmark's task YAMLs declare expected outputs as structured fields:

    expected_fields: {status: active}

But the model produces free-form prose:

    "The product record P104 corresponds to the Alpha Sensor, which is currently active."

We need to decide whether the prose contains the expected answer. Three
strategies, in order of strictness:

  STRICT     — both the field NAME and the field VALUE appear in the text,
               within `proximity_window` characters of each other. (Strategy B
               in the design plan.)
  LENIENT    — the field VALUE appears as a standalone token in the text,
               but the field NAME is absent. (Strategy A.) We accept this with
               a `weak_match: True` flag in details so the metrics layer can
               surface weak matches for review.
  FAIL       — value not in text at all.

The locked design said "Strategy B." After looking at real Qwen outputs
during step 0 recon, every A1 success ("currently active") would have failed
strict B because Qwen omits the field name. So we ship a hybrid: STRICT first,
LENIENT fallback with the weak_match flag. This is strictly more lenient than
B; surfaced for user veto at checkpoint 5.

Numeric matching uses extract_numbers() against the text and treats the
expected value as found if any extracted number equals it (within tolerance).

These functions are pure: no I/O, no global state, deterministic.
"""

from __future__ import annotations

import re
from typing import Any

# How many characters between the field name and field value count as
# "proximate." 60 chars covers a typical short sentence.
DEFAULT_PROXIMITY_WINDOW = 60

# Pattern for extracting numeric tokens. Catches integers, decimals,
# negatives, comma-grouped thousands ("1,234.5"), and trailing percent
# signs ("50%"). The optional trailing "%" is captured so the extractor
# can emit both the literal value (50) and the fractional equivalent
# (0.5) for any numeric_match task whose gold value uses the other form.
_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s*%)?"   # 1,234 or 1,234.5 [optional %]
    r"|-?\d+(?:\.\d+)?(?:\s*%)?"                  # 1, 1.5, -3 [optional %]
)


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace for case-insensitive matching."""
    return re.sub(r"\s+", " ", s.lower())


def _word_bounded_pattern(value: str) -> str:
    r"""Build a case-insensitive-boundary-aware regex for `value`.

    The naive pattern ``\b...\b`` fails when the value ends (or starts)
    with a non-word character — e.g. "Inc." in JSON like ``"Inc.",``
    never matches because both ``.`` and ``,`` are non-word, so no
    word-boundary transition exists at the trailing edge. We substitute
    explicit ``(?<!\w)`` / ``(?!\w)`` lookarounds which enforce the
    same "not inside a larger word" semantics but work regardless of
    whether the value itself starts or ends with punctuation.
    """
    return r"(?<!\w)" + re.escape(value) + r"(?!\w)"


def value_in_text(value: Any, text: str) -> bool:
    """Case-insensitive word-boundary search for `value` in `text`.

    Treats numeric values by string conversion. Falsy text → False.
    Empty value → False (we never want to match the empty string).
    """
    if not text:
        return False
    sval = str(value).strip()
    if not sval:
        return False
    norm_text = _normalize(text)
    norm_val = _normalize(sval)
    return bool(re.search(_word_bounded_pattern(norm_val), norm_text))


def key_value_proximate(
    key: str,
    value: Any,
    text: str,
    window: int = DEFAULT_PROXIMITY_WINDOW,
) -> bool:
    """Both `key` and `value` appear in `text`, within `window` chars.

    Strict-mode field match. Returns False if either is missing or they
    are too far apart in the text. Used for the STRICT pass before
    falling back to value-only.
    """
    if not text:
        return False
    sval = str(value).strip()
    skey = str(key).strip()
    if not sval or not skey:
        return False
    norm_text = _normalize(text)
    norm_key = _normalize(skey)
    norm_val = _normalize(sval)

    # Find every position where the key appears (word-bounded).
    key_positions = [
        m.start() for m in re.finditer(_word_bounded_pattern(norm_key), norm_text)
    ]
    val_positions = [
        m.start() for m in re.finditer(_word_bounded_pattern(norm_val), norm_text)
    ]
    if not key_positions or not val_positions:
        return False

    # Any pair within `window` of each other passes.
    for kp in key_positions:
        for vp in val_positions:
            if abs(kp - vp) <= window:
                return True
    return False


def extract_numbers(text: str) -> list[float]:
    """Pull all numeric tokens from `text` and return them as floats.

    Handles:
      - Bare integers and decimals: 1, -3, 1.5
      - Comma-grouped thousands: "1,234" → 1234.0
      - Trailing percent: "50%" emits BOTH 50 and 0.5 so a numeric_match
        task whose gold is in the other convention still finds a match.

    Tokens like "P104" are rejected: a digit immediately preceded by a
    letter is not extracted (so "P104" never produces 104).
    """
    if not text:
        return []
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        start = m.start()
        if start > 0 and text[start - 1].isalpha():
            continue
        token = m.group()
        is_percent = token.rstrip().endswith("%")
        digits = token.rstrip().rstrip("%").rstrip().replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        if is_percent:
            # Emit both interpretations so gold can be in either form.
            out.append(value)
            out.append(value / 100.0)
        else:
            out.append(value)
    return out


def field_match_one(
    key: str,
    expected_value: Any,
    text: str,
    *,
    window: int = DEFAULT_PROXIMITY_WINDOW,
) -> tuple[bool, str, dict[str, Any]]:
    """Check one (key, value) pair against `text` using STRICT-then-LENIENT.

    Returns (matched, mode, details).
      mode is one of: "strict", "lenient", "missing"
      details carries weak_match flag for the lenient case.
    """
    if key_value_proximate(key, expected_value, text, window=window):
        return True, "strict", {"matched_field": key, "match_mode": "strict"}
    if value_in_text(expected_value, text):
        return (
            True,
            "lenient",
            {
                "matched_field": key,
                "match_mode": "lenient",
                "weak_match": True,
                "note": "value matched but field name absent — review for false positive",
            },
        )
    return False, "missing", {"matched_field": key, "match_mode": "missing"}


def numeric_match_one(
    key: str,
    expected_value: float,
    text: str,
    tolerance: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Check that `expected_value` (numeric) appears in `text` numbers.

    Tolerance is absolute (|extracted - expected| <= tolerance).
    Returns (matched, details).
    """
    extracted = extract_numbers(text)
    for n in extracted:
        if abs(n - float(expected_value)) <= tolerance:
            return True, {
                "matched_field": key,
                "matched_number": n,
                "expected": float(expected_value),
            }
    return False, {
        "matched_field": key,
        "expected": float(expected_value),
        "extracted_numbers": extracted,
    }
