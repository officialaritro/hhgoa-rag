"""Rule-based guardrails: off-topic, unsafe-input, and groundedness checks.

Plan Global Constraints: no second LLM call for guardrails -- every check
here is a threshold on a similarity score or a pattern match.

The similarity thresholds below are set from measured score distributions, not
guessed, and each carries the measurement that justifies it. Re-measure with
scripts/tune_thresholds.py before changing one, and check the change against
tests/test_guardrail_calibration.py, which pins the false-refusal budget.
"""

import os
import re

from pydantic import BaseModel

from app.embeddings import cosine_similarity, embed
from app.schemas import RetrievalOutput


def _threshold(env_var: str, default: float) -> float:
    """Read a guardrail threshold from the environment, falling back to the
    documented default. These are meant to be retuned against measured score
    distributions (scripts/tune_thresholds.py), and retuning should not
    require a code change -- set the value in .env and restart."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Off-topic gate, measured PER STRATEGY against the built indices (2026-08-19,
# 200 sampled corpus queries vs. 7 off-topic probes; the recorded distributions
# and the budget they have to satisfy live in tests/test_guardrail_calibration.py).
#
# The two indices sit on different score scales and cannot share a threshold.
# Semantic chunks are shorter -- 338k chunks over the same corpus that
# fixed-size splits into 101k -- so every cosine runs higher: in-corpus median
# 0.779 vs 0.741, highest off-topic probe 0.743 vs 0.612. Any single shared
# value is therefore miscalibrated for one of them. That is what went wrong
# before: 0.70 was verified against the semantic index and shipped against a
# fixed_size default, where it refused 38.5% of real in-corpus questions,
# including one whose query is in the corpus verbatim (it scores 0.687 on
# fixed_size and 0.828 on semantic).
#
# Each value sits at that index's in-corpus p05 lower tail, costing 4.5%
# (fixed_size) / 8.0% (semantic) false refusals, and rejects every
# clearly-unrelated probe except "what is my bank account password". That one
# scores 0.612/0.743 because a web-search corpus really does contain
# bank-and-password passages: its *topic* is in-corpus even though its answer
# cannot be, so no retrieval-similarity threshold separates it without cutting
# deep into real questions. The generation prompt ("say so if the passages do
# not answer") and the groundedness guard are what refuse that class.
#
# Mean-of-top-5 was measured as an alternative signal and rejected: at matched
# leak rates it refused 2.0% vs 4.5% of real questions on fixed_size but 10%
# vs 8.0% on semantic -- no consistent gain for a second signal to maintain.
_DEFAULT_OFFTOPIC_THRESHOLDS = {
    "fixed_size": _threshold("OFFTOPIC_SIMILARITY_THRESHOLD_FIXED_SIZE", 0.55),
    "semantic": _threshold("OFFTOPIC_SIMILARITY_THRESHOLD_SEMANTIC", 0.60),
}

# Real generated answers score 0.756-0.902 against their retrieved context;
# answers paired with unrelated context score below 0.11. 0.40 clears the
# grounded floor with margin while staying far above the ungrounded range.
#
# Note scripts/tune_thresholds.py recommends a much lower value here (~0.02)
# because it measures the dataset's terse Eng_Answer rather than real model
# output. Generated answers quote the passages, so they score far higher --
# trust the measured pipeline over that proxy.
_DEFAULT_GROUNDEDNESS_THRESHOLD = _threshold("GROUNDEDNESS_SIMILARITY_THRESHOLD", 0.40)

# Chosen empirically as an illustrative starting set for the MVP guardrail,
# not an exhaustive safety classifier. Env-overridable is left for Task 9
# retuning; the list itself is a plain module constant since it is not a
# single scalar.
#
# Hate speech is deliberately not covered here: matching it reliably needs a
# slur list, and a slur list does not belong in this repo's source -- that
# category needs a hosted classifier, which the "no second LLM call for
# guardrails" constraint above already rules out for this MVP.
_UNSAFE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bbomb\b",
        r"\bhurt (people|someone|myself)\b",
        r"\bkill (myself|someone)\b",
        r"\bmake a weapon\b",
        r"\bhow to (make|build) (a )?(bomb|explosive|weapon)\b",
        r"\b(synthesize|make|cook) (methamphetamine|meth|crystal meth|cocaine|heroin)\b",
        r"\b(home address|phone number|personal information) (of|for)\b",
    ]
]


class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None


def check_off_topic(
    top_similarity_score: float,
    strategy: str,
    threshold: float | None = None,
) -> GuardrailResult:
    """Reject a question whose best retrieved passage is too weak to answer it.

    `strategy` selects the measured threshold for the index that produced the
    score; `threshold` overrides it outright. An unknown strategy raises rather
    than falling back to a shared default -- the entire reason this table is
    keyed by strategy is that the scales are not interchangeable, so silently
    borrowing another index's threshold would reintroduce the original bug.
    """
    if threshold is None:
        if strategy not in _DEFAULT_OFFTOPIC_THRESHOLDS:
            raise ValueError(
                f"no measured off-topic threshold for strategy {strategy!r}; "
                f"measured strategies are "
                f"{sorted(_DEFAULT_OFFTOPIC_THRESHOLDS)}"
            )
        threshold = _DEFAULT_OFFTOPIC_THRESHOLDS[strategy]
    if top_similarity_score < threshold:
        return GuardrailResult(passed=False, reason="off-topic")
    return GuardrailResult(passed=True)


def check_unsafe_input(transcript: str) -> GuardrailResult:
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(transcript):
            return GuardrailResult(passed=False, reason="unsafe input")
    return GuardrailResult(passed=True)


def check_groundedness(
    answer: str,
    retrieval: RetrievalOutput,
    threshold: float = _DEFAULT_GROUNDEDNESS_THRESHOLD,
) -> GuardrailResult:
    context_text = " ".join(p.text for p in retrieval.passages)
    answer_vec = embed(answer)
    context_vec = embed(context_text)
    similarity = cosine_similarity(answer_vec, context_vec)
    if similarity < threshold:
        return GuardrailResult(passed=False, reason="ungrounded")
    return GuardrailResult(passed=True)
