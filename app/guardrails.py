"""Rule-based guardrails: off-topic, unsafe-input, and groundedness checks.

Plan Global Constraints: no second LLM call for guardrails -- every check
here is a threshold on a similarity score or a pattern match. Thresholds are
literal constants below with a one-line note; Task 9's benchmark run is the
first real signal on whether they need retuning (plan Key Decisions, Task 7).
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


# Starting values, chosen before the corpus existed. scripts/tune_thresholds.py
# measures the distributions these are supposed to separate and recommends
# replacements; override via the env vars rather than editing these.
_DEFAULT_OFFTOPIC_THRESHOLD = _threshold("OFFTOPIC_SIMILARITY_THRESHOLD", 0.3)
_DEFAULT_GROUNDEDNESS_THRESHOLD = _threshold("GROUNDEDNESS_SIMILARITY_THRESHOLD", 0.5)

# Chosen empirically as an illustrative starting set for the MVP guardrail,
# not an exhaustive safety classifier. Env-overridable is left for Task 9
# retuning; the list itself is a plain module constant since it is not a
# single scalar.
_UNSAFE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bbomb\b",
        r"\bhurt (people|someone|myself)\b",
        r"\bkill (myself|someone)\b",
        r"\bmake a weapon\b",
        r"\bhow to (make|build) (a )?(bomb|explosive|weapon)\b",
    ]
]


class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None


def check_off_topic(
    top_similarity_score: float, threshold: float = _DEFAULT_OFFTOPIC_THRESHOLD
) -> GuardrailResult:
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
