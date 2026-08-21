"""Rule-based guardrails: off-topic, unsafe-input, and groundedness checks.

Plan Global Constraints: no second LLM call for guardrails -- every check
here is a threshold on a similarity score or a pattern match.

The similarity thresholds below are set from measured score distributions, not
guessed, and each carries the measurement that justifies it. Re-measure with
scripts/calibrate_thresholds.py before changing one, and check the change against
tests/test_guardrail_calibration.py, which pins the false-refusal budget.
"""

import json
import os
import re
import statistics
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from app.groundedness import sentence_support, unsupported_numbers
from app.schemas import RetrievalOutput


def _threshold(env_var: str, default: float) -> float:
    """Read a guardrail threshold from the environment, falling back to the
    documented default. These are meant to be retuned against measured score
    distributions (scripts/calibrate_thresholds.py), and retuning should not
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
#
# The thresholds themselves are no longer listed here. They are measured per
# index at build time and written into that index's manifest, because a
# hand-maintained dict is what produced the original defect: a value verified
# against the semantic index shipped as the fixed_size default. Two strategies
# made that a coin flip; eight would make it eight chances to forget one, and
# forgetting one is silent. See scripts/calibrate_thresholds.py and
# `offtopic_threshold` below.


class MissingCalibration(RuntimeError):
    """An index was built but never calibrated, and something tried to serve it.

    Raised rather than falling back to another strategy's threshold. The whole
    reason thresholds are per-strategy is that the score scales are not
    interchangeable -- semantic chunks are shorter, so every cosine runs higher
    (in-corpus median 0.779 against 0.741) -- so borrowing a number silently
    reintroduces the original bug.
    """


@cache
def offtopic_threshold(strategy: str, index_path: str | None = None) -> float:
    """The measured off-topic threshold for one strategy's index.

    Read from the manifest the build wrote, so a strategy cannot reach the
    serving path with an unmeasured threshold. An explicit environment variable
    still wins, which keeps retuning-without-a-rebuild available as documented.

    Cached: this is on the request path and must not stat and parse JSON per
    query.
    """
    override = os.environ.get(
        f"OFFTOPIC_SIMILARITY_THRESHOLD_{strategy.upper()}", ""
    ).strip()
    if override:
        try:
            return float(override)
        except ValueError:
            pass

    if index_path is None:
        from app.strategies import UnknownStrategy, chunk_paths, get

        # A composed strategy (hybrid, fusion) has no index and so no manifest.
        # It reports a dense cosine taken from its member indices -- ranking
        # comes from RRF, the score does not -- so the primary member's measured
        # threshold is the correct gate. Without this the guard raises on every
        # composed request.
        try:
            spec = get(strategy)
        except UnknownStrategy:
            spec = None
        if spec is not None and spec.members:
            return offtopic_threshold(spec.members[0])
        index_path = chunk_paths(strategy)[0]
    manifest = Path(index_path).with_suffix(".manifest.json")
    if not manifest.exists():
        raise MissingCalibration(
            f"no manifest for strategy {strategy!r} at {manifest}; "
            f"build and calibrate it before serving "
            f"(python -m scripts.build_all && python -m scripts.calibrate_thresholds)"
        )
    recorded = json.loads(manifest.read_text()).get("offtopic_threshold")
    if recorded is None:
        raise MissingCalibration(
            f"index for strategy {strategy!r} has no measured off-topic threshold. "
            f"Serving it would have to borrow another index's number, which is the "
            f"defect this replaces. Run: python -m scripts.calibrate_thresholds"
        )
    return float(recorded)


# Now a threshold on the MEAN per-sentence support (app/groundedness.py), not on
# a whole-answer cosine against concatenated context. The old metric truncated:
# MiniLM caps at 256 tokens and 94% of real k=5 contexts exceed that, so the
# guard was silently scoring against a shortened context on almost every request.
#
# Re-measured against 40 real generated answers after the fix: grounded min
# 0.524, ungrounded max 0.271, a clean gap whose midpoint is 0.397. The shipped
# 0.40 happens to sit almost exactly there, so the number is unchanged while the
# thing it measures is not.
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
        # Credential requests. Deliberately anchored on "what is my <secret>"
        # rather than a bare mention of "password": the corpus is web-search
        # queries, so legitimate questions like "how do I change my bank
        # password" must still be answered. Measured over all 10,000 corpus
        # queries, this pattern matches none of them.
        #
        # This is also the one off-topic query no similarity threshold can
        # reject -- an MS MARCO corpus genuinely contains bank-and-password
        # passages, so it scores 0.612/0.743 and clears the off-topic gate.
        # Catching it here refuses it pre-retrieval instead, for free.
        r"\bwhat (is|are) my\b.*\b(password|passcode|pin|ssn|social security number)\b",
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
        threshold = offtopic_threshold(strategy)
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
    support = sentence_support(answer, retrieval)
    if not support:
        return GuardrailResult(passed=False, reason="ungrounded")

    # Literal check first: it is exact, costs no model call, and catches the
    # failure the semantic score cannot. One fabricated figure among four sound
    # sentences barely moves an average, but "1987" is simply not in the text.
    fabricated = unsupported_numbers(
        answer, " ".join(p.text for p in retrieval.passages)
    )
    if fabricated:
        return GuardrailResult(
            passed=False, reason=f"ungrounded: figures not in context {fabricated}"
        )

    if statistics.mean(support) < threshold:
        return GuardrailResult(passed=False, reason="ungrounded")
    return GuardrailResult(passed=True)
