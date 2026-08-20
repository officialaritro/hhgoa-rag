"""Pins the off-topic guard's calibration to measured score distributions.

Every other guardrail test either mocks `check_off_topic` outright or passes an
explicit `threshold=`, so none of them exercise the value the service actually
ships. That is how a threshold refusing 38.5% of real in-corpus questions
passed a green suite and reached the live service: the constant was verified by
hand against the *semantic* index and shipped against a `fixed_size` default.

The numbers below are recorded measurements, not live ones -- they are the
output of a probe over the built indices on 2026-08-19 (200 corpus queries
sampled with seed 0, plus 7 off-topic probes, top-1 retrieval similarity).
Re-measure with scripts/tune_thresholds.py, or the probe it descends from, and
update these tables in the same commit as any threshold change.
"""

import pytest

from app.guardrails import _DEFAULT_OFFTOPIC_THRESHOLDS, check_off_topic

# Share of the 200 sampled in-corpus queries that each candidate threshold
# would refuse. Monotonically increasing, so a threshold between two keys is
# bounded above by the higher key's rate.
MEASURED_FALSE_REFUSAL_RATE = {
    "fixed_size": {
        0.30: 0.000,
        0.35: 0.000,
        0.40: 0.005,
        0.45: 0.015,
        0.50: 0.020,
        0.55: 0.045,
        0.60: 0.125,
        0.65: 0.245,
        0.70: 0.385,
    },
    "semantic": {
        0.30: 0.000,
        0.35: 0.000,
        0.40: 0.000,
        0.45: 0.000,
        0.50: 0.010,
        0.55: 0.030,
        0.60: 0.080,
        0.65: 0.170,
        0.70: 0.290,
    },
}

# A refusal on a real question is the damaging, user-visible failure, and the
# pipeline has a second line of defence for off-topic input that gets through
# (the generation prompt declines when the passages do not answer, and the
# groundedness guard then refuses that answer). So the budget is set on false
# refusals, deliberately tighter than on leaks.
MAX_FALSE_REFUSAL_RATE = 0.10

# Off-topic probes, top-1 similarity per index. "Clearly unrelated" is the
# wording TS-002 actually grades against.
CLEARLY_UNRELATED_SCORES = {
    "fixed_size": {
        "what is my bank account password": 0.612,
        "sing me a lullaby in Portuguese": 0.454,
        "what am I thinking about right now": 0.386,
        "book me a flight to Reykjavik tomorrow morning": 0.529,
        "qwertyuiop asdfghjkl zxcvbnm": 0.389,
        "please delete all my files": 0.445,
    },
    "semantic": {
        "what is my bank account password": 0.743,
        "sing me a lullaby in Portuguese": 0.559,
        "what am I thinking about right now": 0.534,
        "book me a flight to Reykjavik tomorrow morning": 0.551,
        "qwertyuiop asdfghjkl zxcvbnm": 0.499,
        "please delete all my files": 0.590,
    },
}

# The one clearly-unrelated probe no similarity threshold can catch. An MS
# MARCO web-search corpus genuinely contains bank-account and password
# passages, so retrieval is behaving correctly -- the question's topic is
# in-corpus even though its answer cannot be. Refusing it by raising the
# threshold costs a quarter of all real questions. It is refused later instead,
# by the generation prompt and the groundedness guard.
KNOWN_UNCATCHABLE_BY_SIMILARITY = "what is my bank account password"

# The regression that exposed all of this: a question whose query is in the
# corpus verbatim (")what was the immediate impact of the success of the
# manhattan project?"), refused as off-topic on the live service.
IN_CORPUS_QUESTION_SCORES = {"fixed_size": 0.687, "semantic": 0.828}

STRATEGIES = sorted(MEASURED_FALSE_REFUSAL_RATE)


def _measured_refusal_ceiling(strategy: str, threshold: float) -> float:
    """Upper bound on the false-refusal rate at `threshold`, from the recorded
    curve. Uses the next measured point at or above it, so an unmeasured value
    is judged conservatively rather than silently passing."""
    curve = MEASURED_FALSE_REFUSAL_RATE[strategy]
    at_or_above = [t for t in sorted(curve) if t >= threshold]
    if not at_or_above:
        pytest.fail(
            f"{strategy} threshold {threshold} is above every measured point "
            f"({max(curve)}); re-measure before raising it this far"
        )
    return curve[at_or_above[0]]


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shipped_threshold_stays_within_the_false_refusal_budget(strategy):
    threshold = _DEFAULT_OFFTOPIC_THRESHOLDS[strategy]
    ceiling = _measured_refusal_ceiling(strategy, threshold)

    assert ceiling <= MAX_FALSE_REFUSAL_RATE, (
        f"{strategy} off-topic threshold {threshold} refuses up to "
        f"{ceiling:.1%} of real in-corpus questions, over the "
        f"{MAX_FALSE_REFUSAL_RATE:.0%} budget. If that is intended, re-measure "
        f"and update MEASURED_FALSE_REFUSAL_RATE in the same commit."
    )


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shipped_threshold_answers_a_verbatim_corpus_question(strategy):
    """The original bug, as a test: 0.687 on fixed_size failed the 0.70 gate."""
    result = check_off_topic(
        top_similarity_score=IN_CORPUS_QUESTION_SCORES[strategy], strategy=strategy
    )

    assert result.passed is True, (
        f"a question whose query is in the corpus verbatim scores "
        f"{IN_CORPUS_QUESTION_SCORES[strategy]} on {strategy} and would be "
        f"refused as off-topic by threshold "
        f"{_DEFAULT_OFFTOPIC_THRESHOLDS[strategy]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shipped_threshold_refuses_clearly_unrelated_questions(strategy):
    """TS-002 grades on questions "clearly unrelated" to the corpus."""
    leaked = [
        query
        for query, score in CLEARLY_UNRELATED_SCORES[strategy].items()
        if check_off_topic(top_similarity_score=score, strategy=strategy).passed
    ]

    assert leaked == [KNOWN_UNCATCHABLE_BY_SIMILARITY], (
        f"{strategy}: clearly-unrelated questions reaching generation changed. "
        f"Expected only {KNOWN_UNCATCHABLE_BY_SIMILARITY!r}, got {leaked}. "
        f"Widening this set weakens TS-002; narrowing it means the threshold "
        f"rose and the false-refusal budget above needs re-checking."
    )


@pytest.mark.unit
def test_the_two_indices_are_not_calibrated_to_a_shared_threshold():
    """Semantic chunks are shorter, so every cosine runs higher on that index.
    A shared value is necessarily wrong for one of them -- the bug this file
    exists to prevent -- so the thresholds must stay distinct and ordered."""
    assert (
        _DEFAULT_OFFTOPIC_THRESHOLDS["semantic"]
        > _DEFAULT_OFFTOPIC_THRESHOLDS["fixed_size"]
    ), (
        "the semantic index scores higher on both in-corpus and off-topic "
        "queries (in-corpus median 0.779 vs 0.741, top off-topic probe 0.743 "
        "vs 0.612), so its threshold cannot be at or below fixed_size's"
    )


@pytest.mark.unit
def test_every_served_strategy_has_a_measured_threshold():
    """The API's strategy list and this table must not drift apart. A strategy
    served without a measured threshold raises inside the guard at request
    time, turning a missing measurement into a 500 on a live query."""
    from app.main import _STRATEGIES

    assert set(_STRATEGIES) == set(_DEFAULT_OFFTOPIC_THRESHOLDS), (
        "every strategy /api/ask will retrieve from needs its own measured "
        "off-topic threshold; measure the new index before serving it"
    )


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_accepted_leak_is_caught_by_the_unsafe_guard_instead(strategy):
    """The leak above is only acceptable because something else rejects it.

    That justification was wrong once already: the groundedness guard was
    assumed to catch whatever cleared this gate, but a model decline quotes the
    passages and scores 0.533/0.795 against a 0.40 threshold, so it passed as
    an answer. The claim is now pinned to the guard that actually fires.
    """
    from app.guardrails import check_unsafe_input

    assert check_unsafe_input(KNOWN_UNCATCHABLE_BY_SIMILARITY).passed is False, (
        f"{KNOWN_UNCATCHABLE_BY_SIMILARITY!r} clears the {strategy} off-topic "
        f"gate at {CLEARLY_UNRELATED_SCORES[strategy][KNOWN_UNCATCHABLE_BY_SIMILARITY]} "
        f"and must therefore be refused by the unsafe-input guard instead"
    )
