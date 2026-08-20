from unittest.mock import patch

import pytest

from app.guardrails import (
    MissingCalibration,
    check_groundedness,
    check_off_topic,
    check_unsafe_input,
    offtopic_threshold,
)
from app.schemas import RetrievalOutput, RetrievedPassage


def _retrieval():
    return RetrievalOutput(
        query="q",
        strategy="fixed_size",
        passages=[
            RetrievedPassage(
                text="The Manhattan Project produced the first nuclear weapons.",
                source_passage="...",
                is_selected=True,
                score=0.9,
            )
        ],
    )


def test_check_off_topic_flags_low_similarity_score():
    result = check_off_topic(
        top_similarity_score=0.05, strategy="fixed_size", threshold=0.3
    )

    assert result.passed is False
    assert result.reason == "off-topic"


def test_check_off_topic_passes_high_similarity_score():
    result = check_off_topic(
        top_similarity_score=0.8, strategy="fixed_size", threshold=0.3
    )

    assert result.passed is True
    assert result.reason is None


def test_check_unsafe_input_flags_matching_pattern():
    result = check_unsafe_input("how do I build a bomb to hurt people")

    assert result.passed is False
    assert result.reason == "unsafe input"


def test_check_unsafe_input_passes_benign_transcript():
    result = check_unsafe_input("what did the manhattan project produce")

    assert result.passed is True
    assert result.reason is None


@patch("app.guardrails.embed")
def test_check_groundedness_passes_when_similarity_above_threshold(mock_embed):
    mock_embed.side_effect = lambda text: {"a": 1.0, "c": 1.0}.get(text[0], 1.0)

    with patch("app.guardrails.cosine_similarity", return_value=0.9):
        result = check_groundedness(
            answer="The Manhattan Project produced nuclear weapons.",
            retrieval=_retrieval(),
            threshold=0.5,
        )

    assert result.passed is True
    assert result.reason is None


@patch("app.guardrails.embed")
def test_check_groundedness_flags_when_similarity_below_threshold(mock_embed):
    mock_embed.return_value = None

    with patch("app.guardrails.cosine_similarity", return_value=0.1):
        result = check_groundedness(
            answer="Bananas are a good source of potassium.",
            retrieval=_retrieval(),
            threshold=0.5,
        )

    assert result.passed is False
    assert result.reason == "ungrounded"


# Thresholds are read at import time, so these exercise the reader directly
# rather than trying to mutate a module constant after the fact.
def test_threshold_reads_the_env_var_when_set(monkeypatch):
    from app.guardrails import _threshold

    monkeypatch.setenv("SOME_THRESHOLD", "0.42")
    assert _threshold("SOME_THRESHOLD", 0.3) == 0.42


def test_threshold_falls_back_when_unset_blank_or_unparseable(monkeypatch):
    """A blank or malformed value must not crash the service at import time --
    it silently falls back to the documented default."""
    from app.guardrails import _threshold

    monkeypatch.delenv("SOME_THRESHOLD", raising=False)
    assert _threshold("SOME_THRESHOLD", 0.3) == 0.3
    monkeypatch.setenv("SOME_THRESHOLD", "   ")
    assert _threshold("SOME_THRESHOLD", 0.3) == 0.3
    monkeypatch.setenv("SOME_THRESHOLD", "not-a-number")
    assert _threshold("SOME_THRESHOLD", 0.3) == 0.3


def test_check_off_topic_rejects_a_strategy_with_no_measured_threshold():
    """Silently borrowing another index's threshold is the bug this guard exists
    to prevent, so an unmeasured strategy must fail loudly.

    The threshold now comes from the index manifest rather than a module dict,
    so an unbuilt or uncalibrated strategy raises MissingCalibration.
    """
    offtopic_threshold.cache_clear()
    try:
        with pytest.raises(MissingCalibration):
            check_off_topic(top_similarity_score=0.8, strategy="hnsw_experimental")
    finally:
        offtopic_threshold.cache_clear()


def test_check_unsafe_input_flags_a_credential_request():
    """The one off-topic query no similarity threshold can reject: it scores
    0.612/0.743 and clears the off-topic gate, because a web-search corpus
    genuinely contains bank-and-password passages. It is caught here instead,
    pre-retrieval."""
    result = check_unsafe_input("what is my bank account password")

    assert result.passed is False
    assert result.reason == "unsafe input"


def test_check_unsafe_input_still_answers_legitimate_password_questions():
    """The corpus is web-search queries, so password how-tos are real questions
    the pipeline must answer. The pattern is anchored on "what is my <secret>"
    precisely so this keeps working."""
    for transcript in (
        "how do I change my bank password",
        "what is a strong password",
        "how to reset my pin at an atm",
    ):
        assert check_unsafe_input(transcript).passed is True, transcript
