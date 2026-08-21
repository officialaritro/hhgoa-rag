"""Claim-level groundedness.

The guard this replaces embedded the answer against all five retrieved passages
*concatenated into one string*. MiniLM-L6-v2 truncates at 256 tokens and the
median k=5 context measures 334 tokens, so **94% of requests silently compared
the answer against only part of the context** -- up to two thirds discarded in
the worst case observed. That is a correctness bug, and it explains the measured
discrimination (AUC ~0.80, with 76 of 80 ungrounded answers scoring inside the
grounded range) far better than the threshold did.

Passages are now scored individually, so nothing is truncated: a single passage
averages 333 characters, roughly 80 tokens. And support is computed per answer
sentence rather than for the answer as a whole, because a four-sentence answer
with one fabricated claim averages out to looking grounded.

A literal check runs alongside the semantic one. A hallucinated number is a
string mismatch, not a cosine drift -- "founded in 1897" against a context
saying 1987 scores essentially the same as the truth.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.groundedness import (
    sentence_support,
    unsupported_numbers,
)
from app.schemas import RetrievalOutput, RetrievedPassage


def _retrieval(*texts):
    return RetrievalOutput(
        query="q",
        strategy="whole_passage",
        passages=[
            RetrievedPassage(text=t, source_passage=t, is_selected=False, score=0.9)
            for t in texts
        ],
    )


# ------------------------------------------------------------- literal numbers


def test_a_number_present_in_the_context_is_not_flagged():
    assert (
        unsupported_numbers("It was founded in 1897.", "Founded in 1897 nearby.") == []
    )


def test_a_fabricated_number_is_flagged():
    """The failure cosine cannot see: 1987 against 1897 is one digit apart and
    semantically identical to an embedding model."""
    assert unsupported_numbers("It was founded in 1987.", "Founded in 1897.") == [
        "1987"
    ]


def test_numbers_are_matched_ignoring_thousands_separators():
    """ "1,234" in an answer and "1234" in a passage are the same number, and
    flagging that as a hallucination would refuse correct answers."""
    assert unsupported_numbers("about 1,234 metres", "exactly 1234 metres") == []


def test_percentages_and_decimals_are_compared_as_written():
    assert unsupported_numbers("around 12.5 percent", "some 12.5 percent") == []
    assert unsupported_numbers("around 13.5 percent", "some 12.5 percent") == ["13.5"]


def test_an_answer_with_no_numbers_flags_nothing():
    assert unsupported_numbers("Dehydration causes thirst.", "Thirst is a sign.") == []


def test_small_integers_that_are_ordinary_words_are_not_flagged():
    """Enumerations the model writes itself ("1." or "one of three") are not
    factual claims lifted from context, and flagging them would refuse most
    list-shaped answers."""
    assert (
        unsupported_numbers(
            "There are 3 signs: thirst, fatigue, dizziness.", "Signs include thirst."
        )
        == []
    )


# --------------------------------------------------------- per-sentence support


@patch("app.groundedness.embed_batch")
def test_support_is_computed_per_answer_sentence(mock_embed):
    """Three sentences in, three support scores out. A single answer-level score
    lets one fabricated claim hide behind three good ones."""
    mock_embed.side_effect = lambda texts, **kw: np.eye(8, dtype="float32")[
        : len(texts)
    ]

    support = sentence_support("One. Two. Three.", _retrieval("a", "b"))

    assert len(support) == 3


@patch("app.groundedness.embed_batch")
def test_each_passage_is_embedded_separately_never_concatenated(mock_embed):
    """The bug being fixed. Concatenating five passages exceeds the model's
    256-token limit for 94% of real contexts, silently dropping context."""
    mock_embed.side_effect = lambda texts, **kw: np.eye(8, dtype="float32")[
        : len(texts)
    ]

    sentence_support(
        "A claim.", _retrieval("passage one", "passage two", "passage three")
    )

    embedded = [t for call in mock_embed.call_args_list for t in call.args[0]]
    assert "passage one" in embedded
    assert "passage two" in embedded
    assert "passage three" in embedded
    assert not any("passage one" in t and "passage two" in t for t in embedded), (
        "passages were concatenated; that is what truncates"
    )


@patch("app.groundedness.embed_batch")
def test_support_takes_the_best_matching_passage_not_the_average(mock_embed):
    """A claim supported by one passage is grounded even if the other four are
    about something else -- averaging would punish a correct citation."""

    def fake(texts, **kw):
        vectors = {
            "The claim.": [1.0, 0.0],
            "exactly the claim": [1.0, 0.0],
            "unrelated turbines": [0.0, 1.0],
        }
        return np.array([vectors[t] for t in texts], dtype="float32")

    mock_embed.side_effect = fake

    support = sentence_support(
        "The claim.", _retrieval("exactly the claim", "unrelated turbines")
    )

    assert support[0] == pytest.approx(1.0)


@patch("app.groundedness.embed_batch")
def test_an_empty_answer_has_no_support_rather_than_raising(mock_embed):
    mock_embed.side_effect = lambda texts, **kw: np.eye(4, dtype="float32")[
        : len(texts)
    ]

    assert sentence_support("", _retrieval("a")) == []


@patch("app.groundedness.embed_batch")
def test_no_retrieved_passages_means_zero_support(mock_embed):
    """A refusal path, not a crash: if retrieval returned nothing there is
    nothing for the answer to be grounded in."""
    mock_embed.side_effect = lambda texts, **kw: np.eye(4, dtype="float32")[
        : len(texts)
    ]

    support = sentence_support("A claim.", _retrieval())

    assert support == [0.0]


@patch("app.groundedness.embed_batch")
def test_embeds_everything_in_a_single_batched_call(mock_embed):
    """Latency. The guard runs on every request and was 110ms of a 130ms budget.
    Sentences and passages go in one call: splitting them pays the model's
    per-call overhead twice for about a dozen short texts and gains nothing."""
    mock_embed.side_effect = lambda texts, **kw: np.eye(16, dtype="float32")[
        : len(texts)
    ]

    sentence_support("One. Two. Three.", _retrieval("a", "b", "c", "d", "e"))

    assert mock_embed.call_count == 1
