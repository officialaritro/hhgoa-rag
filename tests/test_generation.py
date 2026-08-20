from unittest.mock import patch

from app.generation import INSUFFICIENT_CONTEXT, generate_answer
from app.schemas import RetrievalOutput, RetrievedPassage


def _retrieval(
    passages_text=("The Manhattan Project produced the first nuclear weapons.",),
):
    return RetrievalOutput(
        query="what did the manhattan project produce",
        strategy="fixed_size",
        passages=[
            RetrievedPassage(text=t, source_passage=t, is_selected=True, score=0.9)
            for t in passages_text
        ],
    )


@patch("app.generation._call_model")
def test_generate_answer_returns_ok_with_model_text(mock_call):
    mock_call.return_value = "The Manhattan Project produced the first nuclear weapons."

    result = generate_answer(
        query="what did the manhattan project produce", retrieval=_retrieval()
    )

    assert result.ok is True
    assert (
        result.value.answer
        == "The Manhattan Project produced the first nuclear weapons."
    )


@patch("app.generation._call_model")
def test_generate_answer_prompt_includes_retrieved_passages(mock_call):
    mock_call.return_value = "answer"

    generate_answer(query="q", retrieval=_retrieval(("passage about widgets",)))

    call_args = mock_call.call_args
    prompt_text = str(call_args)
    assert "passage about widgets" in prompt_text


@patch("app.generation._call_model")
def test_generate_answer_retries_once_then_succeeds(mock_call):
    mock_call.side_effect = [ConnectionError("dropped"), "recovered answer"]

    result = generate_answer(query="q", retrieval=_retrieval())

    assert mock_call.call_count == 2
    assert result.ok is True
    assert result.value.answer == "recovered answer"


@patch("app.generation._call_model")
def test_generate_answer_returns_error_result_on_persistent_failure(mock_call):
    mock_call.side_effect = ConnectionError("dropped")

    result = generate_answer(query="q", retrieval=_retrieval())

    assert mock_call.call_count == 2
    assert result.ok is False
    assert result.value is None
    assert "dropped" in result.error


@patch("app.generation._call_model")
def test_generate_answer_flags_the_insufficient_context_sentinel(mock_call):
    """A decline must be reported structurally, not left as prose.

    The groundedness guard cannot catch a decline: the model explains itself by
    quoting the passages, so its refusal scores 0.533-0.795 against the context
    and passes as grounded. Measured on the live indices -- that is why the
    model is asked for a sentinel instead of an explanation.
    """
    mock_call.return_value = INSUFFICIENT_CONTEXT

    result = generate_answer(query="q", retrieval=_retrieval())

    assert result.ok is True
    assert result.value.insufficient_context is True


@patch("app.generation._call_model")
def test_generate_answer_tolerates_decoration_around_the_sentinel(mock_call):
    """The contract is a bare sentinel, but a stray period or newline must not
    turn a refusal into an answer that reads as the literal string."""
    mock_call.return_value = f"  {INSUFFICIENT_CONTEXT}.\n"

    result = generate_answer(query="q", retrieval=_retrieval())

    assert result.value.insufficient_context is True


@patch("app.generation._call_model")
def test_generate_answer_does_not_flag_a_real_answer(mock_call):
    """False positives here refuse real questions -- the failure mode the
    off-topic threshold already shipped once. A passage that merely discusses
    missing information must not trip the check."""
    mock_call.return_value = (
        "The passages do not agree on the date, but the project ended in 1946."
    )

    result = generate_answer(query="q", retrieval=_retrieval())

    assert result.value.insufficient_context is False
    assert result.value.answer.startswith("The passages do not agree")
