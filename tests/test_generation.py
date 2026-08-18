from unittest.mock import patch

from app.generation import generate_answer
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
