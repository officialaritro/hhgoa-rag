from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval import _load_cached, retrieve


@pytest.fixture(autouse=True)
def _clear_index_cache():
    _load_cached.cache_clear()
    yield
    _load_cached.cache_clear()


def _fake_index_and_metadata():
    fake_index = MagicMock()
    fake_index.search.return_value = (
        np.array([[0.9, 0.4]]),  # distances (inner-product scores)
        np.array([[1, 0]]),  # indices into metadata
    )
    metadata = [
        {"text": "chunk zero", "source_passage": "passage zero", "is_selected": False},
        {"text": "chunk one", "source_passage": "passage one", "is_selected": True},
    ]
    return fake_index, metadata


@patch("app.retrieval.embed")
@patch("app.retrieval.load_index")
def test_retrieve_returns_passages_ranked_by_score(mock_load_index, mock_embed):
    mock_load_index.return_value = _fake_index_and_metadata()
    mock_embed.return_value = np.zeros(3)

    result = retrieve(query="what happened", strategy="fixed_size", k=2)

    assert result.query == "what happened"
    assert result.strategy == "fixed_size"
    assert len(result.passages) == 2
    assert result.passages[0].text == "chunk one"
    assert result.passages[0].score == 0.9
    assert result.passages[0].is_selected is True
    assert result.passages[1].text == "chunk zero"


@patch("app.retrieval.embed")
@patch("app.retrieval.load_index")
def test_retrieve_loads_the_index_matching_the_requested_strategy(
    mock_load_index, mock_embed
):
    mock_load_index.return_value = _fake_index_and_metadata()
    mock_embed.return_value = np.zeros(3)

    retrieve(query="q", strategy="semantic", k=1)

    call_args = mock_load_index.call_args
    assert "semantic" in str(call_args)


@patch("app.retrieval.embed")
@patch("app.retrieval.load_index")
def test_retrieve_loads_the_index_from_disk_only_once_per_strategy(
    mock_load_index, mock_embed
):
    mock_load_index.return_value = _fake_index_and_metadata()
    mock_embed.return_value = np.zeros(3)

    retrieve(query="first", strategy="fixed_size", k=1)
    retrieve(query="second", strategy="fixed_size", k=1)

    assert mock_load_index.call_count == 1
