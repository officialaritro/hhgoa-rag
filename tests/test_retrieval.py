"""Retrieval over span-addressed chunks.

Two behaviours here are correctness fixes rather than features:

  * dedup by parent. With 20% overlap, several of the top-k slots could be
    near-identical text from one passage, silently shrinking the distinct
    context handed to generation. Retrieval over-fetches and collapses.
  * `text` is the strategy's *return* text, not the embedded text. For
    parent_child and sentence_window those differ deliberately, and returning
    the embedded span instead would erase the difference between those
    strategies and plain fixed-size chunking.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval import _load_cached, _load_passages, retrieve
from app.strategies import UnknownStrategy


@pytest.fixture(autouse=True)
def _clear_caches():
    _load_cached.cache_clear()
    _load_passages.cache_clear()
    yield
    _load_cached.cache_clear()
    _load_passages.cache_clear()


@pytest.fixture(autouse=True)
def _no_reranker(request, monkeypatch):
    """Reranking is on by default in production, so an unmocked retrieve() here
    would download and load a 90MB cross-encoder -- turning unit tests into
    network-dependent integration tests and tripling the suite runtime.

    Tests that are specifically about reranking opt back in by name.
    """
    if "rerank" in request.node.name:
        return
    monkeypatch.setattr("app.retrieval.reranking_enabled", lambda: False)


PASSAGES = [
    {
        "text": "Alpha sentence. Beta sentence. Gamma sentence.",
        "is_selected": True,
        "query_id": 1,
        "query": "what is alpha",
    },
    {
        "text": "An unrelated passage about turbines.",
        "is_selected": False,
        "query_id": 2,
        "query": "what is a turbine",
    },
]


def _index_returning(order, scores=None):
    """A fake FAISS index returning the given row ids in the given order."""
    index = MagicMock()
    scores = scores or [0.9 - 0.1 * i for i in range(len(order))]
    index.search.return_value = (np.array([scores]), np.array([order]))
    return index


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_returns_passages_ranked_by_score(mock_store, mock_load, mock_embed):
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    mock_load.return_value = (_index_returning([1, 0], [0.9, 0.4]), rows)

    result = retrieve(query="what happened", strategy="whole_passage", k=2)

    assert result.query == "what happened"
    assert result.strategy == "whole_passage"
    assert [p.score for p in result.passages] == [0.9, 0.4]
    assert result.passages[0].text == "An unrelated passage about turbines."
    assert result.passages[1].text == "Alpha sentence."


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_collapses_multiple_chunks_from_the_same_parent(
    mock_store, mock_load, mock_embed
):
    """The correctness fix. Three overlapping chunks of passage 0 must not
    occupy three of the caller's slots."""
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 0, "start": 16, "end": 30},
        {"parent_id": 0, "start": 31, "end": 45},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    mock_load.return_value = (_index_returning([0, 1, 2, 3]), rows)

    result = retrieve(query="q", strategy="fixed_size", k=2)

    assert len(result.passages) == 2
    assert [p.source_passage for p in result.passages] == [
        PASSAGES[0]["text"],
        PASSAGES[1]["text"],
    ]


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_keeps_the_best_scoring_chunk_of_a_collapsed_parent(
    mock_store, mock_load, mock_embed
):
    rows = [
        {"parent_id": 0, "start": 16, "end": 30},
        {"parent_id": 0, "start": 0, "end": 15},
    ]
    mock_load.return_value = (_index_returning([0, 1], [0.95, 0.5]), rows)

    result = retrieve(query="q", strategy="fixed_size", k=5)

    assert len(result.passages) == 1
    assert result.passages[0].score == 0.95
    assert result.passages[0].text == "Beta sentence."


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_over_fetches_so_dedup_can_still_fill_k(mock_store, mock_load, mock_embed):
    """Searching for exactly k and then deduping would return fewer than k
    whenever any two hits share a parent."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    index = _index_returning([0])
    mock_load.return_value = (index, rows)

    retrieve(query="q", strategy="fixed_size", k=5)

    requested = index.search.call_args.args[1]
    assert requested > 5, f"searched for only {requested}, leaving no room to dedup"


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_text_is_the_return_span_and_source_passage_is_the_parent(
    mock_store, mock_load, mock_embed
):
    """sentence_window embeds one sentence and returns a wider window. `text`
    must carry the window; `source_passage` must stay the exact parent so
    evaluation can match it against the corpus relevance labels."""
    # ret_end from len(), not hand-counted: the passage is 46 chars and an
    # off-by-one here silently clips the last character of every window.
    rows = [
        {
            "parent_id": 0,
            "start": 16,
            "end": 30,
            "ret_start": 0,
            "ret_end": len(PASSAGES[0]["text"]),
        }
    ]
    mock_load.return_value = (_index_returning([0]), rows)

    result = retrieve(query="q", strategy="sentence_window", k=1)

    assert result.passages[0].text == PASSAGES[0]["text"]
    assert result.passages[0].source_passage == PASSAGES[0]["text"]


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_stored_text_chunks_report_their_own_text_as_the_source(
    mock_store, mock_load, mock_embed
):
    """query_group chunks span several passages, so no single parent is their
    source. Reporting a nominal parent would understate them in evaluation."""
    rows = [{"parent_id": 0, "start": 0, "end": 0, "text": "passage one. passage two."}]
    mock_load.return_value = (_index_returning([0]), rows)

    result = retrieve(query="q", strategy="query_group", k=1)

    assert result.passages[0].text == "passage one. passage two."
    assert result.passages[0].source_passage == "passage one. passage two."


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_is_selected_comes_from_the_parent_passage(mock_store, mock_load, mock_embed):
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    mock_load.return_value = (_index_returning([0, 1]), rows)

    result = retrieve(query="q", strategy="whole_passage", k=2)

    assert [p.is_selected for p in result.passages] == [True, False]


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_ignores_the_empty_slots_faiss_pads_with(mock_store, mock_load, mock_embed):
    """FAISS returns -1 for rows it could not fill. Indexing metadata with -1
    silently returns the last chunk in the store as a match."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0, -1, -1], [0.9, -1.0, -1.0]), rows)

    result = retrieve(query="q", strategy="whole_passage", k=3)

    assert len(result.passages) == 1


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_loads_each_index_from_disk_only_once(mock_store, mock_load, mock_embed):
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    retrieve(query="first", strategy="fixed_size", k=1)
    retrieve(query="second", strategy="fixed_size", k=1)

    assert mock_load.call_count == 1


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_loads_the_index_matching_the_requested_strategy(
    mock_store, mock_load, mock_embed
):
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    retrieve(query="q", strategy="query_aware", k=1)

    assert "query_aware" in str(mock_load.call_args)


def test_unknown_strategy_raises_the_registry_error():
    with pytest.raises(UnknownStrategy):
        retrieve(query="q", strategy="not_a_strategy", k=1)


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_every_registered_strategy_is_retrievable(mock_store, mock_load, mock_embed):
    """The coupling that matters at request time: a strategy in the registry
    with no path or no dispatch branch fails per request, not at startup."""
    from app.strategies import dense_names

    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    for name in dense_names():
        result = retrieve(query="q", strategy=name, k=1)
        assert result.strategy == name
        assert len(result.passages) == 1


# ------------------------------------------------------ composed strategies


class _FakeBM25:
    """Stands in for the real BM25 index; returns (passage_id, score) pairs."""

    def __init__(self, ranked):
        self._ranked = ranked

    def top_k(self, query, k):
        return self._ranked[:k]


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
@patch("app.retrieval._load_bm25")
def test_hybrid_merges_lexical_and_dense_rankings(
    mock_bm25, mock_store, mock_load, mock_embed
):
    """A passage both halves rank highly must beat one only the dense half
    likes. That agreement is the entire reason to run two retrievers."""
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    # dense prefers passage 0; lexical prefers passage 1 strongly and also has 0
    mock_load.return_value = (_index_returning([0, 1], [0.9, 0.5]), rows)
    mock_bm25.return_value = _FakeBM25([(1, 22.0), (0, 3.0)])

    result = retrieve(query="q", strategy="hybrid", k=2)

    assert len(result.passages) == 2
    assert {p.source_passage for p in result.passages} == {
        PASSAGES[0]["text"],
        PASSAGES[1]["text"],
    }


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
@patch("app.retrieval._load_bm25")
def test_hybrid_reports_a_cosine_score_not_a_fusion_score(
    mock_bm25, mock_store, mock_load, mock_embed
):
    """RRF scores are around 0.03 and have nothing to do with cosine. The
    off-topic guard's thresholds are calibrated on cosines (0.499-0.574), so
    reporting a fusion score would refuse every query. Ranking comes from RRF;
    the reported score stays a dense similarity."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0], [0.87]), rows)
    mock_bm25.return_value = _FakeBM25([(0, 19.0)])

    result = retrieve(query="q", strategy="hybrid", k=1)

    assert result.passages[0].score == pytest.approx(0.87)


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
@patch("app.retrieval._load_bm25")
def test_hybrid_still_answers_when_no_keyword_matches(
    mock_bm25, mock_store, mock_load, mock_embed
):
    """A lexical miss must abstain, not veto. Voice transcripts routinely
    contain none of a passage's exact terms."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0], [0.8]), rows)
    mock_bm25.return_value = _FakeBM25([])

    result = retrieve(query="q", strategy="hybrid", k=1)

    assert len(result.passages) == 1


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_fusion_merges_every_member_strategy(mock_store, mock_load, mock_embed):
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    mock_load.return_value = (_index_returning([1, 0], [0.9, 0.6]), rows)

    result = retrieve(query="q", strategy="fusion", k=2)

    assert len(result.passages) == 2
    # one load per member index, not one per request
    from app.strategies import get

    assert mock_load.call_count == len(get("fusion").members)


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_fusion_embeds_the_query_only_once(mock_store, mock_load, mock_embed):
    """Embedding is 9.9ms P50 and strategy-independent; doing it per member
    would multiply the dominant cost by the number of members."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    retrieve(query="q", strategy="fusion", k=1)

    assert mock_embed.call_count == 1


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
def test_composed_strategies_return_the_parent_passage_as_text(
    mock_store, mock_load, mock_embed
):
    """Fusion merges at parent granularity, so a fused hit has no single chunk
    to speak for it."""
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    result = retrieve(query="q", strategy="fusion", k=1)

    assert result.passages[0].text == PASSAGES[0]["text"]
    assert result.passages[0].source_passage == PASSAGES[0]["text"]


# ----------------------------------------------------------------- reranking


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
@patch("app.retrieval.rerank_passages")
def test_dense_retrieval_reranks_before_truncating_to_k(
    mock_rerank, mock_store, mock_load, mock_embed
):
    """The reranker must see more candidates than the caller asked for. Handing
    it only k would leave it nothing to promote -- the measured gain comes from
    reordering a deeper pool."""
    rows = [
        {"parent_id": 0, "start": 0, "end": 15},
        {"parent_id": 1, "start": 0, "end": 36},
    ]
    mock_load.return_value = (_index_returning([0, 1]), rows)
    mock_rerank.side_effect = lambda q, ps, top_k: ps[:top_k]

    retrieve(query="q", strategy="whole_passage", k=1)

    handed_to_reranker = mock_rerank.call_args.args[1]
    assert len(handed_to_reranker) > 1, "reranker received only the final k"
    assert mock_rerank.call_args.kwargs["top_k"] == 1


@patch("app.retrieval.embed", return_value=np.zeros(3))
@patch("app.retrieval.load_index")
@patch("app.retrieval.load_passage_store", return_value=PASSAGES)
@patch("app.retrieval.reranking_enabled", return_value=False)
@patch("app.retrieval.rerank_passages")
def test_reranking_can_be_switched_off(
    mock_rerank, mock_enabled, mock_store, mock_load, mock_embed
):
    rows = [{"parent_id": 0, "start": 0, "end": 15}]
    mock_load.return_value = (_index_returning([0]), rows)

    result = retrieve(query="q", strategy="whole_passage", k=1)

    mock_rerank.assert_not_called()
    assert len(result.passages) == 1
