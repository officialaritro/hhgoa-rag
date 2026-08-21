"""Cross-encoder reranking.

Measured on 500 labelled queries against dense order over whole passages:

    recall@1   0.410 -> 0.504   (+23%)
    recall@5   0.848 -> 0.916   (+8%)
    MRR@10     0.591 -> 0.671   (+14%)

That is a larger gain than every chunking strategy and both fusion modes
combined -- `fusion` reached 0.854 recall@5, inside the ~1.6pp error bar of
plain dense retrieval. The reason is in the baseline: recall@10 was already
0.960, so the relevant passage was nearly always among the candidates and simply
not ranked first. A bi-encoder cannot fix that, because it embeds query and
passage separately and never sees them together.

Depth 10, measured: 0.504 at depth 10 against 0.500 at both 20 and 50, so deeper
candidate pools add noise rather than signal, and depth 10 is also the cheapest
(36ms CPU against 66ms and 164ms).
"""

from unittest.mock import MagicMock, patch

import pytest

from app.reranking import RERANK_DEPTH, rerank_passages, reranking_enabled
from app.schemas import RetrievedPassage


def _passages(*specs):
    return [
        RetrievedPassage(text=t, source_passage=t, is_selected=False, score=s)
        for t, s in specs
    ]


def _encoder(scores):
    encoder = MagicMock()
    encoder.predict.return_value = scores
    return encoder


@patch("app.reranking._load_encoder")
def test_reorders_by_cross_encoder_score(mock_load):
    """The whole point: the dense order is wrong 59% of the time at rank 1."""
    mock_load.return_value = _encoder([0.1, 0.9, 0.4])
    passages = _passages(("first", 0.88), ("second", 0.80), ("third", 0.75))

    result = rerank_passages("q", passages, top_k=3)

    assert [p.text for p in result] == ["second", "third", "first"]


@patch("app.reranking._load_encoder")
def test_truncates_to_top_k_after_reordering(mock_load):
    """Reorder first, then cut. Cutting first would discard the passage the
    reranker was going to promote, which is the entire gain."""
    mock_load.return_value = _encoder([0.1, 0.2, 0.9])
    passages = _passages(("a", 0.9), ("b", 0.8), ("c", 0.7))

    result = rerank_passages("q", passages, top_k=1)

    assert [p.text for p in result] == ["c"]


@patch("app.reranking._load_encoder")
def test_preserves_the_dense_cosine_on_each_passage(mock_load):
    """Cross-encoder outputs are unbounded logits, not cosines. The off-topic
    guard's thresholds are calibrated on cosines (0.499-0.574 measured), so
    overwriting `score` would make every gate meaningless."""
    mock_load.return_value = _encoder([0.1, 0.9])
    passages = _passages(("a", 0.88), ("b", 0.62))

    result = rerank_passages("q", passages, top_k=2)

    assert result[0].text == "b"
    assert result[0].score == pytest.approx(0.62)
    assert result[1].score == pytest.approx(0.88)


@patch("app.reranking._load_encoder")
def test_scores_only_the_candidates_it_is_given(mock_load):
    encoder = _encoder([0.5, 0.4])
    mock_load.return_value = encoder

    rerank_passages("what is x", _passages(("a", 0.9), ("b", 0.8)), top_k=2)

    pairs = encoder.predict.call_args.args[0]
    assert pairs == [("what is x", "a"), ("what is x", "b")]


@patch("app.reranking._load_encoder")
def test_a_single_candidate_skips_the_model_entirely(mock_load):
    """Nothing to reorder, so paying 36ms of cross-encoder for it is waste."""
    passages = _passages(("only", 0.9))

    result = rerank_passages("q", passages, top_k=5)

    assert result == passages
    mock_load.assert_not_called()


@patch("app.reranking._load_encoder")
def test_no_candidates_returns_nothing_without_loading_the_model(mock_load):
    assert rerank_passages("q", [], top_k=5) == []
    mock_load.assert_not_called()


@patch("app.reranking._load_encoder")
def test_a_reranker_failure_falls_back_to_dense_order(mock_load):
    """A reranker is an improvement, not a dependency. If the model cannot load
    on the instance, the service must degrade to the dense ranking it had
    before rather than fail the request."""
    mock_load.side_effect = RuntimeError("model unavailable")
    passages = _passages(("a", 0.9), ("b", 0.8))

    result = rerank_passages("q", passages, top_k=2)

    assert [p.text for p in result] == ["a", "b"]


def test_rerank_depth_fits_the_measured_latency_budget():
    """Measured on the instance, not extrapolated. Depth 7 costs 118.8ms there
    against a 52ms boundary-A baseline, landing at 170.8ms inside the 200ms
    target. Depth 10 measured 162.3ms and lands at 214.3ms, over budget -- so
    the default must stay at or below 8, and 8 leaves nothing for the guard's
    23-36ms variance."""
    assert RERANK_DEPTH <= 8, "default depth would breach the measured budget"


def test_rerank_depth_is_overridable_for_faster_hardware(monkeypatch):
    """The depth that fits is a property of the machine, not of the code. On 8
    vCPU depth 10 fits and is worth another 1.6pp of recall@5; that should not
    require a redeploy of code."""
    import importlib

    monkeypatch.setenv("RERANK_DEPTH", "10")
    import app.reranking

    importlib.reload(app.reranking)
    try:
        assert app.reranking.RERANK_DEPTH == 10
    finally:
        monkeypatch.delenv("RERANK_DEPTH", raising=False)
        importlib.reload(app.reranking)


def test_reranking_can_be_disabled_by_environment(monkeypatch):
    """An escape hatch for the instance: 2 vCPU against this machine's 8, so if
    measured latency there breaks the budget it must be switchable without a
    redeploy of code."""
    reranking_enabled.cache_clear()
    monkeypatch.setenv("RERANK_ENABLED", "0")
    try:
        assert reranking_enabled() is False
    finally:
        reranking_enabled.cache_clear()


def test_reranking_is_on_by_default(monkeypatch):
    reranking_enabled.cache_clear()
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    try:
        assert reranking_enabled() is True
    finally:
        reranking_enabled.cache_clear()
