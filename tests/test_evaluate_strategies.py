"""Tests for the retrieval-quality metrics.

These are the numbers the submission is graded on, and recall@k has never been
measured on this project before, so the arithmetic gets pinned rather than
eyeballed. Each test states the hand-computable expected value.
"""

import pytest

from scripts.evaluate_strategies import (
    dcg,
    is_hit,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_is_hit_matches_an_exact_passage():
    assert is_hit(["the selected passage"], "the selected passage") is True


def test_is_hit_rejects_an_unrelated_passage():
    assert is_hit(["the selected passage"], "something else entirely") is False


def test_is_hit_matches_a_passage_contained_in_a_multi_passage_chunk():
    """query_group chunks concatenate several passages, so the labelled passage
    is a substring of the retrieved chunk rather than equal to it. Without this
    that strategy would score near zero for a reason unrelated to its quality."""
    chunk = "First passage here. The selected passage. Third passage."

    assert is_hit(["The selected passage."], chunk) is True


def test_is_hit_is_false_when_no_label_exists():
    assert is_hit([], "anything") is False


def test_recall_at_k_counts_a_query_as_hit_only_within_k():
    ranked = ["miss", "miss", "target"]

    assert recall_at_k([(ranked, ["target"])], k=2) == 0.0
    assert recall_at_k([(ranked, ["target"])], k=3) == 1.0


def test_recall_at_k_averages_over_queries():
    queries = [
        (["target"], ["target"]),
        (["miss"], ["target"]),
    ]

    assert recall_at_k(queries, k=1) == 0.5


def test_mrr_uses_the_reciprocal_of_the_first_relevant_rank():
    """Third position means 1/3, not 1/2 -- an off-by-one here inflates every
    reported MRR."""
    queries = [(["miss", "miss", "target"], ["target"])]

    assert mrr_at_k(queries, k=10) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_nothing_relevant_is_retrieved():
    assert mrr_at_k([(["miss"], ["target"])], k=10) == 0.0


def test_mrr_ignores_relevant_results_beyond_k():
    queries = [(["miss", "miss", "target"], ["target"])]

    assert mrr_at_k(queries, k=2) == 0.0


def test_dcg_discounts_by_log2_of_rank_plus_one():
    """rank 1 contributes 1/log2(2) = 1.0; rank 2 contributes 1/log2(3)."""
    import math

    assert dcg([1, 0]) == pytest.approx(1.0)
    assert dcg([0, 1]) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_one_when_the_relevant_result_is_first():
    queries = [(["target", "miss"], ["target"])]

    assert ndcg_at_k(queries, k=2) == pytest.approx(1.0)


def test_ndcg_is_below_one_when_the_relevant_result_is_second():
    """Hand-computed: relevance [0, 1] gives DCG = 1/log2(3) = 0.6309, and the
    ideal ranking [1, 0] gives 1.0, so nDCG = 0.6309."""
    import math

    queries = [(["miss", "target"], ["target"])]

    assert ndcg_at_k(queries, k=2) == pytest.approx(1 / math.log2(3))


def test_ndcg_normalises_against_the_ideal_ranking():
    """Two relevant results retrieved at ranks 1 and 2 is a perfect ranking, so
    nDCG must be 1.0 -- not the raw DCG of 1.63."""
    queries = [(["a", "b"], ["a", "b"])]

    assert ndcg_at_k(queries, k=2) == pytest.approx(1.0)


def test_metrics_return_zero_for_no_evaluable_queries():
    assert recall_at_k([], k=5) == 0.0
    assert mrr_at_k([], k=10) == 0.0
    assert ndcg_at_k([], k=10) == 0.0
