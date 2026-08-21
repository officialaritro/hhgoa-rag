"""Reciprocal rank fusion.

RRF is used rather than score averaging because the rankings being merged are
not on comparable scales: BM25 scores are unbounded term sums, dense scores are
cosines in [-1, 1], and two dense indices over different chunk granularities
have different score distributions again (measured: in-corpus median 0.740 for
whole_passage against 0.781 for sentence_window). Rank is the only thing they
share, so fusing on rank is the only merge that does not silently privilege
whichever list happens to produce bigger numbers.
"""

import pytest

from app.fusion import reciprocal_rank_fusion


def test_a_document_ranked_first_everywhere_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])

    assert fused[0][0] == "a"


def test_agreement_beats_a_single_first_place():
    """The property that makes fusion worth doing: something both rankings like
    should outrank something only one ranking loves."""
    fused = reciprocal_rank_fusion([["solo", "agreed"], ["agreed", "other"]])

    order = [doc for doc, _ in fused]
    assert order.index("agreed") < order.index("solo")


def test_documents_appearing_in_only_one_ranking_are_still_included():
    fused = reciprocal_rank_fusion([["a"], ["b"]])

    assert {doc for doc, _ in fused} == {"a", "b"}


def test_scores_are_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_the_k_constant_damps_the_advantage_of_rank_one():
    """With k=60 the gap between rank 1 and rank 2 is small, which is the point:
    it stops one list's top hit from dominating a document that several lists
    agree on slightly lower down."""
    # Asymmetric on purpose: with mirrored rankings every document ties by
    # symmetry and there is no spread to measure at any k.
    small_k = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]], k=0)
    large_k = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]], k=60)

    spread_small = small_k[0][1] - small_k[-1][1]
    spread_large = large_k[0][1] - large_k[-1][1]
    assert spread_large < spread_small


def test_an_empty_ranking_is_ignored_rather_than_breaking_the_merge():
    """A lexical search with no keyword matches returns nothing; the dense half
    must still produce a result."""
    fused = reciprocal_rank_fusion([[], ["a", "b"]])

    assert [doc for doc, _ in fused] == ["a", "b"]


def test_no_rankings_at_all_gives_no_results():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_let_one_ranking_count_more_than_another():
    """Needed to fuse a strong ranking with a weak one without the weak one
    dragging it down -- measured recall@5 spans 0.848 to 0.422 across the
    strategies, so treating every list as equally trustworthy is a choice that
    has to be defensible, not a default."""
    unweighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[3.0, 1.0])

    assert unweighted[0][1] == pytest.approx(unweighted[1][1])
    assert weighted[0][0] == "a"
    assert weighted[0][1] > weighted[1][1]


def test_duplicate_entries_within_one_ranking_count_once():
    """Dedup happens upstream by parent passage, but a list arriving with a
    repeat must not let that document accumulate rank credit twice."""
    fused = reciprocal_rank_fusion([["a", "a", "b"]])

    assert len(fused) == 2
    assert fused[0][0] == "a"
