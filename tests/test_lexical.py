"""BM25 lexical retrieval.

Implemented on scipy sparse rather than adding a BM25 package: sklearn and scipy
are already dependencies, and the scoring is a precomputed weight matrix plus a
column sum. That avoids a new dependency and a lock update on a deadline, and it
means the behaviour below is pinned rather than trusted.

BM25 earns its place here because MS MARCO is a keyword-heavy web-search corpus
where lexical matching fails on *different* queries than a dense bi-encoder
does, which is the only reason fusing them can beat either alone. The tests
below pin the four behaviours that distinguish BM25 from raw term counting:
IDF weighting, term-frequency saturation, length normalisation, and graceful
handling of out-of-vocabulary query terms.
"""

import pytest

from app.lexical import BM25Index


def _index(documents):
    return BM25Index.build(documents)


def test_a_document_containing_the_query_term_outranks_one_that_does_not():
    index = _index(["the coati is a mammal", "turbines generate electricity"])

    scores = index.scores("coati")

    assert scores[0] > scores[1]
    assert scores[1] == pytest.approx(0.0)


def test_a_rare_term_contributes_more_than_a_common_one():
    """IDF. Without it, matching a stopword-like term counts as much as matching
    the one word that actually identifies the document."""
    documents = [
        "common common common rare",
        "common common common",
        "common common common",
        "common common common",
    ]
    index = _index(documents)

    assert index.scores("rare")[0] > index.scores("common")[0]


def test_repeated_terms_saturate_rather_than_scaling_linearly():
    """The k1 term. Ten occurrences must not score ten times one occurrence, or
    keyword stuffing wins every ranking."""
    index = _index(
        ["alpha", "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha"]
    )

    once, ten_times = index.scores("alpha")

    assert ten_times > once
    assert ten_times < 10 * once, "term frequency scaled linearly; k1 not applied"


def test_a_shorter_document_scores_higher_for_the_same_match():
    """The b term. One match in a five-word document is stronger evidence than
    one match in a hundred-word document."""
    index = _index(["alpha beta", "alpha " + "filler " * 60])

    short, long = index.scores("alpha")

    assert short > long


def test_an_unknown_query_term_scores_everything_zero_without_raising():
    index = _index(["alpha beta", "gamma delta"])

    scores = index.scores("nonexistentword")

    assert list(scores) == [0.0, 0.0]


def test_a_query_with_some_known_terms_still_matches_on_those():
    index = _index(["the coati is a mammal", "turbines generate electricity"])

    scores = index.scores("nonexistentword coati")

    assert scores[0] > scores[1]


def test_ranking_returns_document_ids_best_first():
    index = _index(
        [
            "turbines generate electricity",
            "the coati is a mammal of central america",
            "coati coati coati",
        ]
    )

    ranked = index.top_k("coati", k=2)

    assert len(ranked) == 2
    assert ranked[0][0] in (1, 2)
    assert all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1)), (
        "results not sorted by descending score"
    )


def test_top_k_never_returns_documents_that_do_not_match_at_all():
    """A zero-scoring document is not a lexical candidate. Padding the list with
    zeros would hand RRF meaningless ranks to fuse."""
    index = _index(["alpha", "beta", "gamma"])

    assert len(index.top_k("alpha", k=3)) == 1


def test_case_and_punctuation_do_not_prevent_a_match():
    index = _index(["The Coati, a mammal."])

    assert index.scores("coati")[0] > 0


def test_round_trips_through_disk(tmp_path):
    """Built once at index time, loaded per process at serve time -- the same
    contract the FAISS indices have."""
    index = _index(["the coati is a mammal", "turbines generate electricity"])
    path = tmp_path / "bm25.pkl"

    index.save(str(path))
    reloaded = BM25Index.load(str(path))

    assert reloaded.top_k("coati", k=1) == index.top_k("coati", k=1)


def test_reports_how_many_documents_it_indexed():
    index = _index(["alpha beta", "gamma delta", "epsilon zeta"])

    assert index.n_documents == 3


def test_a_corpus_with_no_usable_tokens_does_not_crash():
    """sklearn's default token pattern drops single characters, so a corpus of
    them yields an empty vocabulary. Degenerate, but it must score zero rather
    than raise from inside the vectorizer -- the build should not be able to die
    on unexpected corpus content."""
    index = _index(["a b", "c d"])

    assert index.n_documents == 2
    assert list(index.scores("a")) == [0.0, 0.0]
    assert index.top_k("a", k=5) == []
