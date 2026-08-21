"""Reciprocal rank fusion, for merging rankings that do not share a scale.

Score averaging is not an option here. BM25 produces unbounded term sums, dense
retrieval produces cosines in [-1, 1], and two dense indices over different
chunk granularities produce different distributions again -- measured on this
corpus, in-corpus median top-1 similarity is 0.740 for whole_passage against
0.781 for sentence_window. Averaging those would silently weight whichever list
emits larger numbers. Rank is the only property they share.
"""

from collections.abc import Hashable, Sequence

# The standard damping constant. Large enough that rank 1 is not overwhelmingly
# better than rank 2, which is the whole point: a document several rankings agree
# on a little further down should be able to beat one list's favourite.
DEFAULT_K = 60

# Measured lexical weight sweep for `hybrid` on this corpus, 500 labelled
# queries, recall@5 (scripts/evaluate_strategies.py plus a weight probe):
#
#   BM25 alone                    0.558
#   dense alone (whole_passage)   0.848
#   lexical weight 1.0            0.740   <- standard, equal-weight RRF
#                  0.5            0.790
#                  0.2            0.820
#                  0.1            0.840
#                  0.05           0.850
#
# Fusion only stops hurting once the lexical ranking is weighted almost to
# nothing, and 0.850 against 0.848 is inside the noise of 500 queries (one query
# is 0.2 points). So `hybrid` ships at equal weight: that is what hybrid
# retrieval means, and the honest result is that it does not help here. Tuning
# the weight down to 0.05 would produce a number that looks like a tie while
# actually having switched lexical retrieval off.
#
# The reason is the corpus, not the implementation. These are natural-language
# questions against short web passages, and an answer rarely repeats its
# question's words -- the vocabulary gap dense retrieval exists to close. BM25's
# reputation on MS MARCO comes from official document ranking with tuned
# stemming and stopword handling, neither of which applies here.
HYBRID_LEXICAL_WEIGHT = 1.0


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Merges ranked lists into one, best first.

    Each ranking contributes `weight / (k + rank)` per document. Documents absent
    from a ranking simply receive nothing from it, so a lexical search that
    matched no keywords does not penalise the dense half -- it just abstains.

    `weights` exists because these rankings are not equally trustworthy: measured
    recall@5 across the strategies spans 0.848 to 0.422, so treating every list
    as equal is a decision that should be stated rather than defaulted into.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"{len(rankings)} rankings but {len(weights)} weights; "
            f"a mismatch here silently drops or misweights a ranking"
        )

    totals: dict[Hashable, float] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[Hashable] = set()
        for position, document in enumerate(ranking, start=1):
            # A repeat inside one ranking must not accumulate credit twice.
            if document in seen:
                continue
            seen.add(document)
            totals[document] = totals.get(document, 0.0) + weight / (k + position)

    return sorted(totals.items(), key=lambda item: -item[1])
