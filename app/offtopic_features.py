"""Features for deciding whether a question is answerable from the corpus.

Top-1 cosine alone cannot do it. Measured across all nine indices, the in-corpus
and off-topic score distributions overlap, so every threshold is a compromise
between refusing real questions and admitting unanswerable ones rather than a
decision. A previous session recorded that as an accepted limitation; it is a
limitation of the *signal*, and the fix is a better signal.

What the peak misses is the shape of the score profile. A question the corpus can
answer produces a spike -- one or two passages match well and the rest fall away.
Off-topic input produces a plateau: nothing matches especially well and nothing
stands out from its neighbours.

Several features are scale-invariant on purpose. Raw cosines are not comparable
across indices -- shorter chunks push every score up, which is precisely why nine
separate thresholds had to be measured -- but margins expressed as a fraction of
the peak are. That is what lets one classifier serve every strategy.
"""

import math

FEATURE_NAMES = (
    "top1",
    "mean_top5",
    "margin",
    "relative_margin",
    "spread_top5",
    "top1_minus_top2",
    "query_words",
    "query_chars_log",
)


def extract_features(scores: list[float], query: str) -> list[float]:
    """Feature vector for one retrieval result.

    `scores` need not be sorted: FAISS returns descending order but hybrid and
    fusion reorder by reciprocal rank, so this sorts defensively rather than
    trusting the caller.
    """
    ordered = sorted((float(s) for s in scores), reverse=True)
    words = float(len(query.split()))
    chars_log = math.log1p(len(query))

    if not ordered:
        # Retrieval found nothing. A valid vector, not an exception: this is a
        # refusal path and the guard still has to return a decision.
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, words, chars_log]

    top1 = ordered[0]
    top5 = ordered[:5]
    mean_top5 = sum(top5) / len(top5)
    margin = top1 - mean_top5
    # Guard the denominator: a top-1 near zero means nothing matched, in which
    # case the relative margin carries no information and 0.0 is honest.
    relative_margin = margin / top1 if abs(top1) > 1e-6 else 0.0
    spread = (
        math.sqrt(sum((s - mean_top5) ** 2 for s in top5) / len(top5))
        if len(top5) > 1
        else 0.0
    )
    top1_minus_top2 = top1 - ordered[1] if len(ordered) > 1 else 0.0

    return [
        top1,
        mean_top5,
        margin,
        relative_margin,
        spread,
        top1_minus_top2,
        words,
        chars_log,
    ]
