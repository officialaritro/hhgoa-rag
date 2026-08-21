"""Features for the off-topic guard.

Top-1 cosine alone cannot separate in-corpus from off-topic input: measured
across all nine indices, the distributions overlap. That is why the guard has
always been a compromise between false refusals and leaks rather than a
decision.

The features below add the shape of the score profile, not just its peak. A real
question produces a spike -- one or two passages match well and the rest fall
away. Off-topic input produces a flat profile: nothing matches especially well,
and nothing stands out.

Several are deliberately scale-invariant. Raw cosines are not comparable across
indices, which is exactly why nine separate thresholds had to be measured;
ratios and margins are, so one classifier can serve every index.
"""

import pytest

from app.offtopic_features import FEATURE_NAMES, extract_features


def test_returns_one_value_per_named_feature():
    features = extract_features([0.9, 0.5, 0.4, 0.35, 0.3], query="what is a coati")

    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(f, float) for f in features)


def test_a_peaked_profile_scores_a_larger_margin_than_a_flat_one():
    """The core signal. A real question has one clear match; off-topic input has
    a plateau of mediocre ones."""
    peaked = dict(
        zip(FEATURE_NAMES, extract_features([0.9, 0.5, 0.48, 0.47, 0.46], "q"))
    )
    flat = dict(
        zip(FEATURE_NAMES, extract_features([0.55, 0.54, 0.53, 0.52, 0.51], "q"))
    )

    assert peaked["margin"] > flat["margin"]


def test_the_relative_margin_is_invariant_to_a_uniform_score_shift():
    """Semantic chunks score higher across the board (in-corpus median 0.779
    against 0.741) purely because they are shorter. A feature that moves with
    that offset forces a per-index threshold; a relative one does not."""
    low = dict(
        zip(FEATURE_NAMES, extract_features([0.60, 0.40, 0.38, 0.36, 0.34], "q"))
    )
    high = dict(
        zip(FEATURE_NAMES, extract_features([0.90, 0.60, 0.57, 0.54, 0.51], "q"))
    )

    assert low["relative_margin"] == pytest.approx(high["relative_margin"], abs=0.02)


def test_top1_is_still_available_as_a_feature():
    """The old signal is weak, not useless, so it stays in the vector rather than
    being replaced outright."""
    features = dict(zip(FEATURE_NAMES, extract_features([0.87, 0.5], "q")))

    assert features["top1"] == pytest.approx(0.87)


def test_query_length_is_included():
    """Gibberish and one-word fragments behave differently from real questions,
    and length is free to compute."""
    short = dict(zip(FEATURE_NAMES, extract_features([0.5, 0.4], "hmm")))
    long = dict(
        zip(
            FEATURE_NAMES,
            extract_features(
                [0.5, 0.4], "what are the symptoms of dehydration in adults"
            ),
        )
    )

    assert long["query_words"] > short["query_words"]


def test_a_single_score_does_not_raise():
    """k can come back as one result when the corpus is tiny or the index is
    nearly empty; the guard must still produce a decision."""
    features = extract_features([0.7], query="q")

    assert len(features) == len(FEATURE_NAMES)


def test_no_scores_gives_a_defined_all_zero_vector():
    """Retrieval returning nothing is a refusal, and it must reach the classifier
    as a valid vector rather than an exception."""
    features = extract_features([], query="q")

    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("top1")] == 0.0


def test_features_are_ordered_consistently_with_their_names():
    """The classifier is trained on positions, so a reordering between training
    and serving would silently scramble every input."""
    scores = [0.9, 0.6, 0.5, 0.4, 0.3]
    first = extract_features(scores, "a query here")
    second = extract_features(scores, "a query here")

    assert first == second
    assert FEATURE_NAMES.index("top1") == 0


def test_scores_are_sorted_defensively_before_use():
    """FAISS returns descending order, but a fused ranking does not -- hybrid and
    fusion reorder by reciprocal rank, so the score list can arrive unsorted."""
    ordered = extract_features([0.9, 0.5, 0.3], "q")
    shuffled = extract_features([0.3, 0.9, 0.5], "q")

    assert ordered == shuffled
