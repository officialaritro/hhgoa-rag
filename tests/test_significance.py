"""Paired bootstrap confidence intervals.

Written because I overclaimed. I reported `fusion` at 0.854 recall@5 as beating
`whole_passage` at 0.848 and called it "the only thing that beat a plain
passage". At 500 queries that margin is three queries against a ~1.6pp standard
error -- not a result. Every comparison in the report now carries an interval so
the same mistake is not available.

Paired rather than independent: the systems are evaluated on the *same* queries,
so the per-query outcomes are correlated. Treating them as independent samples
throws away that pairing and produces intervals far wider than the truth,
which would hide real differences as surely as point estimates invent them.
"""

import pytest

from scripts.significance import bootstrap_ci, paired_bootstrap


def test_identical_systems_give_an_interval_containing_zero():
    hits = [1, 0, 1, 1, 0, 1, 0, 1] * 10

    low, high = paired_bootstrap(hits, list(hits), seed=0)

    assert low <= 0.0 <= high


def test_a_strictly_better_system_gives_a_wholly_positive_interval():
    """B wins every query A wins, plus twenty more. No resample can reverse it."""
    a = [1] * 40 + [0] * 60
    b = [1] * 60 + [0] * 40

    low, high = paired_bootstrap(a, b, seed=0)

    assert low > 0.0, f"interval ({low:.4f}, {high:.4f}) should exclude zero"


def test_a_three_query_margin_in_five_hundred_is_not_significant():
    """The actual case that prompted this: fusion 0.854 against whole_passage
    0.848 is three queries. The interval must contain zero."""
    a = [1] * 424 + [0] * 76
    b = [1] * 427 + [0] * 73

    low, high = paired_bootstrap(a, b, seed=0)

    assert low <= 0.0 <= high, (
        f"a three-query margin was reported as significant: ({low:.4f}, {high:.4f})"
    )


def test_a_seven_point_margin_in_five_hundred_is_significant():
    """And the reranking result, +6.8 points, must survive."""
    a = [1] * 424 + [0] * 76
    b = [1] * 458 + [0] * 42

    low, high = paired_bootstrap(a, b, seed=0)

    assert low > 0.0
    assert high > low


def test_pairing_is_preserved_across_resamples():
    """If pairing were dropped, two systems that always agree would still show a
    spread. They must not."""
    hits = [1, 0] * 50

    low, high = paired_bootstrap(hits, list(hits), seed=1)

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_the_same_seed_gives_the_same_interval():
    a = [1, 0, 1, 1, 0] * 20
    b = [1, 1, 0, 1, 0] * 20

    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_different_seeds_give_similar_but_not_identical_intervals():
    a = [1, 0, 1, 1, 0] * 40
    b = [1, 1, 0, 1, 0] * 40

    first = paired_bootstrap(a, b, seed=1)
    second = paired_bootstrap(a, b, seed=2)

    assert first != second
    assert abs(first[0] - second[0]) < 0.05


def test_mismatched_lengths_are_rejected():
    """A silent zip would truncate and quietly compare different query sets."""
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0, 1], [1, 0], seed=0)


def test_single_system_interval_brackets_its_own_mean():
    hits = [1] * 424 + [0] * 76

    low, high = bootstrap_ci(hits, seed=0)

    assert low < 0.848 < high
    assert high - low < 0.10, "interval implausibly wide for n=500"


def test_a_wider_confidence_level_gives_a_wider_interval():
    hits = [1] * 424 + [0] * 76

    narrow = bootstrap_ci(hits, seed=0, confidence=0.80)
    wide = bootstrap_ci(hits, seed=0, confidence=0.99)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_no_observations_gives_a_degenerate_interval_rather_than_raising():
    assert bootstrap_ci([], seed=0) == (0.0, 0.0)
    assert paired_bootstrap([], [], seed=0) == (0.0, 0.0)
