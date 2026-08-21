"""Paired bootstrap confidence intervals for retrieval comparisons.

This exists because a point estimate invited a wrong conclusion. `fusion` scored
0.854 recall@5 against `whole_passage`'s 0.848 and was reported as beating it.
At 500 queries that is three queries, against a standard error of roughly 1.6
points -- not a result. Every comparison in docs/CHUNKING_REPORT.md now carries
an interval.

The bootstrap is **paired**: both systems are evaluated on the same queries, so
their per-query outcomes are correlated. Resampling them independently discards
that pairing and widens the interval well past the truth, which hides real
differences as reliably as point estimates manufacture fake ones. Resampling
query *indices* and reading both systems at those indices keeps it.
"""

import numpy as np

DEFAULT_ITERATIONS = 10_000
DEFAULT_CONFIDENCE = 0.95


def _percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    )


def bootstrap_ci(
    outcomes: list[int] | list[float],
    seed: int = 0,
    iterations: int = DEFAULT_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Confidence interval for one system's mean per-query outcome."""
    if not outcomes:
        return (0.0, 0.0)
    values = np.asarray(outcomes, dtype="float64")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(iterations, len(values)))
    return _percentile_interval(values[picks].mean(axis=1), confidence)


def paired_bootstrap(
    baseline: list[int] | list[float],
    candidate: list[int] | list[float],
    seed: int = 0,
    iterations: int = DEFAULT_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Interval for `mean(candidate) - mean(baseline)`, preserving pairing.

    An interval containing zero means the difference is not established at this
    sample size, whatever the point estimates say.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and "
            f"{len(candidate)}; a silent zip would compare different query sets"
        )
    if not baseline:
        return (0.0, 0.0)

    # Per-query differences, then resample those. Equivalent to resampling query
    # indices and differencing, and it makes the pairing impossible to lose.
    differences = np.asarray(candidate, dtype="float64") - np.asarray(
        baseline, dtype="float64"
    )
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(differences), size=(iterations, len(differences)))
    return _percentile_interval(differences[picks].mean(axis=1), confidence)


def describe(low: float, high: float) -> str:
    """One-word verdict for a report table."""
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "no difference"
