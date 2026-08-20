"""Pins the off-topic guard's calibration to what the build actually measured.

History this exists to prevent. Every other guardrail test either mocks
`check_off_topic` outright or passes an explicit `threshold=`, so none of them
exercise the value the service ships. That is how a threshold refusing 38.5% of
real in-corpus questions reached the live service: the constant was verified by
hand against the *semantic* index and shipped as the `fixed_size` default.

The fix is structural rather than a better constant. Thresholds are measured per
index at build time and written into that index's manifest
(scripts/calibrate_thresholds.py), and `offtopic_threshold` raises
`MissingCalibration` rather than borrowing another strategy's number. So these
tests read the manifests instead of restating measurements, which is what keeps
them from going stale: the previous version of this file pinned six semantic
scores that became fiction the moment the semantic threshold was retuned, while
every assertion kept passing.

These tests skip when `data/` is unpopulated, which is the case in CI.
"""

import json
from pathlib import Path

import pytest

from app.guardrails import MissingCalibration, check_off_topic, offtopic_threshold
from app.strategies import chunk_paths, dense_names

# A refusal on a real question is the damaging, user-visible failure, and the
# pipeline has two further lines of defence for off-topic input that gets
# through: the generation prompt returns a decline sentinel when the passages do
# not answer, and the groundedness guard refuses ungrounded answers. So the
# budget is set on false refusals, deliberately tighter than on leaks.
MAX_FALSE_REFUSAL_RATE = 0.10


def _manifest(strategy: str) -> dict:
    index_path, _ = chunk_paths(strategy)
    path = Path(index_path).with_suffix(".manifest.json")
    if not path.exists():
        pytest.skip(f"{path} not built; run scripts.build_all")
    return json.loads(path.read_text())


def _calibrated() -> list[str]:
    built = []
    for name in dense_names():
        index_path, _ = chunk_paths(name)
        if Path(index_path).with_suffix(".manifest.json").exists():
            built.append(name)
    if not built:
        pytest.skip("no indices built; run scripts.build_all")
    return built


@pytest.mark.parametrize("strategy", dense_names())
def test_every_served_strategy_carries_a_measured_threshold(strategy):
    """The coupling that matters. A strategy `/api/ask` will retrieve from and
    whose index was never calibrated must not be servable at all."""
    manifest = _manifest(strategy)

    assert "offtopic_threshold" in manifest, (
        f"{strategy} has an index but no measured threshold; "
        f"run scripts.calibrate_thresholds"
    )
    assert 0.0 < manifest["offtopic_threshold"] < 1.0


@pytest.mark.parametrize("strategy", dense_names())
def test_each_threshold_stays_within_the_false_refusal_budget(strategy):
    manifest = _manifest(strategy)
    rate = manifest["calibration"]["false_refusal_rate"]

    assert rate <= MAX_FALSE_REFUSAL_RATE, (
        f"{strategy} refuses {rate:.1%} of real in-corpus questions, over the "
        f"{MAX_FALSE_REFUSAL_RATE:.0%} budget"
    )


def test_thresholds_are_not_all_the_same_number():
    """The original defect in one assertion. The indices sit on different score
    scales -- shorter chunks make every cosine run higher -- so a single shared
    value is miscalibrated for all but one of them."""
    thresholds = {s: _manifest(s)["offtopic_threshold"] for s in _calibrated()}

    assert len(set(thresholds.values())) > 1, (
        f"every strategy got the identical threshold {thresholds}, which means "
        f"they were not measured per index"
    )


def test_a_query_enriched_index_is_calibrated_on_held_out_queries():
    """query_aware bakes each passage's own gold query into its vector, so an
    in-corpus query matches its own row almost perfectly. Calibrated naively it
    measured 0.722 with zero leaks and looked like the best strategy in the
    slate; held out it measures 0.400 with 5 of 8 probes leaking, the worst.
    A threshold from the naive measurement would refuse most real traffic.
    """
    from app.strategies import get

    for name in _calibrated():
        strategy = get(name)
        # Only the served enriched index. The control index is held out by
        # construction rather than by a search-time filter, and is not served.
        if strategy.axis != "enrichment" or not strategy.served:
            continue
        calibration = _manifest(name)["calibration"]
        assert calibration.get("held_out") is True, (
            f"{name} varies the embedded text using its own query, so its "
            f"threshold must be measured with that query's row excluded"
        )


@pytest.mark.parametrize("strategy", dense_names())
def test_the_shipped_threshold_is_what_the_guard_actually_uses(strategy):
    """Closes the gap that let the original bug through: every other guardrail
    test passes an explicit threshold, so none of them touch the shipped value."""
    manifest = _manifest(strategy)
    offtopic_threshold.cache_clear()
    try:
        effective = offtopic_threshold(strategy)
    finally:
        offtopic_threshold.cache_clear()

    assert effective == pytest.approx(manifest["offtopic_threshold"])

    just_under = effective - 0.01
    just_over = effective + 0.01
    assert check_off_topic(just_under, strategy=strategy).passed is False
    assert check_off_topic(just_over, strategy=strategy).passed is True
    offtopic_threshold.cache_clear()


def test_an_uncalibrated_strategy_cannot_be_served():
    """The structural guarantee. Without it, a newly added strategy silently
    borrows whatever number happens to be lying around."""
    offtopic_threshold.cache_clear()
    with pytest.raises(MissingCalibration):
        offtopic_threshold("a_strategy_that_was_never_built")
    offtopic_threshold.cache_clear()
