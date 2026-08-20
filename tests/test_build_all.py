"""Tests for the unattended build orchestrator.

The value of these is that the script runs for an hour with nobody watching, so
its retry, resume and reporting logic has to be right the first time -- a bug
here costs a night, and there are two days to the deadline.
"""

import json
from pathlib import Path

import pytest

from scripts.build_all import (
    PhaseFailed,
    format_duration,
    render_progress,
    run_with_retries,
    work_remaining,
)


def test_run_with_retries_returns_the_first_success():
    calls = []

    def succeeds():
        calls.append(1)
        return "done"

    assert run_with_retries(succeeds, attempts=3, backoff=0) == "done"
    assert len(calls) == 1


def test_run_with_retries_recovers_after_a_transient_failure():
    """A dropped model load or a momentary memory spike should not end the
    night's work when the next attempt would succeed."""
    attempts = []

    def fails_once():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        return "recovered"

    assert run_with_retries(fails_once, attempts=3, backoff=0) == "recovered"
    assert len(attempts) == 2


def test_run_with_retries_raises_phase_failed_after_exhausting_attempts():
    """It must fail loudly with the underlying cause, not return None -- a
    silent None would let the orchestrator record the strategy as built."""

    def always_fails():
        raise RuntimeError("disk full")

    with pytest.raises(PhaseFailed) as excinfo:
        run_with_retries(always_fails, attempts=2, backoff=0)

    assert "disk full" in str(excinfo.value)
    assert "2 attempts" in str(excinfo.value)


def test_work_remaining_skips_a_strategy_whose_index_is_already_built(tmp_path):
    """Resume. A run interrupted at strategy six must not redo the first five."""
    index_path = tmp_path / "index_done.faiss"
    index_path.write_bytes(b"stub")
    manifest = tmp_path / "index_done.manifest.json"
    manifest.write_text(json.dumps({"chunks": 10, "dimension": 384}))

    assert work_remaining(str(index_path), force=False) == ()


def test_work_remaining_rebuilds_when_the_manifest_is_missing(tmp_path):
    """An index without a manifest is the signature of a build killed between
    writing the index and writing the manifest. It cannot be trusted, and
    load_index refuses it anyway."""
    index_path = tmp_path / "index_partial.faiss"
    index_path.write_bytes(b"stub")

    assert work_remaining(str(index_path), force=False) == ("embed", "index")


def test_work_remaining_reindexes_when_vectors_exist_but_the_index_does_not(tmp_path):
    """Phase one succeeded and phase two died. Re-embedding would throw away the
    expensive half of the work for nothing."""
    index_path = tmp_path / "index_x.faiss"
    vectors_path = tmp_path / "vectors_x.f32"
    vectors_path.write_bytes(b"stub")
    Path(str(vectors_path) + ".meta.json").write_text(
        json.dumps({"count": 5, "dimension": 384, "embedding_model": "m"})
    )

    assert work_remaining(
        str(index_path), force=False, vectors_path=str(vectors_path)
    ) == ("index",)


def test_work_remaining_redoes_everything_when_forced(tmp_path):
    index_path = tmp_path / "index_done.faiss"
    index_path.write_bytes(b"stub")
    (tmp_path / "index_done.manifest.json").write_text(json.dumps({"chunks": 10}))

    assert work_remaining(str(index_path), force=True) == ("embed", "index")


def test_format_duration_is_readable_at_overnight_scale():
    assert format_duration(45) == "45s"
    assert format_duration(150) == "2m30s"
    assert format_duration(3725) == "1h02m"


def test_render_progress_reports_percent_rate_and_eta():
    line = render_progress(
        label="whole_passage/embed", done=25_000, total=100_000, elapsed=50.0
    )

    assert "25%" in line
    assert "500" in line  # 25,000 in 50s = 500/sec
    assert "whole_passage/embed" in line


def test_render_progress_omits_percent_when_the_total_is_unknown():
    """Semantic chunking cannot report a total without doing the work twice, so
    the bar has to degrade to a count rather than print a fake percentage."""
    line = render_progress(
        label="semantic/embed", done=1234, total=None, elapsed=10.0
    )

    assert "%" not in line
    assert "1,234" in line


def test_render_progress_survives_a_zero_elapsed_time():
    """Called immediately after the first batch, elapsed can round to zero and
    a naive rate calculation divides by it."""
    line = render_progress(label="x/embed", done=10, total=100, elapsed=0.0)

    assert "10%" in line
