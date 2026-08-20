"""Off-topic thresholds come from the index manifest, not a hand-maintained dict.

The defect this replaces: `_DEFAULT_OFFTOPIC_THRESHOLDS` was a module constant
listing two strategies. A value verified against the semantic index shipped as
the fixed_size default and refused 38.5% of real in-corpus questions. With eight
strategies the same dict would be eight chances to forget one, and forgetting
one is silent.

Writing the threshold into the manifest at build time makes an uncalibrated
strategy unservable rather than mis-served.
"""

import json
from pathlib import Path

import pytest

from app.guardrails import MissingCalibration, offtopic_threshold


def _manifest(tmp_path, strategy, **extra):
    index_path = tmp_path / f"index_{strategy}.faiss"
    manifest = index_path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "chunks": 100,
                **extra,
            }
        )
    )
    return str(index_path)


def test_reads_the_threshold_the_build_measured(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFTOPIC_SIMILARITY_THRESHOLD_RECURSIVE", raising=False)
    path = _manifest(tmp_path, "recursive", offtopic_threshold=0.531)
    offtopic_threshold.cache_clear()

    assert offtopic_threshold("recursive", index_path=path) == 0.531


def test_refuses_to_serve_a_strategy_with_no_measured_threshold(tmp_path, monkeypatch):
    """The whole point. An index that was built but never calibrated must fail
    loudly at the guard rather than fall back to some other strategy's number."""
    monkeypatch.delenv("OFFTOPIC_SIMILARITY_THRESHOLD_RECURSIVE", raising=False)
    path = _manifest(tmp_path, "recursive")  # no offtopic_threshold key
    offtopic_threshold.cache_clear()

    with pytest.raises(MissingCalibration) as excinfo:
        offtopic_threshold("recursive", index_path=path)

    assert "recursive" in str(excinfo.value)
    assert "calibrate" in str(excinfo.value).lower()


def test_environment_override_still_wins(tmp_path, monkeypatch):
    """Retuning without a rebuild is a documented operational path."""
    path = _manifest(tmp_path, "recursive", offtopic_threshold=0.531)
    monkeypatch.setenv("OFFTOPIC_SIMILARITY_THRESHOLD_RECURSIVE", "0.62")
    offtopic_threshold.cache_clear()

    assert offtopic_threshold("recursive", index_path=path) == 0.62


def test_a_missing_manifest_is_reported_as_uncalibrated(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFTOPIC_SIMILARITY_THRESHOLD_RECURSIVE", raising=False)
    offtopic_threshold.cache_clear()

    with pytest.raises(MissingCalibration):
        offtopic_threshold("recursive", index_path=str(tmp_path / "absent.faiss"))


def test_threshold_is_read_from_disk_only_once(tmp_path, monkeypatch):
    """Called on the request path, so it must not stat and parse JSON per query."""
    monkeypatch.delenv("OFFTOPIC_SIMILARITY_THRESHOLD_RECURSIVE", raising=False)
    path = _manifest(tmp_path, "recursive", offtopic_threshold=0.531)
    offtopic_threshold.cache_clear()

    assert offtopic_threshold("recursive", index_path=path) == 0.531
    Path(path).with_suffix(".manifest.json").unlink()
    assert offtopic_threshold("recursive", index_path=path) == 0.531
