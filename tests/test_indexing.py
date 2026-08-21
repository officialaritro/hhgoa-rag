"""Tests for loading an index back, and refusing to load a mismatched one.

Building is covered by tests/test_index_from_vectors.py. What is left here is
`load_index`, and specifically the guard that made it necessary: two different
embedding models both emit 384-dim vectors, so an index built with one and
searched with the other does not crash. FAISS searches happily and returns
neighbours from an unrelated vector space, and the answers look plausible and
mean nothing. That happened in practice -- `.env` pointed at bge-small while the
indices had been built with all-MiniLM -- so the manifest check exists to make
it loud.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.indexing import IndexModelMismatch, index_from_vectors, load_index
from app.passages import PICKLE_PROTOCOL
from app.vectors import sidecar_path

DIM = 8


def _build(tmp_path, count=40, model="sentence-transformers/all-MiniLM-L6-v2"):
    """Writes a vector file and metadata the way the embedding phase would, then
    builds a real index from them."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((count, DIM)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    vectors_path = tmp_path / "v.f32"
    vectors_path.write_bytes(np.ascontiguousarray(vectors).tobytes())
    Path(sidecar_path(str(vectors_path))).write_text(
        json.dumps({"count": count, "dimension": DIM, "embedding_model": model})
    )
    metadata_path = tmp_path / "chunks.pkl"
    rows = [{"parent_id": i, "start": 0, "end": 10} for i in range(count)]
    with open(metadata_path, "wb") as f:
        pickle.dump(rows, f, protocol=PICKLE_PROTOCOL)

    index_path = tmp_path / "index.faiss"
    index_from_vectors(str(vectors_path), str(index_path), str(metadata_path))
    return vectors, str(index_path), str(metadata_path)


def test_load_index_returns_the_index_and_its_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    _, index_path, metadata_path = _build(tmp_path)

    index, rows = load_index(index_path, metadata_path)

    assert index.ntotal == 40
    assert len(rows) == 40
    assert rows[0] == {"parent_id": 0, "start": 0, "end": 10}


def test_a_loaded_index_still_finds_its_own_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    vectors, index_path, metadata_path = _build(tmp_path)

    index, rows = load_index(index_path, metadata_path)
    _, ids = index.search(vectors[17:18], 1)

    assert rows[ids[0][0]]["parent_id"] == 17


def test_load_index_refuses_an_index_built_by_a_different_model(tmp_path, monkeypatch):
    """The dangerous case, because nothing fails on its own: both models emit
    384-dim vectors, so retrieval returns confident nonsense instead of an
    error."""
    _, index_path, metadata_path = _build(tmp_path, model="BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises(IndexModelMismatch) as excinfo:
        load_index(index_path, metadata_path)

    message = str(excinfo.value)
    assert "bge-small-en-v1.5" in message
    assert "all-MiniLM-L6-v2" in message
    assert "rebuild" in message.lower()


def test_an_index_predating_manifests_loads_with_a_warning_not_a_failure(
    tmp_path, monkeypatch
):
    """Refusing every manifest-less index would make an old artifact unloadable
    rather than merely unverified."""
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "anything/at-all")
    _, index_path, metadata_path = _build(tmp_path)
    Path(index_path).with_suffix(".manifest.json").unlink()

    index, rows = load_index(index_path, metadata_path)

    assert index.ntotal == 40
    assert len(rows) == 40
