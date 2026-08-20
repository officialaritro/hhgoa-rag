"""Tests for the indexing phase.

This phase must never import torch, for the mirror of the reason app/vectors.py
must never import faiss: on macOS arm64 both bundle their own libomp, and a
process holding both segfaults once MPS is used. It reads the flat float32 file
the embedding phase wrote and knows nothing about models.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.indexing import index_from_vectors
from app.passages import PICKLE_PROTOCOL
from app.vectors import sidecar_path

DIM = 16
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write_vectors(tmp_path, count, dim=DIM, model=MODEL):
    """Writes a vector file plus sidecar the way the embedding phase would."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((count, dim)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_path = tmp_path / "v.f32"
    vectors_path.write_bytes(np.ascontiguousarray(vectors).tobytes())
    Path(sidecar_path(str(vectors_path))).write_text(
        json.dumps({"count": count, "dimension": dim, "embedding_model": model})
    )
    metadata_path = tmp_path / "m.pkl"
    rows = [{"parent_id": i, "start": 0, "end": 10} for i in range(count)]
    with open(metadata_path, "wb") as f:
        pickle.dump(rows, f, protocol=PICKLE_PROTOCOL)
    return vectors, str(vectors_path), str(metadata_path)


def test_importing_the_indexing_module_does_not_pull_in_torch():
    """Mirror of the faiss check on app.vectors. Importing something convenient
    from app.embeddings that happens to touch torch would silently reunite the
    two OpenMP runtimes in the indexing process."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import app.indexing, sys; "
                "print('TORCH_LOADED' if 'torch' in sys.modules else 'CLEAN')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, f"app.indexing imported torch: {result.stdout}"


def test_builds_a_searchable_index_from_vectors_on_disk(tmp_path):
    import faiss

    vectors, vectors_path, metadata_path = _write_vectors(tmp_path, 300)
    index_path = str(tmp_path / "i.faiss")

    written = index_from_vectors(vectors_path, index_path, metadata_path)

    assert written == 300
    index = faiss.read_index(index_path)
    assert index.ntotal == 300
    _, ids = index.search(vectors[42:43], 1)
    assert ids[0][0] == 42


def test_manifest_takes_the_model_name_from_the_sidecar(tmp_path):
    """The indexing process never loads a model, so the only truthful source
    for the manifest is what the embedding process recorded."""
    _, vectors_path, metadata_path = _write_vectors(
        tmp_path, 50, model="some-other/model"
    )
    index_path = str(tmp_path / "i.faiss")

    index_from_vectors(vectors_path, index_path, metadata_path)

    manifest = json.loads(Path(index_path).with_suffix(".manifest.json").read_text())
    assert manifest["embedding_model"] == "some-other/model"
    assert manifest["dimension"] == DIM
    assert manifest["chunks"] == 50


def test_adds_in_slices_rather_than_loading_every_vector(tmp_path):
    """Peak memory has to stay bounded: the largest strategy is 349,983 x 384
    float32 = 537 MB, which is what pushed the single-process build to ~6.1 GB
    on an 8 GB machine."""
    _, vectors_path, metadata_path = _write_vectors(tmp_path, 500)
    seen: list[int] = []

    index_from_vectors(
        vectors_path,
        str(tmp_path / "i.faiss"),
        metadata_path,
        train_sample=100,
        add_batch=64,
        progress=seen.append,
    )

    assert seen == sorted(seen)
    assert seen[-1] == 500
    assert len(seen) >= 500 // 64


def test_trains_on_what_exists_when_fewer_vectors_than_the_sample(tmp_path):
    _, vectors_path, metadata_path = _write_vectors(tmp_path, 9)

    written = index_from_vectors(
        vectors_path, str(tmp_path / "i.faiss"), metadata_path, train_sample=1000
    )

    assert written == 9


def test_refuses_when_metadata_row_count_disagrees_with_the_vectors(tmp_path):
    """A mismatch means FAISS row ids no longer address the right metadata row,
    so every result would resolve the wrong parent passage while still looking
    like a plausible answer. That must fail loudly at build time."""
    _, vectors_path, metadata_path = _write_vectors(tmp_path, 100)
    with open(metadata_path, "wb") as f:
        pickle.dump([{"parent_id": 0, "start": 0, "end": 1}] * 99, f)

    with pytest.raises(ValueError) as excinfo:
        index_from_vectors(vectors_path, str(tmp_path / "i.faiss"), metadata_path)

    assert "99" in str(excinfo.value) and "100" in str(excinfo.value)
