"""Tests for the embedding phase.

This phase must never import faiss. On macOS arm64, faiss-cpu and torch each
link their own copy of libomp; using MPS in a process where both are loaded
segfaults (verified: exit 139, and KMP_DUPLICATE_LIB_OK=TRUE does not save it,
it only converts the abort into the segfault). The build is therefore split
into an embedding process and an indexing process that never co-load. These
tests pin the contract between them: raw float32 vectors on disk plus a sidecar
recording the shape.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.vectors import embed_chunks_to_disk, read_vector_shape

DIM = 16


def _fake_embed(texts, **kwargs):
    out = np.empty((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.standard_normal(DIM).astype("float32")
        out[i] = vector / np.linalg.norm(vector)
    return out


def _passages(n):
    return [
        {
            "text": f"passage {i} " + "filler " * (i % 5 + 1),
            "is_selected": i % 3 == 0,
            "query_id": i,
            "query": f"query {i}",
        }
        for i in range(n)
    ]


def _spans(passages):
    for passage_id, passage in enumerate(passages):
        yield {"parent_id": passage_id, "start": 0, "end": len(passage["text"])}


def test_importing_the_embedding_module_does_not_pull_in_faiss():
    """The whole point of the split, and it has to be checked in a fresh
    interpreter: a sibling test in this session may already have imported faiss,
    so `"faiss" in sys.modules` proves nothing here. What matters is that a
    process whose only import is app.vectors never loads faiss -- if it does,
    the embedding process co-loads both OpenMP runtimes and segfaults on MPS.

    This also guards the transitive path: app.vectors -> app.strategies ->
    app.passages must all stay faiss-free, which is easy to break by importing
    something convenient from app.indexing.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import app.vectors, sys; "
                "print('FAISS_LOADED' if 'faiss' in sys.modules else 'CLEAN')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, f"app.vectors imported faiss: {result.stdout}"


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_writes_vectors_metadata_and_shape_sidecar(mock, tmp_path):
    vectors_path = tmp_path / "v.f32"
    metadata_path = tmp_path / "m.pkl"

    written = embed_chunks_to_disk(
        chunks=_spans(_passages(40)),
        passages=_passages(40),
        vectors_path=str(vectors_path),
        metadata_path=str(metadata_path),
        batch_size=8,
    )

    assert written == 40
    assert vectors_path.exists()
    assert metadata_path.exists()
    count, dim = read_vector_shape(str(vectors_path))
    assert (count, dim) == (40, DIM)


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_vectors_on_disk_are_readable_as_a_float32_matrix(mock, tmp_path):
    vectors_path = tmp_path / "v.f32"

    embed_chunks_to_disk(
        chunks=_spans(_passages(25)),
        passages=_passages(25),
        vectors_path=str(vectors_path),
        metadata_path=str(tmp_path / "m.pkl"),
        batch_size=7,
    )

    count, dim = read_vector_shape(str(vectors_path))
    matrix = np.memmap(vectors_path, dtype="float32", mode="r", shape=(count, dim))
    assert matrix.shape == (25, DIM)
    # every row must be a unit vector -- the index relies on inner product
    # being cosine, so an unnormalized row silently distorts ranking
    norms = np.linalg.norm(np.asarray(matrix), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_row_order_on_disk_matches_metadata_order(mock, tmp_path):
    """Index row i must correspond to metadata row i. If these drift, retrieval
    resolves the wrong parent passage and every answer cites the wrong source."""
    vectors_path = tmp_path / "v.f32"
    metadata_path = tmp_path / "m.pkl"
    passages = _passages(30)

    embed_chunks_to_disk(
        chunks=_spans(passages),
        passages=passages,
        vectors_path=str(vectors_path),
        metadata_path=str(metadata_path),
        batch_size=4,
    )

    rows = pickle.loads(metadata_path.read_bytes())
    count, dim = read_vector_shape(str(vectors_path))
    matrix = np.asarray(
        np.memmap(vectors_path, dtype="float32", mode="r", shape=(count, dim))
    )
    for i in (0, 13, 29):
        expected = _fake_embed([passages[rows[i]["parent_id"]]["text"]])[0]
        assert np.allclose(matrix[i], expected, atol=1e-6)


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_embeds_in_bounded_batches(mock, tmp_path):
    embed_chunks_to_disk(
        chunks=_spans(_passages(100)),
        passages=_passages(100),
        vectors_path=str(tmp_path / "v.f32"),
        metadata_path=str(tmp_path / "m.pkl"),
        batch_size=16,
    )

    sizes = [len(call.args[0]) for call in mock.call_args_list]
    assert max(sizes) <= 16
    assert len(sizes) >= 100 // 16


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_reports_monotonic_progress(mock, tmp_path):
    seen: list[int] = []

    embed_chunks_to_disk(
        chunks=_spans(_passages(50)),
        passages=_passages(50),
        vectors_path=str(tmp_path / "v.f32"),
        metadata_path=str(tmp_path / "m.pkl"),
        batch_size=10,
        progress=seen.append,
    )

    assert seen == sorted(seen)
    assert seen[-1] == 50


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_raises_when_the_chunker_yields_nothing(mock, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        embed_chunks_to_disk(
            chunks=iter([]),
            passages=_passages(5),
            vectors_path=str(tmp_path / "v.f32"),
            metadata_path=str(tmp_path / "m.pkl"),
        )

    assert "yielded nothing" in str(excinfo.value)


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_sidecar_records_the_embedding_model(mock, tmp_path):
    """The indexing process writes the manifest but never loads the model, so
    the model name has to travel with the vectors."""
    vectors_path = tmp_path / "v.f32"

    embed_chunks_to_disk(
        chunks=_spans(_passages(12)),
        passages=_passages(12),
        vectors_path=str(vectors_path),
        metadata_path=str(tmp_path / "m.pkl"),
        batch_size=6,
    )

    sidecar = json.loads(Path(str(vectors_path) + ".meta.json").read_text())
    assert sidecar["count"] == 12
    assert sidecar["dimension"] == DIM
    assert "all-MiniLM-L6-v2" in sidecar["embedding_model"]


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_embeds_the_embed_text_not_the_return_text(mock, tmp_path):
    """Three strategies embed something different from what they return. If the
    build embeds the return text, query_aware silently loses its query and
    parent_child embeds the whole parent -- both become plain whole_passage
    while still looking like distinct strategies in the report.
    """
    passages = [
        {
            "text": "Coatis are raccoon relatives.",
            "is_selected": False,
            "query_id": 1,
            "query": "what is a coati",
        }
    ]
    chunks = [
        {"parent_id": 0, "start": 0, "end": 29, "embed_query": True},
    ]

    embed_chunks_to_disk(
        chunks=iter(chunks),
        passages=passages,
        vectors_path=str(tmp_path / "v.f32"),
        metadata_path=str(tmp_path / "m.pkl"),
        batch_size=4,
    )

    embedded = mock.call_args_list[0].args[0][0]
    assert "what is a coati" in embedded, f"query was not embedded: {embedded!r}"


@patch("app.vectors.embed_batch", side_effect=_fake_embed)
def test_embeds_the_child_span_when_the_return_span_is_wider(mock, tmp_path):
    passages = [
        {
            "text": "A" * 100 + "B" * 100,
            "is_selected": False,
            "query_id": 1,
            "query": "q",
        }
    ]
    chunks = [
        {"parent_id": 0, "start": 100, "end": 200, "ret_start": 0, "ret_end": 200}
    ]

    embed_chunks_to_disk(
        chunks=iter(chunks),
        passages=passages,
        vectors_path=str(tmp_path / "v.f32"),
        metadata_path=str(tmp_path / "m.pkl"),
    )

    embedded = mock.call_args_list[0].args[0][0]
    assert embedded == "B" * 100, "embedded the return span instead of the child"
