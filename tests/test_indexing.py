import json
from unittest.mock import patch

import numpy as np

from app.indexing import build_index, load_index


def _write_chunks(path, chunks):
    with open(path, "w") as f:
        f.writelines(json.dumps(chunk) + "\n" for chunk in chunks)


@patch("app.indexing.embed_batch")
def test_build_index_returns_chunk_count_and_persists_files(mock_embed, tmp_path):
    mock_embed.side_effect = lambda texts, **kwargs: np.array(
        [[float(len(text)), 0.0, 0.0] for text in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "short",
                "source_passage": "short",
                "is_selected": True,
                "query_id": 1,
                "strategy": "fixed_size",
            },
            {
                "text": "a bit longer text",
                "source_passage": "a bit longer text",
                "is_selected": False,
                "query_id": 2,
                "strategy": "fixed_size",
            },
        ],
    )
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.pkl"

    written = build_index(str(chunks_path), str(index_path), str(metadata_path))

    assert written == 2
    assert index_path.exists()
    assert metadata_path.exists()


@patch("app.indexing.embed_batch")
def test_load_index_returns_index_and_metadata_matching_input(mock_embed, tmp_path):
    mock_embed.side_effect = lambda texts, **kwargs: np.array(
        [[float(len(text)), 0.0, 0.0] for text in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "hello",
                "source_passage": "hello",
                "is_selected": True,
                "query_id": 7,
                "strategy": "fixed_size",
            }
        ],
    )
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.pkl"
    build_index(str(chunks_path), str(index_path), str(metadata_path))

    index, metadata = load_index(str(index_path), str(metadata_path))

    assert index.ntotal == 1
    assert metadata[0]["query_id"] == 7
    assert metadata[0]["text"] == "hello"


@patch("app.indexing.embed_batch")
def test_built_index_returns_nearest_neighbor_for_query_vector(mock_embed, tmp_path):
    mock_embed.side_effect = lambda texts, **kwargs: np.array(
        [[float(len(text)), 0.0, 0.0] for text in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "aa",
                "source_passage": "aa",
                "is_selected": False,
                "query_id": 1,
                "strategy": "fixed_size",
            },
            {
                "text": "aaaaaaaaaa",
                "source_passage": "aaaaaaaaaa",
                "is_selected": True,
                "query_id": 2,
                "strategy": "fixed_size",
            },
        ],
    )
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.pkl"
    build_index(str(chunks_path), str(index_path), str(metadata_path))
    index, metadata = load_index(str(index_path), str(metadata_path))

    query_vec = np.array([[9.0, 0.0, 0.0]], dtype="float32")
    _distances, indices = index.search(query_vec, k=1)

    assert metadata[indices[0][0]]["query_id"] == 2


@patch("app.indexing.embed_batch")
def test_build_index_quantizes_vectors_to_fit_memory_budget(mock_embed, tmp_path):
    """Both chunking strategies' indices must be resident together on a
    t3.small's 2GiB RAM; a raw float32 IndexFlatIP (4 bytes/dim) doesn't
    leave headroom, so build_index must persist compressed (int8) codes."""
    mock_embed.side_effect = lambda texts, **kwargs: np.array(
        [[float(len(text)), 0.0, 0.0] for text in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "a" * n,
                "source_passage": "a" * n,
                "is_selected": False,
                "query_id": n,
                "strategy": "fixed_size",
            }
            for n in range(1, 9)
        ],
    )
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.pkl"

    build_index(str(chunks_path), str(index_path), str(metadata_path))
    index, _ = load_index(str(index_path), str(metadata_path))

    dimension = 3
    float32_bytes_per_vector = dimension * 4
    assert index.sa_code_size() < float32_bytes_per_vector


@patch("app.indexing.embed_batch")
def test_build_index_records_the_embedding_model_in_a_manifest(mock_embed, tmp_path):
    mock_embed.side_effect = lambda texts, **kw: np.array(
        [[1.0, 0.0, 0.0] for _ in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "a",
                "source_passage": "a",
                "is_selected": True,
                "query_id": 1,
                "strategy": "fixed_size",
            }
        ],
    )
    index_path = tmp_path / "index.faiss"
    build_index(str(chunks_path), str(index_path), str(tmp_path / "meta.pkl"))

    manifest = json.loads((tmp_path / "index.manifest.json").read_text())
    assert manifest["embedding_model"]
    assert manifest["dimension"] == 3
    assert manifest["chunks"] == 1


@patch("app.indexing.embed_batch")
def test_load_index_refuses_an_index_built_by_a_different_model(mock_embed, tmp_path):
    """The failure this prevents is silent: both models emit 384-dim vectors,
    so FAISS returns confident neighbours from an unrelated vector space."""
    from app.indexing import IndexModelMismatch

    mock_embed.side_effect = lambda texts, **kw: np.array(
        [[1.0, 0.0, 0.0] for _ in texts], dtype="float32"
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "text": "a",
                "source_passage": "a",
                "is_selected": True,
                "query_id": 1,
                "strategy": "fixed_size",
            }
        ],
    )
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "meta.pkl"
    with patch("app.indexing.active_model_name", return_value="model-A"):
        build_index(str(chunks_path), str(index_path), str(metadata_path))

    with patch("app.indexing.active_model_name", return_value="model-B"):
        try:
            load_index(str(index_path), str(metadata_path))
        except IndexModelMismatch as exc:
            assert "model-A" in str(exc) and "model-B" in str(exc)
        else:
            raise AssertionError("expected IndexModelMismatch")

    # Matching model still loads.
    with patch("app.indexing.active_model_name", return_value="model-A"):
        _, metadata = load_index(str(index_path), str(metadata_path))
        assert len(metadata) == 1
