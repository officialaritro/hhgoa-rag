from unittest.mock import patch

import numpy as np

import app.embeddings
from app.embeddings import active_device, cosine_similarity, embed, embed_batch


@patch("app.embeddings._get_model")
def test_embed_calls_model_encode_and_returns_its_result(mock_get_model):
    fake_model = mock_get_model.return_value
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])

    result = embed("hello world")

    fake_model.encode.assert_called_once_with("hello world", normalize_embeddings=True)
    assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))


@patch("app.embeddings._get_model")
def test_embed_batch_calls_model_encode_once_with_all_texts(mock_get_model):
    fake_model = mock_get_model.return_value
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    result = embed_batch(["first", "second"], batch_size=32)

    fake_model.encode.assert_called_once_with(
        ["first", "second"], normalize_embeddings=True, batch_size=32
    )
    assert np.array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([0.6, 0.8])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_active_device_is_unset_by_default(monkeypatch):
    """Unset means let sentence-transformers choose, which is what the service
    wants on the instance -- it has no GPU and auto-detection lands on CPU."""
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)

    assert active_device() is None


def test_active_device_reads_the_environment(monkeypatch):
    """The overnight build sets this to 'mps'. Without it the build silently
    takes the 326 texts/sec CPU path instead of 506 on Apple Silicon -- a
    78-minute run instead of 50, discovered only in the morning."""
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")

    assert active_device() == "mps"


def test_model_is_constructed_on_the_requested_device(monkeypatch):
    """A stub module is injected rather than patching the real
    `sentence_transformers.SentenceTransformer`.

    Patching the real attribute forces the real package to import, which imports
    torch. Elsewhere in this suite faiss is used, and on macOS arm64 a process
    holding both aborts with `OMP: Error #15` -- both bundle their own libomp.
    That took down the whole run at collection-order-dependent points. The stub
    keeps this test honest about the behaviour under test (the device argument
    `_get_model` passes) without loading a deep-learning stack to check one
    keyword argument.
    """
    import sys
    import types

    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    captured = {}

    class FakeModel:
        def __init__(self, name, device=None):
            captured["name"] = name
            captured["device"] = device

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

    app.embeddings._get_model.cache_clear()
    try:
        app.embeddings._get_model()
    finally:
        app.embeddings._get_model.cache_clear()

    assert captured["device"] == "mps"
    assert captured["name"].endswith("all-MiniLM-L6-v2")
