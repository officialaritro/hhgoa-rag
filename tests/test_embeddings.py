from unittest.mock import patch

import numpy as np

from app.embeddings import cosine_similarity, embed, embed_batch


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
