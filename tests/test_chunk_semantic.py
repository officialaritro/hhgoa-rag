import json
from unittest.mock import patch

import numpy as np

from scripts.chunk_semantic import chunk_corpus, chunk_text_semantic, split_sentences


def test_split_sentences_splits_on_sentence_boundaries():
    result = split_sentences("First sentence. Second sentence! Third one?")

    assert result == ["First sentence.", "Second sentence!", "Third one?"]


def test_chunk_text_semantic_single_sentence_returns_as_is():
    result = chunk_text_semantic("Only one sentence here.")

    assert result == ["Only one sentence here."]


@patch("scripts.chunk_semantic.cosine_similarity")
@patch("scripts.chunk_semantic.embed_batch")
def test_chunk_text_semantic_merges_similar_adjacent_sentences(mock_embed, mock_cosine):
    mock_embed.side_effect = lambda sentences: np.zeros((len(sentences), 3))
    mock_cosine.return_value = 0.9  # above threshold -> merge

    result = chunk_text_semantic(
        "Sentence one. Sentence two. Sentence three.", similarity_threshold=0.5
    )

    assert result == ["Sentence one. Sentence two. Sentence three."]


@patch("scripts.chunk_semantic.cosine_similarity")
@patch("scripts.chunk_semantic.embed_batch")
def test_chunk_text_semantic_splits_on_dissimilar_adjacent_sentences(
    mock_embed, mock_cosine
):
    mock_embed.side_effect = lambda sentences: np.zeros((len(sentences), 3))
    mock_cosine.return_value = 0.1  # below threshold -> split every time

    result = chunk_text_semantic(
        "Sentence one. Sentence two. Sentence three.", similarity_threshold=0.5
    )

    assert result == ["Sentence one.", "Sentence two.", "Sentence three."]


@patch("scripts.chunk_semantic.cosine_similarity", return_value=0.9)
@patch(
    "scripts.chunk_semantic.embed_batch",
    side_effect=lambda sentences: np.zeros((len(sentences), 3)),
)
def test_chunk_corpus_carries_is_selected_metadata_and_strategy_label(
    mock_embed, mock_cosine, tmp_path
):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "query_id": 1,
                "query": "q",
                "answer": "a",
                "passages": [
                    {"text": "First sentence. Second sentence.", "is_selected": True},
                    {"text": "Only one.", "is_selected": False},
                ],
            }
        )
        + "\n"
    )
    output_path = tmp_path / "chunks.jsonl"

    written = chunk_corpus(str(corpus_path), str(output_path))

    lines = [json.loads(line) for line in output_path.read_text().strip().split("\n")]
    assert written == len(lines)
    assert any(c["is_selected"] is True for c in lines)
    assert any(c["is_selected"] is False for c in lines)
    assert all(c["strategy"] == "semantic" for c in lines)
