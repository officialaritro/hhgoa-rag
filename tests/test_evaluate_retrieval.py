import json
from unittest.mock import patch

from app.schemas import RetrievalOutput, RetrievedPassage
from scripts.evaluate_retrieval import recall_at_k


def _write_corpus(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def _retrieval_output(query, source_passages):
    return RetrievalOutput(
        query=query,
        strategy="fixed_size",
        passages=[
            RetrievedPassage(text=sp, source_passage=sp, is_selected=True, score=0.9)
            for sp in source_passages
        ],
    )


@patch("scripts.evaluate_retrieval.retrieve")
def test_recall_at_k_counts_hit_when_selected_passage_is_retrieved(
    mock_retrieve, tmp_path
):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "query_id": 1,
                "query": "q1",
                "answer": "a1",
                "passages": [{"text": "correct passage", "is_selected": True}],
            }
        ],
    )
    mock_retrieve.return_value = _retrieval_output("q1", ["correct passage", "other"])

    score = recall_at_k("fixed_size", corpus_path=str(corpus_path), k=2, sample_size=1)

    assert score == 1.0


@patch("scripts.evaluate_retrieval.retrieve")
def test_recall_at_k_counts_miss_when_selected_passage_is_not_retrieved(
    mock_retrieve, tmp_path
):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "query_id": 1,
                "query": "q1",
                "answer": "a1",
                "passages": [{"text": "correct passage", "is_selected": True}],
            }
        ],
    )
    mock_retrieve.return_value = _retrieval_output(
        "q1", ["unrelated one", "unrelated two"]
    )

    score = recall_at_k("fixed_size", corpus_path=str(corpus_path), k=2, sample_size=1)

    assert score == 0.0


@patch("scripts.evaluate_retrieval.retrieve")
def test_recall_at_k_skips_rows_with_no_selected_passage(mock_retrieve, tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "query_id": 1,
                "query": "q1",
                "answer": "a1",
                "passages": [{"text": "x", "is_selected": False}],
            },
        ],
    )

    score = recall_at_k("fixed_size", corpus_path=str(corpus_path), k=2, sample_size=1)

    assert score == 0.0
    mock_retrieve.assert_not_called()
