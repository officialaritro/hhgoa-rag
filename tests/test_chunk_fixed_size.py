import json

from scripts.chunk_fixed_size import chunk_corpus, chunk_text


def test_chunk_text_returns_single_chunk_when_shorter_than_chunk_size():
    result = chunk_text("short text", chunk_size=200, overlap=0.2)

    assert result == ["short text"]


def test_chunk_text_splits_long_text_into_overlapping_windows():
    text = "a" * 500

    result = chunk_text(text, chunk_size=200, overlap=0.2)

    assert len(result) > 1
    assert all(len(c) <= 200 for c in result)
    # step = 200 * (1 - 0.2) = 160, so consecutive windows overlap by 40 chars
    assert result[0][160:200] == result[1][0:40]


def test_chunk_text_covers_the_full_text():
    text = "0123456789" * 30  # 300 chars, deterministic content

    chunks = chunk_text(text, chunk_size=100, overlap=0.2)

    # every character position must appear in at least one chunk
    reconstructed = set()
    start = 0
    step = int(100 * 0.8)
    for chunk in chunks:
        for offset, ch in enumerate(chunk):
            reconstructed.add((start + offset, ch))
        start += step
    for i, ch in enumerate(text):
        assert (i, ch) in reconstructed


def test_chunk_corpus_carries_is_selected_metadata(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "query_id": 1,
                "query": "q",
                "answer": "a",
                "passages": [
                    {"text": "a" * 500, "is_selected": True},
                    {"text": "short", "is_selected": False},
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
    assert all(c["strategy"] == "fixed_size" for c in lines)
