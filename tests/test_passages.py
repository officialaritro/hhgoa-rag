import json

from app.passages import build_passage_store, load_passage_store, resolve_text


def _write_corpus(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def _corpus_row(query_id, query, passage_texts, selected_index=0):
    return {
        "query_id": query_id,
        "query": query,
        "answer": "an answer",
        "passages": [
            {"text": text, "is_selected": i == selected_index}
            for i, text in enumerate(passage_texts)
        ],
    }


def test_build_passage_store_assigns_sequential_ids_across_rows(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            _corpus_row(10, "first query", ["p0", "p1"]),
            _corpus_row(20, "second query", ["p2"]),
        ],
    )
    store_path = tmp_path / "passages.pkl"

    written = build_passage_store(str(corpus), str(store_path))

    assert written == 3
    passages = load_passage_store(str(store_path))
    assert [p["text"] for p in passages] == ["p0", "p1", "p2"]


def test_passage_store_is_addressable_by_list_position(tmp_path):
    """passage_id is the index into the store -- retrieval resolves parents by
    position, so this is the contract the span-addressed chunks depend on."""
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [_corpus_row(10, "q", ["alpha", "beta", "gamma"])])
    store_path = tmp_path / "passages.pkl"

    build_passage_store(str(corpus), str(store_path))
    passages = load_passage_store(str(store_path))

    assert passages[1]["text"] == "beta"


def test_build_passage_store_carries_query_and_is_selected(tmp_path):
    """The row's query is what query_aware embeds and what evaluation labels
    come from; is_selected is the relevance label. Both must survive."""
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus, [_corpus_row(42, "what is a coati", ["irrelevant", "relevant"], 1)]
    )
    store_path = tmp_path / "passages.pkl"

    build_passage_store(str(corpus), str(store_path))
    passages = load_passage_store(str(store_path))

    assert passages[1]["query"] == "what is a coati"
    assert passages[1]["query_id"] == 42
    assert passages[1]["is_selected"] is True
    assert passages[0]["is_selected"] is False


def test_store_does_not_duplicate_passage_text_per_chunk(tmp_path):
    """The whole point of the store: one copy of each passage, shared by every
    strategy. Two passages of identical text still get two entries (they are
    distinct passages), but no chunk carries its own copy."""
    corpus = tmp_path / "corpus.jsonl"
    long_text = "x" * 5000
    _write_corpus(corpus, [_corpus_row(1, "q", [long_text])])
    store_path = tmp_path / "passages.pkl"

    build_passage_store(str(corpus), str(store_path))

    # One passage of 5000 chars must not produce a store far larger than itself.
    assert store_path.stat().st_size < len(long_text) * 2


def test_resolve_text_slices_the_parent_for_a_span_chunk():
    passages = [
        {"text": "hello world", "is_selected": False, "query_id": 1, "query": "q"}
    ]
    chunk = {"parent_id": 0, "start": 6, "end": 11}

    assert resolve_text(chunk, passages) == "world"


def test_resolve_text_prefers_stored_text_when_chunk_is_not_a_span():
    """query_group chunks deliberately cross passage boundaries, so they carry
    their own text and cannot be resolved by slicing one parent."""
    passages = [
        {"text": "hello world", "is_selected": False, "query_id": 1, "query": "q"}
    ]
    chunk = {"parent_id": 0, "start": 0, "end": 0, "text": "spans two passages"}

    assert resolve_text(chunk, passages) == "spans two passages"
