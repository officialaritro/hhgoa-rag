import pytest

from app.strategies import (
    Strategy,
    UnknownStrategy,
    chunk_paths,
    get,
    names,
    per_passage,
)


def test_get_returns_the_registered_strategy():
    strategy = get("whole_passage")

    assert isinstance(strategy, Strategy)
    assert strategy.name == "whole_passage"


def test_get_raises_a_named_error_for_an_unknown_strategy():
    with pytest.raises(UnknownStrategy) as excinfo:
        get("does_not_exist")

    # The message must list what is valid -- a bare KeyError from a dict lookup
    # gave no way to tell a typo from an unbuilt strategy.
    assert "does_not_exist" in str(excinfo.value)
    assert "whole_passage" in str(excinfo.value)


def test_names_returns_every_registered_strategy():
    assert "whole_passage" in names()


def test_every_dense_strategy_has_a_chunker():
    """The coupling test. Registering a dense strategy without a chunker means
    build_index silently produces nothing for it -- an empty index that serves
    zero results rather than failing."""
    for name in names():
        strategy = get(name)
        if strategy.kind == "dense":
            assert strategy.chunker is not None, f"{name} is dense but has no chunker"


def test_every_composed_strategy_has_registered_members():
    """A fusion strategy naming a member that does not exist would fail at
    request time, per request, rather than at registration."""
    for name in names():
        strategy = get(name)
        if strategy.kind in ("fusion", "hybrid"):
            assert strategy.members, f"{name} is composed but names no members"
            for member in strategy.members:
                assert member in names(), f"{name} names unregistered member {member}"


def test_chunk_paths_are_distinct_per_strategy():
    assert chunk_paths("whole_passage") != chunk_paths("fixed_size")


def test_chunk_paths_returns_an_index_and_a_metadata_path():
    index_path, metadata_path = chunk_paths("whole_passage")

    assert index_path.endswith(".faiss")
    assert metadata_path.endswith(".pkl")
    assert "whole_passage" in index_path


def test_per_passage_adapter_yields_chunks_for_every_passage():
    passages = [
        {"text": "alpha", "is_selected": False, "query_id": 1, "query": "q"},
        {"text": "beta", "is_selected": False, "query_id": 1, "query": "q"},
    ]

    def one_span(passage, passage_id):
        yield {"parent_id": passage_id, "start": 0, "end": len(passage["text"])}

    chunks = list(per_passage(one_span)(passages))

    assert [c["parent_id"] for c in chunks] == [0, 1]
    assert [c["end"] for c in chunks] == [5, 4]


def test_whole_passage_chunker_produces_exactly_one_span_per_passage():
    passages = [
        {"text": "a" * 900, "is_selected": False, "query_id": 1, "query": "q"},
        {"text": "short", "is_selected": False, "query_id": 2, "query": "q"},
    ]

    chunks = list(get("whole_passage").chunker(passages))

    assert len(chunks) == 2
    assert chunks[0] == {"parent_id": 0, "start": 0, "end": 900}
    assert chunks[1] == {"parent_id": 1, "start": 0, "end": 5}
