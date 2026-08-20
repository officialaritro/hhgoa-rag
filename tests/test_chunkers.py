"""Tests for the eight chunking strategies.

Each chunker is a pure function over the passage store, so these are cheap and
exact. The properties that matter across all of them:

  * spans must address real substrings of their parent (an off-by-one here
    silently truncates or bleeds text into every answer)
  * every chunk must be reachable, i.e. no passage silently produces nothing
  * embed text and return text are allowed to differ, and for three strategies
    they must
"""

import pytest

from app.chunkers import (
    fixed_size_chunker,
    parent_child_chunker,
    query_aware_chunker,
    query_group_chunker,
    recursive_chunker,
    sentence_spans,
    sentence_window_chunker,
    whole_passage_chunker,
)
from app.passages import resolve_embed_text, resolve_text


def _passage(text, query="a query", query_id=1, is_selected=False):
    return {
        "text": text,
        "is_selected": is_selected,
        "query_id": query_id,
        "query": query,
    }


# ------------------------------------------------------------ sentence spans


def test_sentence_spans_returns_offsets_that_slice_back_to_the_sentences():
    text = "First one. Second one! Third one?"

    spans = sentence_spans(text)

    assert [text[s:e] for s, e in spans] == [
        "First one.",
        "Second one!",
        "Third one?",
    ]


def test_sentence_spans_treats_text_without_terminators_as_one_sentence():
    text = "no terminator here"

    assert sentence_spans(text) == [(0, 18)]


def test_sentence_spans_of_empty_text_is_empty():
    assert sentence_spans("") == []


# ------------------------------------------------------------------- helpers


def _assert_spans_are_valid(chunks, passages):
    """Every span must slice its parent without going out of bounds, and must
    not be empty -- an empty chunk embeds the empty string and pollutes the
    index with a vector that matches nothing in particular."""
    for chunk in chunks:
        if "text" in chunk:
            assert chunk["text"], "stored text must not be empty"
            continue
        parent = passages[chunk["parent_id"]]["text"]
        assert 0 <= chunk["start"] < chunk["end"] <= len(parent), chunk
        assert resolve_embed_text(chunk, passages).strip()


# ----------------------------------------------------------- fixed_size (2)


def test_fixed_size_leaves_a_short_passage_whole():
    passages = [_passage("short passage")]

    chunks = list(fixed_size_chunker(passages))

    assert len(chunks) == 1
    assert resolve_text(chunks[0], passages) == "short passage"


def test_fixed_size_splits_a_long_passage_with_overlap():
    passages = [_passage("x" * 1600)]

    chunks = list(fixed_size_chunker(passages))

    assert len(chunks) > 1
    _assert_spans_are_valid(chunks, passages)
    # step = 700 * (1 - 0.2) = 560, so consecutive spans overlap by 140
    assert chunks[1]["start"] - chunks[0]["start"] == 560


def test_fixed_size_covers_every_character_of_the_passage():
    passages = [_passage("y" * 2000)]

    chunks = list(fixed_size_chunker(passages))

    covered = set()
    for chunk in chunks:
        covered.update(range(chunk["start"], chunk["end"]))
    assert covered == set(range(2000))


# ------------------------------------------------------------ recursive (3)


def test_recursive_prefers_sentence_boundaries_over_mid_word_cuts():
    """The defect this strategy exists to fix: fixed_size slices at a character
    offset, so it cuts words in half. Recursive splitting must not."""
    sentence = "The quick brown fox jumps over the lazy dog every single day. "
    passages = [_passage(sentence * 12)]

    chunks = list(recursive_chunker(passages))

    assert len(chunks) > 1
    _assert_spans_are_valid(chunks, passages)
    for chunk in chunks:
        text = resolve_embed_text(chunk, passages)
        assert not text.startswith(" "), "chunk begins mid-whitespace"
        # a sentence-aligned chunk ends on a terminator or is the tail
        assert text.rstrip()[-1] in ".!?" or chunk["end"] == len(passages[0]["text"])


def test_recursive_leaves_a_passage_under_the_target_whole():
    passages = [_passage("One sentence only, comfortably short.")]

    chunks = list(recursive_chunker(passages))

    assert len(chunks) == 1


def test_recursive_splits_a_single_unbroken_run_that_exceeds_the_target():
    """No separator to fall back on. It must still bound the chunk size rather
    than emit one oversized chunk."""
    passages = [_passage("z" * 1500)]

    chunks = list(recursive_chunker(passages))

    _assert_spans_are_valid(chunks, passages)
    assert all(chunk["end"] - chunk["start"] <= 400 for chunk in chunks)


# --------------------------------------------------------- parent_child (5)


def test_parent_child_embeds_the_child_but_returns_the_whole_parent():
    """The entire premise of the strategy. If return text were the child, this
    would just be fixed_size with a smaller window."""
    body = "A. " * 300
    passages = [_passage(body)]

    chunks = list(parent_child_chunker(passages))

    assert len(chunks) > 1
    _assert_spans_are_valid(chunks, passages)
    for chunk in chunks:
        assert resolve_text(chunk, passages) == body
        assert len(resolve_embed_text(chunk, passages)) <= 200


# ------------------------------------------------------ sentence_window (6)


def test_sentence_window_embeds_one_sentence_and_returns_its_neighbours():
    passages = [_passage("One. Two. Three. Four.")]

    chunks = list(sentence_window_chunker(passages))

    assert len(chunks) == 4
    # the second chunk embeds "Two." and returns "One. Two. Three."
    assert resolve_embed_text(chunks[1], passages) == "Two."
    assert resolve_text(chunks[1], passages) == "One. Two. Three."


def test_sentence_window_clamps_at_the_passage_edges():
    passages = [_passage("Alpha. Beta.")]

    chunks = list(sentence_window_chunker(passages))

    assert resolve_text(chunks[0], passages) == "Alpha. Beta."
    assert resolve_text(chunks[-1], passages) == "Alpha. Beta."


# ----------------------------------------------------------- query_aware (7)


def test_query_aware_embeds_the_query_with_the_passage_but_returns_it_bare():
    """Document expansion using the query the dataset already ships, instead of
    paying an LLM to invent synthetic ones."""
    passages = [_passage("Coatis are raccoon relatives.", query="what is a coati")]

    chunks = list(query_aware_chunker(passages))

    assert len(chunks) == 1
    embedded = resolve_embed_text(chunks[0], passages)
    assert "what is a coati" in embedded
    assert "Coatis are raccoon relatives." in embedded
    assert resolve_text(chunks[0], passages) == "Coatis are raccoon relatives."


# ----------------------------------------------------------- query_group (8)


def test_query_group_merges_every_passage_sharing_a_query_id():
    passages = [
        _passage("First passage.", query_id=7),
        _passage("Second passage.", query_id=7),
        _passage("Different query.", query_id=9),
    ]

    chunks = list(query_group_chunker(passages))

    texts = [resolve_text(chunk, passages) for chunk in chunks]
    assert any("First passage." in t and "Second passage." in t for t in texts)
    assert any("Different query." in t for t in texts)


def test_query_group_chunks_carry_their_own_text_because_they_cross_parents():
    """A concatenation of two passages is not a substring of either, so it
    cannot be span-addressed and must store text."""
    passages = [
        _passage("A" * 40, query_id=1),
        _passage("B" * 40, query_id=1),
    ]

    chunks = list(query_group_chunker(passages))

    assert all("text" in chunk for chunk in chunks)


def test_query_group_splits_an_oversized_group():
    passages = [_passage("word " * 200, query_id=1) for _ in range(4)]

    chunks = list(query_group_chunker(passages))

    assert len(chunks) > 1
    assert all(len(resolve_text(c, passages)) <= 1200 for c in chunks)


# --------------------------------------------------------------- whole (1)


def test_whole_passage_emits_one_chunk_per_passage():
    passages = [_passage("first"), _passage("second")]

    chunks = list(whole_passage_chunker(passages))

    assert [resolve_text(c, passages) for c in chunks] == ["first", "second"]


# ---------------------------------------------------- cross-strategy property


@pytest.mark.parametrize(
    "chunker",
    [
        whole_passage_chunker,
        fixed_size_chunker,
        recursive_chunker,
        parent_child_chunker,
        sentence_window_chunker,
        query_aware_chunker,
        query_group_chunker,
    ],
)
def test_no_passage_is_silently_dropped(chunker):
    """A chunker that yields nothing for some passage makes that passage
    unretrievable, and nothing downstream would notice."""
    passages = [
        _passage("Short."),
        _passage("A much longer passage. " * 40),
        _passage("Mid-length passage with several sentences. Here is another. " * 3),
        _passage("x" * 1300),
    ]

    chunks = list(chunker(passages))

    _assert_spans_are_valid(chunks, passages)
    if chunker is not query_group_chunker:
        covered = {chunk["parent_id"] for chunk in chunks}
        assert covered == set(range(len(passages)))
