"""The shared parent-passage store.

Every chunking strategy indexes spans of these passages rather than carrying
its own copy of the text. Before this store existed, each chunk row held both
its own `text` and a full duplicate of its `source_passage`, which cost
555-765 bytes per chunk (measured) and made a wide strategy slate unaffordable:
at ~1.15M chunks across eight strategies that duplication alone is ~700 MB.

`passage_id` is the index into the loaded list. That is the contract the
span-addressed chunk rows depend on, so the build must never reorder or filter
passages after assigning ids.
"""

import json
import pickle
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

# Explicit rather than the interpreter default. Indices are built on the
# development machine and read on the instance; pinning the protocol means the
# artifact does not silently change if either side's Python moves.
PICKLE_PROTOCOL = 4


class Passage(TypedDict):
    text: str
    is_selected: bool
    query_id: int
    query: str


class Chunk(TypedDict, total=False):
    """A span of one parent passage.

    `start`/`end` delimit what gets embedded. `ret_start`/`ret_end` optionally
    widen what gets returned. `text` replaces both for chunks that are not a
    substring of any single parent. `embed_query` prepends the parent's gold
    query at embed time only.
    """

    parent_id: int
    start: int
    end: int
    ret_start: int
    ret_end: int
    text: str
    embed_query: bool


# A chunker sees the whole store, not one passage at a time, because not every
# strategy is per-passage: query_group deliberately aggregates across all the
# passages sharing a query_id. Per-passage strategies wrap themselves in
# `per_passage` rather than each re-implementing the loop.
Chunker = Callable[[list["Passage"]], Iterator[Chunk]]


def per_passage(
    fn: Callable[["Passage", int], Iterable[Chunk]],
) -> Chunker:
    """Adapts a per-passage chunker into a corpus-level one, passing each
    passage's id so the yielded spans can address their parent."""

    def chunker(passages: list["Passage"]) -> Iterator[Chunk]:
        for passage_id, passage in enumerate(passages):
            yield from fn(passage, passage_id)

    return chunker


def build_passage_store(corpus_path: str, output_path: str) -> int:
    """Flattens the corpus's per-row passage lists into one positionally
    addressed store, preserving corpus order so ids stay stable across rebuilds."""
    passages: list[Passage] = []
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            for passage in row["passages"]:
                passages.append(
                    {
                        "text": passage["text"],
                        "is_selected": bool(passage["is_selected"]),
                        "query_id": row["query_id"],
                        "query": row["query"],
                    }
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(passages, f, protocol=PICKLE_PROTOCOL)
    return len(passages)


def load_passage_store(path: str) -> list[Passage]:
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_text(chunk: dict[str, Any], passages: list[Passage]) -> str:
    """The RETURN text: what generation and the UI see for this chunk.

    Defaults to the embed span, but `ret_start`/`ret_end` override it, because
    three strategies deliberately return more than they embed:
    `parent_child` embeds a small child and returns the whole parent, and
    `sentence_window` embeds one sentence and returns it with its neighbours.
    Without that split those strategies would be indistinguishable from
    fixed-size chunking with a smaller window.

    Chunks carrying `text` are not a substring of any single parent --
    query_group concatenates across passages -- so they store their own.
    """
    stored = chunk.get("text")
    if stored is not None:
        return stored
    parent = passages[chunk["parent_id"]]["text"]
    return parent[
        chunk.get("ret_start", chunk["start"]) : chunk.get("ret_end", chunk["end"])
    ]


def resolve_embed_text(chunk: dict[str, Any], passages: list[Passage]) -> str:
    """The EMBED text: what actually goes into the vector.

    `embed_query` prepends the passage's own gold query. That is document
    expansion -- the doc2query/HyDE pattern -- except the synthetic-query
    generation step is replaced by ground truth the dataset already ships in
    every corpus row, so it costs no model calls.
    """
    stored = chunk.get("text")
    if stored is not None:
        base = stored
    else:
        parent = passages[chunk["parent_id"]]["text"]
        base = parent[chunk["start"] : chunk["end"]]
    if chunk.get("embed_query"):
        return f"Q: {passages[chunk['parent_id']]['query']}\n{base}"
    return base
