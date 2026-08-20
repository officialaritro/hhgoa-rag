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
    """The chunk's own text: a slice of its parent for span chunks, or the
    stored text for chunks that are not a contiguous substring of one parent
    (query_group concatenates across passages, so it cannot be a span)."""
    stored = chunk.get("text")
    if stored is not None:
        return stored
    return passages[chunk["parent_id"]]["text"][chunk["start"] : chunk["end"]]
