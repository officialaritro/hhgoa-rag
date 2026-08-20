"""Given a text query, embed it and search a chosen strategy's FAISS index.

Chunks are spans of a shared parent-passage store rather than self-contained
text, so a hit is resolved in three steps: the FAISS row id gives a span row,
the span row names its parent, and the parent supplies both the text and the
relevance label.

Two things here are correctness fixes, not features:

  * **Dedup by parent.** With 20% overlap, several of the top-k slots could be
    near-identical text from a single passage, silently shrinking the distinct
    context handed to generation. Retrieval over-fetches and collapses.
  * **`text` is the return span, not the embedded span.** parent_child and
    sentence_window deliberately return more than they embed; handing back the
    embedded span would erase what makes them different from fixed-size
    chunking with a smaller window.
"""

from functools import cache

import numpy as np

from app.embeddings import embed
from app.indexing import load_index
from app.passages import load_passage_store, resolve_text
from app.schemas import RetrievalOutput, RetrievedPassage
from app.strategies import chunk_paths, dense_names, get

PASSAGE_STORE_PATH = "data/passages.pkl"

# Search this multiple of k before collapsing by parent. Searching for exactly
# k and then deduping returns fewer than k whenever two hits share a parent,
# which overlap makes common. 4x is cheap in FAISS (measured: retrieval is
# 19.6ms P50 at k=5 over 338k vectors) and leaves room for the worst realistic
# case of every hit collapsing into a handful of parents.
OVERFETCH = 4

# Kept as a module constant because callers outside retrieval reference it to
# discover which strategies are servable.
INDEX_PATHS = {name: chunk_paths(name) for name in dense_names()}


@cache
def _load_cached(strategy: str):
    """Loads each strategy's index+spans from disk once per process and keeps
    them resident -- reloading (and re-unpickling) on every request blew both
    the latency budget and memory (allocation churn) on the deployed instance."""
    index_path, metadata_path = chunk_paths(strategy)
    return load_index(index_path, metadata_path)


@cache
def _load_passages():
    """One passage store, shared by every strategy. Loaded once per process:
    ~36 MB for 99,767 passages, against which a per-request load would be
    absurd."""
    return load_passage_store(PASSAGE_STORE_PATH)


def retrieve(query: str, strategy: str, k: int = 5) -> RetrievalOutput:
    get(strategy)  # raises UnknownStrategy with the valid list, before any I/O
    index, rows = _load_cached(strategy)
    passages = _load_passages()

    query_vector = np.array([embed(query)], dtype="float32")
    scores, indices = index.search(query_vector, k * OVERFETCH)

    seen_parents: set[int] = set()
    found: list[RetrievedPassage] = []
    for score, row_id in zip(scores[0], indices[0]):
        # FAISS pads with -1 when it cannot fill the request. Indexing the span
        # rows with -1 would silently return the last chunk in the store as a
        # match, scored as if it were a real hit.
        if row_id < 0:
            continue
        row = rows[row_id]
        parent_id = row["parent_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        text = resolve_text(row, passages)
        # A chunk carrying its own text spans several passages (query_group), so
        # no single parent is its source; reporting a nominal parent would
        # understate it against the corpus relevance labels.
        source = text if row.get("text") is not None else passages[parent_id]["text"]
        found.append(
            RetrievedPassage(
                text=text,
                source_passage=source,
                is_selected=bool(passages[parent_id]["is_selected"]),
                score=float(score),
            )
        )
        if len(found) >= k:
            break

    return RetrievalOutput(query=query, strategy=strategy, passages=found)
