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
from app.fusion import reciprocal_rank_fusion
from app.indexing import load_index
from app.lexical import BM25Index
from app.passages import load_passage_store, resolve_text
from app.schemas import RetrievalOutput, RetrievedPassage
from app.strategies import chunk_paths, dense_names, get

PASSAGE_STORE_PATH = "data/passages.pkl"
BM25_PATH = "data/bm25_passages.pkl"

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
def _load_bm25() -> BM25Index:
    """The lexical index, loaded once per process like the FAISS ones."""
    return BM25Index.load(BM25_PATH)


@cache
def _load_passages():
    """One passage store, shared by every strategy. Loaded once per process:
    ~36 MB for 99,767 passages, against which a per-request load would be
    absurd."""
    return load_passage_store(PASSAGE_STORE_PATH)


def _dense_parent_ranking(
    query_vector, strategy: str, depth: int
) -> tuple[list[int], dict[int, float]]:
    """Parent ids from one dense index, best first, plus each parent's best
    cosine. Collapsed by parent so the ranking RRF sees is a ranking of
    passages -- fusing rankings of different chunk sets would compare
    incomparable positions."""
    index, rows = _load_cached(strategy)
    scores, indices = index.search(query_vector, depth)
    ordered: list[int] = []
    best: dict[int, float] = {}
    for score, row_id in zip(scores[0], indices[0]):
        if row_id < 0:
            continue
        parent_id = rows[row_id]["parent_id"]
        if parent_id in best:
            continue
        best[parent_id] = float(score)
        ordered.append(parent_id)
    return ordered, best


def _compose(query: str, strategy: str, k: int) -> RetrievalOutput:
    """hybrid and fusion. Both merge parent-level rankings with RRF.

    The reported score stays a dense cosine rather than the RRF score. RRF
    scores land around 0.03 and are not on the cosine scale the off-topic guard
    is calibrated against (0.499-0.574 measured), so reporting one would refuse
    every query. Ranking and gating answer different questions: RRF decides which
    passage is best, the cosine decides whether anything is close enough at all.
    """
    spec = get(strategy)
    depth = k * OVERFETCH
    query_vector = np.array([embed(query)], dtype="float32")
    passages = _load_passages()

    rankings: list[list[int]] = []
    cosines: dict[int, float] = {}
    for member in spec.members:
        ordered, best = _dense_parent_ranking(query_vector, member, depth)
        rankings.append(ordered)
        for parent_id, score in best.items():
            cosines[parent_id] = max(cosines.get(parent_id, -1.0), score)

    if spec.kind == "hybrid":
        # Lexical abstains rather than vetoing: a voice transcript often shares
        # no exact term with the passage that answers it, and an empty ranking
        # contributes nothing to the fusion instead of blocking it.
        rankings.append([pid for pid, _ in _load_bm25().top_k(query, depth)])

    fused = reciprocal_rank_fusion(rankings)
    found = [
        RetrievedPassage(
            text=passages[parent_id]["text"],
            source_passage=passages[parent_id]["text"],
            is_selected=bool(passages[parent_id]["is_selected"]),
            # A lexical-only hit has no cosine of its own; 0.0 keeps it ranked by
            # RRF while never letting it raise the guard's top score.
            score=cosines.get(parent_id, 0.0),
        )
        for parent_id, _ in fused[:k]
    ]
    return RetrievalOutput(query=query, strategy=strategy, passages=found)


def retrieve(query: str, strategy: str, k: int = 5) -> RetrievalOutput:
    spec = get(strategy)  # raises UnknownStrategy with the valid list, before I/O
    if spec.kind != "dense":
        return _compose(query, strategy, k)
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
