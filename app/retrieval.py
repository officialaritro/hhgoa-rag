"""Given a text query, embed it and search a chosen chunking strategy's
FAISS index for the most relevant passages (plan Task 4).
"""

from functools import cache

import numpy as np

from app.embeddings import embed
from app.indexing import load_index
from app.schemas import RetrievalOutput, RetrievedPassage

INDEX_PATHS = {
    "fixed_size": ("data/index_fixed_size.faiss", "data/metadata_fixed_size.pkl"),
    "semantic": ("data/index_semantic.faiss", "data/metadata_semantic.pkl"),
}


@cache
def _load_cached(strategy: str):
    """Loads each strategy's index+metadata from disk once per process and
    keeps it resident -- reloading (and re-unpickling metadata) on every
    request blew both the latency budget and memory (repeated allocation
    churn) on the t3.small the app is deployed on."""
    index_path, metadata_path = INDEX_PATHS[strategy]
    return load_index(index_path, metadata_path)


def retrieve(query: str, strategy: str, k: int = 5) -> RetrievalOutput:
    index, metadata = _load_cached(strategy)

    query_vector = np.array([embed(query)], dtype="float32")
    scores, indices = index.search(query_vector, k)

    passages = [
        RetrievedPassage(
            text=metadata[idx]["text"],
            source_passage=metadata[idx]["source_passage"],
            is_selected=bool(metadata[idx]["is_selected"]),
            score=float(score),
        )
        for score, idx in zip(scores[0], indices[0])
    ]

    return RetrievalOutput(query=query, strategy=strategy, passages=passages)
