"""Local, in-process embedding model -- no hosted embeddings API on the
critical path (plan Global Constraints). Used by retrieval (Task 4) and the
groundedness guardrail (Task 7).

Pulled forward from Task 3: this module has no dependency on the ingested
dataset, unlike chunk_semantic.py/indexing.py/build_index.py which are still
blocked pending disk space for the corpus download.
"""

import os
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("EMBEDDING_MODEL_NAME", _DEFAULT_MODEL_NAME)
    return SentenceTransformer(model_name)


def embed(text: str) -> NDArray[np.float32]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
    """Batched embedding for bulk indexing (Task 3's build_index).
    Per-call model overhead makes embedding texts one at a time far slower
    than batching -- measured ~354 texts/sec batched vs. a small fraction of
    that one at a time on this hardware, discovered when build_index took
    far longer than the batched-throughput estimate projected."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=batch_size)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Both inputs are assumed L2-normalized (embed() normalizes), so the
    dot product equals cosine similarity."""
    return float(np.dot(a, b))
