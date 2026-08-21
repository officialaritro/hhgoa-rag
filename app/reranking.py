"""Cross-encoder reranking of retrieved candidates.

The measured case for this, on 500 labelled queries over whole passages:

    recall@1   0.410 -> 0.504   (+23%)
    recall@5   0.848 -> 0.916   (+8%)
    MRR@10     0.591 -> 0.671   (+14%)

That is a larger improvement than every chunking strategy and both fusion modes
put together. `fusion` reached 0.854 recall@5, inside the ~1.6pp error bar of
plain dense retrieval; this is +6.8pp, well outside it.

The reason is visible in the baseline. Dense recall@10 was already 0.960, so the
relevant passage was nearly always retrieved and simply not ranked first. A
bi-encoder embeds query and passage independently and never compares them
directly; a cross-encoder reads them together, which is what an ordering problem
needs.

Depth is a budget decision. Measured on CPU, which is what the instance runs:
36ms at 10 candidates, 66ms at 20, 164ms at 50, against 70ms already owned by
retrieval, query embedding and the groundedness guard, inside a 200ms target.
Depth 10 also happens to score best -- 0.504 against 0.500 at both 20 and 50 --
so a deeper pool gives the model more chances to promote something wrong rather
than more chances to find the answer.
"""

import logging
import os
from functools import cache

from app.schemas import RetrievedPassage

logger = logging.getLogger("uvicorn.error")

MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Measured optimum on both axes: best recall@1 and cheapest. See module docstring.
RERANK_DEPTH = 10

_MAX_LENGTH = 512


@cache
def reranking_enabled() -> bool:
    """Whether to rerank. On by default, switchable without a code deploy.

    The instance has 2 vCPU against the 8 these latencies were measured on, so
    if reranking there breaks the budget it has to be possible to turn off with
    an environment variable and a restart.
    """
    return os.environ.get("RERANK_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


@cache
def _load_encoder():
    """Loaded once per process, like the FAISS indices and the passage store."""
    from sentence_transformers import CrossEncoder

    device = os.environ.get("RERANK_DEVICE") or None
    return CrossEncoder(MODEL_NAME, max_length=_MAX_LENGTH, device=device)


def rerank_passages(
    query: str, passages: list[RetrievedPassage], top_k: int
) -> list[RetrievedPassage]:
    """Reorders candidates by cross-encoder relevance, then truncates to top_k.

    Reorder before truncating: cutting first would discard exactly the passage
    the reranker was about to promote, which is the whole gain.

    `score` keeps each passage's dense cosine. Cross-encoder outputs are
    unbounded logits, and the off-topic guard's thresholds are calibrated on
    cosines (0.499-0.574 measured), so overwriting it would make every gate
    meaningless.

    A reranker is an improvement, not a dependency: if the model cannot load,
    this returns the dense order the service had before rather than failing the
    request.
    """
    if len(passages) < 2:
        return passages[:top_k] if passages else []

    candidates = passages[:RERANK_DEPTH]
    try:
        encoder = _load_encoder()
        scores = encoder.predict([(query, p.text) for p in candidates])
    except Exception:
        logger.exception("rerank: falling back to dense order")
        return passages[:top_k]

    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    reranked = [candidates[i] for i in order]
    # Anything past the rerank depth keeps its dense position behind the
    # reordered head, so a top_k larger than the depth still returns top_k.
    return (reranked + passages[RERANK_DEPTH:])[:top_k]
