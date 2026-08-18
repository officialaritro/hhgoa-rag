"""Recall@k evaluation of retrieval quality against the dataset's own
is_selected relevance labels (plan Task 4 / PRD Acceptance Criteria 3).
"""

import json
import random
from typing import Any

from app.retrieval import retrieve

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 42


def load_eval_queries(
    corpus_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(corpus_path) as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return rows[:sample_size]


def recall_at_k(
    strategy: str,
    corpus_path: str = "data/corpus.jsonl",
    k: int = 5,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> float:
    """For each sampled query with at least one is_selected passage, checks
    whether that passage's text appears among the top-k retrieved chunks'
    source_passage values."""
    rows = load_eval_queries(corpus_path, sample_size)
    evaluated = 0
    hits = 0
    for row in rows:
        selected_texts = {p["text"] for p in row["passages"] if p["is_selected"]}
        if not selected_texts:
            continue
        evaluated += 1
        result = retrieve(query=row["query"], strategy=strategy, k=k)
        retrieved_source_passages = {p.source_passage for p in result.passages}
        if selected_texts & retrieved_source_passages:
            hits += 1
    return hits / evaluated if evaluated else 0.0


if __name__ == "__main__":
    for strategy in ("fixed_size", "semantic"):
        score = recall_at_k(strategy)
        print(f"{strategy}: recall@5 = {score:.3f}")
