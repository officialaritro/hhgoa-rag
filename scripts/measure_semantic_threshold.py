"""Measures the adjacent-sentence similarity distribution the semantic chunker
splits on, so its threshold is set from data rather than argued from intuition.

The shipped value of 0.8 was justified by the similarity of *unrelated*
sentences (commonly 0.3-0.6). That reasoning does not apply: adjacent sentences
inside one web passage are usually related, so 0.8 sat above nearly the whole
distribution and the chunker almost never merged -- it emitted 338,544 chunks
against 349,983 sentences, reproducing re.split at the cost of one embedding
call per sentence.

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.measure_semantic_threshold
"""

import argparse
import random
import statistics

from app.chunkers import sentence_spans
from app.embeddings import cosine_similarity, embed_batch
from app.passages import load_passage_store

PASSAGE_STORE = "data/passages.pkl"


def measure(sample: int, seed: int) -> list[float]:
    passages = load_passage_store(PASSAGE_STORE)
    picked = random.Random(seed).sample(passages, min(sample, len(passages)))
    similarities: list[float] = []
    for passage in picked:
        spans = sentence_spans(passage["text"])
        if len(spans) < 2:
            continue
        sentences = [passage["text"][s:e] for s, e in spans]
        vectors = embed_batch(sentences)
        similarities.extend(
            cosine_similarity(vectors[i - 1], vectors[i])
            for i in range(1, len(vectors))
        )
    return similarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    values = sorted(measure(args.sample, args.seed))
    n = len(values)

    def pct(p: float) -> float:
        return values[min(n - 1, int(p / 100 * n))]

    print(f"\nadjacent-sentence cosine, {n:,} pairs from {args.sample:,} passages")
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{p:<2} {pct(p):.3f}")
    print(f"  mean {statistics.mean(values):.3f}   max {values[-1]:.3f}")

    print("\nmerge rate by threshold (share of adjacent pairs that would merge):")
    for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        merged = sum(1 for v in values if v >= t) / n
        # Each merge removes one chunk; sentences/chunk = 1/(1-merge_rate).
        per_chunk = 1 / (1 - merged) if merged < 1 else float("inf")
        print(f"  {t:.2f}  merge {merged:6.1%}   ~{per_chunk:.2f} sentences/chunk")

    print(f"\nmedian of the distribution = {pct(50):.3f}")
    print("A threshold at the median merges about half of adjacent pairs, giving")
    print("~2 sentences per chunk -- genuinely between whole_passage and")
    print("sentence_window, which is the gap the shipped 0.8 left empty.")
