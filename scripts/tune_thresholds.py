"""Empirically tune the two guardrail thresholds in app/guardrails.py.

Both thresholds started as documented guesses (0.3 off-topic, 0.5
groundedness). This measures the score distributions they are supposed to
separate and reports a threshold placed between them, so the values in
app/guardrails.py can be set from data rather than intuition.

No LLM calls: the "grounded" and "ungrounded" answer distributions are built
from the dataset's own Eng_Answer field -- an answer paired with its own
query's retrieved context is grounded by construction; the same answer paired
with a different query's context is not.

Run on the instance after the indices exist:
    .venv/bin/python -m scripts.tune_thresholds
"""

import argparse
import json
import random
import statistics
from typing import Any

from app.embeddings import cosine_similarity, embed
from app.retrieval import retrieve

# Deliberately far from a passage-retrieval corpus built out of MS MARCO web
# search results -- these should score low if the off-topic guard works.
OFF_TOPIC_QUERIES = [
    "what is my bank account password",
    "sing me a lullaby in Portuguese",
    "ਮੈਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਇੱਕ ਕਹਾਣੀ ਸੁਣਾਓ",
    "what am I thinking about right now",
    "book me a flight to Reykjavik tomorrow morning",
    "qwertyuiop asdfghjkl zxcvbnm",
    "who won the 2047 world cup final",
    "please delete all my files",
]


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]

    return {
        "min": ordered[0],
        "p05": pct(5),
        "p50": pct(50),
        "p95": pct(95),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def _report(name: str, stats: dict[str, float]) -> None:
    if not stats:
        print(f"  {name}: no samples")
        return
    print(
        f"  {name:<26} min {stats['min']:.3f}  p05 {stats['p05']:.3f}  "
        f"p50 {stats['p50']:.3f}  p95 {stats['p95']:.3f}  max {stats['max']:.3f}"
    )


def _recommend(on: list[float], off: list[float]) -> float | None:
    """Place the threshold midway between the off-topic p95 and the in-corpus
    p05 -- the two values that actually decide false accepts and false
    rejects. Overlapping distributions mean no clean cut exists; the midpoint
    is reported anyway, with the overlap called out."""
    if not on or not off:
        return None
    on_low = _percentiles(on)["p05"]
    off_high = _percentiles(off)["p95"]
    if off_high >= on_low:
        print(
            f"  ! distributions overlap (off-topic p95 {off_high:.3f} >= "
            f"in-corpus p05 {on_low:.3f}) -- no threshold separates them cleanly"
        )
    return (off_high + on_low) / 2


def tune(corpus_path: str, strategy: str, sample: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    with open(corpus_path) as f:
        for line in f:
            rows.append(json.loads(line))
    picked = rng.sample(rows, min(sample, len(rows)))

    print(f"\nstrategy={strategy}  corpus_rows={len(rows)}  sampled={len(picked)}\n")

    print("OFF-TOPIC THRESHOLD (top retrieval score)")
    in_corpus = [retrieve(r["query"], strategy, k=5).passages[0].score for r in picked]
    off_topic = [
        retrieve(q, strategy, k=5).passages[0].score for q in OFF_TOPIC_QUERIES
    ]
    _report("in-corpus queries", _percentiles(in_corpus))
    _report("off-topic queries", _percentiles(off_topic))
    off_rec = _recommend(in_corpus, off_topic)

    print("\nGROUNDEDNESS THRESHOLD (answer vs retrieved context)")
    grounded: list[float] = []
    ungrounded: list[float] = []
    for i, row in enumerate(picked):
        answer = (row.get("answer") or "").strip()
        if not answer:
            continue
        ctx = " ".join(p.text for p in retrieve(row["query"], strategy, k=5).passages)
        grounded.append(cosine_similarity(embed(answer), embed(ctx)))
        # Same answer, a different query's context -> ungrounded by construction.
        other = picked[(i + len(picked) // 2) % len(picked)]
        other_ctx = " ".join(
            p.text for p in retrieve(other["query"], strategy, k=5).passages
        )
        ungrounded.append(cosine_similarity(embed(answer), embed(other_ctx)))
    _report("grounded (own context)", _percentiles(grounded))
    _report("ungrounded (other context)", _percentiles(ungrounded))
    ground_rec = _recommend(grounded, ungrounded)

    print("\nRECOMMENDED (set these in app/guardrails.py)")
    if off_rec is not None:
        print(f"  _DEFAULT_OFFTOPIC_THRESHOLD     = {off_rec:.2f}")
    if ground_rec is not None:
        print(f"  _DEFAULT_GROUNDEDNESS_THRESHOLD = {ground_rec:.2f}")
    return {"off_topic": off_rec, "groundedness": ground_rec}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--strategy", default="fixed_size")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tune(args.corpus, args.strategy, args.sample, args.seed)
