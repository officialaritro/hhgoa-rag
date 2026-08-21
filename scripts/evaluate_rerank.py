"""Measures what a cross-encoder reranker buys, and at what depth.

The case for reranking is in docs/CHUNKING_REPORT.md: recall@1 is 0.410 while
recall@10 is 0.960. The relevant passage is almost always among the candidates
and is simply not ranked first. That is an ordering problem, and a bi-encoder
cannot fix it -- it scores query and passage in isolation, so it never sees them
together. A cross-encoder does.

Depth is a budget decision, not a free parameter. Measured cross-encoder latency
on CPU, which is what the instance runs: 36ms at 10 candidates, 66ms at 20,
164ms at 50, against 70ms already owned by retrieval, query embedding and the
groundedness guard, inside a 200ms target. So this reports quality at each depth
and the choice follows from both columns.

Two phases, because retrieval needs faiss and the reranker needs torch:
    candidates  faiss only -- top 50 per query, with labels
    rerank      torch only -- score, reorder, recompute recall

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.evaluate_rerank
"""

import argparse
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORK = "data/rerank_candidates.json"
EVAL_VECTORS = "data/eval_queries.f32"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_DEPTH = 50


def _phase_candidates(strategy: str) -> None:
    """faiss only."""
    import faiss
    import numpy as np

    from app.passages import load_passage_store, resolve_text
    from app.strategies import chunk_paths

    meta = json.loads(Path(EVAL_VECTORS + ".meta.json").read_text())
    count, dim = meta["count"], meta["dimension"]
    vectors = np.frombuffer(Path(EVAL_VECTORS).read_bytes(), dtype="float32").reshape(
        count, dim
    )
    passages = load_passage_store("data/passages.pkl")
    index_path, metadata_path = chunk_paths(strategy)
    index = faiss.read_index(index_path)
    rows = pickle.loads(Path(metadata_path).read_bytes())

    items = []
    for i in range(count):
        _, ids = index.search(vectors[i : i + 1], MAX_DEPTH * 4)
        seen: set[int] = set()
        candidates: list[str] = []
        for row_id in ids[0]:
            if row_id < 0:
                continue
            parent = rows[row_id]["parent_id"]
            if parent in seen:
                continue
            seen.add(parent)
            candidates.append(resolve_text(rows[row_id], passages))
            if len(candidates) >= MAX_DEPTH:
                break
        items.append(
            {
                "query": meta["queries"][i],
                "candidates": candidates,
                "labels": meta["labels"][i],
            }
        )
    Path(WORK).write_text(json.dumps(items))
    print(
        f"retrieved up to {MAX_DEPTH} candidates for {len(items)} queries ({strategy})"
    )


def _recall(ranked: list[list[str]], labels: list[list[str]], k: int) -> float:
    hits = 0
    for sources, labelled in zip(ranked, labels):
        if any(any(lab == s or lab in s for lab in labelled) for s in sources[:k]):
            hits += 1
    return hits / len(ranked) if ranked else 0.0


def _hits(ranked: list[list[str]], labels: list[list[str]], k: int) -> list[int]:
    """Per-query outcomes, for the paired bootstrap. The +6.8pp headline needs an
    interval, not a point estimate -- that is how the fusion result went wrong."""
    return [
        1
        if any(any(lab == s or lab in s for lab in labelled) for s in sources[:k])
        else 0
        for sources, labelled in zip(ranked, labels)
    ]


def _mrr(ranked: list[list[str]], labels: list[list[str]], k: int) -> float:
    total = 0.0
    for sources, labelled in zip(ranked, labels):
        for position, source in enumerate(sources[:k], start=1):
            if any(lab == source or lab in source for lab in labelled):
                total += 1 / position
                break
    return total / len(ranked) if ranked else 0.0


def _phase_rerank(depths: list[int], device: str) -> None:
    """torch only."""
    from sentence_transformers import CrossEncoder

    items = json.loads(Path(WORK).read_text())
    labels = [i["labels"] for i in items]
    baseline = [i["candidates"] for i in items]

    encoder = CrossEncoder(MODEL, max_length=512, device=device)
    encoder.predict([(items[0]["query"], items[0]["candidates"][0])])  # warm

    print(f"\nbaseline (dense order), {len(items)} queries")
    print(
        f"  recall@1 {_recall(baseline, labels, 1):.3f}  "
        f"recall@5 {_recall(baseline, labels, 5):.3f}  "
        f"MRR@10 {_mrr(baseline, labels, 10):.3f}"
    )

    print(f"\nreranked with {MODEL}, device={device}")
    print(
        f"{'depth':>6}{'recall@1':>11}{'recall@5':>10}{'MRR@10':>9}{'rerank P50':>13}"
    )
    print("-" * 49)
    results = []
    for depth in depths:
        reordered = []
        latencies = []
        for item in items:
            pool = item["candidates"][:depth]
            if not pool:
                reordered.append([])
                continue
            start = time.perf_counter()
            scores = encoder.predict([(item["query"], c) for c in pool])
            latencies.append((time.perf_counter() - start) * 1000)
            order = sorted(range(len(pool)), key=lambda j: -scores[j])
            reordered.append([pool[j] for j in order])
        row = {
            "depth": depth,
            "recall@1": _recall(reordered, labels, 1),
            "recall@5": _recall(reordered, labels, 5),
            "mrr@10": _mrr(reordered, labels, 10),
            "rerank_ms_p50": statistics.median(latencies),
            "hits@5": _hits(reordered, labels, 5),
            "hits@1": _hits(reordered, labels, 1),
        }
        results.append(row)
        print(
            f"{depth:>6}{row['recall@1']:>11.3f}{row['recall@5']:>10.3f}"
            f"{row['mrr@10']:>9.3f}{row['rerank_ms_p50']:>11.1f} ms"
        )

    from scripts.significance import describe, paired_bootstrap

    print(f"\n{'depth':>6}  paired 95% CI on recall@5 against dense order")
    print("-" * 56)
    for row in results:
        low, high = paired_bootstrap(_hits(baseline, labels, 5), row["hits@5"], seed=0)
        print(f"{row['depth']:>6}  {low:+.3f} to {high:+.3f}   {describe(low, high)}")

    best = max(results, key=lambda r: r["recall@1"])
    lift = best["recall@1"] - _recall(baseline, labels, 1)
    print(
        f"\nbest recall@1 {best['recall@1']:.3f} at depth {best['depth']} "
        f"({lift:+.3f} against dense order, {100 * lift / max(1e-9, _recall(baseline, labels, 1)):+.0f}%)"
    )
    Path("data/rerank_results.json").write_text(
        json.dumps(
            {
                "queries": len(items),
                "device": device,
                "baseline_recall@1": _recall(baseline, labels, 1),
                "baseline_recall@5": _recall(baseline, labels, 5),
                "baseline_mrr@10": _mrr(baseline, labels, 10),
                "baseline_hits@5": _hits(baseline, labels, 5),
                "baseline_hits@1": _hits(baseline, labels, 1),
                "reranked": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("candidates", "rerank"))
    parser.add_argument("--strategy", default="whole_passage")
    parser.add_argument("--depths", default="5,10,20,50")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    depths = [int(d) for d in args.depths.split(",")]

    if args.phase == "candidates":
        _phase_candidates(args.strategy)
        sys.exit(0)
    if args.phase == "rerank":
        _phase_rerank(depths, args.device)
        sys.exit(0)

    for phase in ("candidates", "rerank"):
        cmd = [sys.executable, "-m", "scripts.evaluate_rerank", "--phase", phase]
        if phase == "candidates":
            cmd += ["--strategy", args.strategy]
        else:
            cmd += ["--depths", args.depths, "--device", args.device]
        if subprocess.run(cmd, env={**os.environ}, check=False).returncode != 0:
            sys.exit(1)
