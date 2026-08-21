"""Measures the groundedness guard against REAL generated answers.

The dataset's own terse `Eng_Answer` is a misleading proxy: a previous
calibration using it recommended ~0.02, because a one-line answer scores nothing
like the multi-sentence prose the model actually produces. So this generates
real answers and scores those.

Grounded and ungrounded examples come from the same answers: an answer paired
with the context it was written from is grounded by construction, and paired
with a different query's context it is not.

Three phases, because torch, faiss and the network cannot all share a process
here (faiss + torch co-loaded segfault once MPS is used):
    retrieve  faiss only  -- contexts for N queries
    generate  network only -- real answers for those contexts
    score     torch only  -- per-sentence support, then AUC per aggregation

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.calibrate_groundedness
"""

import argparse
import json
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path

WORK = "data/groundedness_probe.json"
EVAL_VECTORS = "data/eval_queries.f32"


def _phase_retrieve(sample: int) -> None:
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
    index_path, metadata_path = chunk_paths("whole_passage")
    index = faiss.read_index(index_path)
    rows = pickle.loads(Path(metadata_path).read_bytes())

    items = []
    for i in range(min(sample, count)):
        _, ids = index.search(vectors[i : i + 1], 20)
        seen, texts = set(), []
        for row_id in ids[0]:
            if row_id < 0:
                continue
            parent = rows[row_id]["parent_id"]
            if parent in seen:
                continue
            seen.add(parent)
            texts.append(resolve_text(rows[row_id], passages))
            if len(texts) >= 5:
                break
        items.append({"query": meta["queries"][i], "context": texts})
    Path(WORK).write_text(json.dumps(items))
    print(f"retrieved contexts for {len(items)} queries")


def _phase_generate() -> None:
    """network only -- no torch, no faiss."""
    from concurrent.futures import ThreadPoolExecutor

    from app.generation import generate_answer
    from app.schemas import RetrievalOutput, RetrievedPassage

    items = json.loads(Path(WORK).read_text())

    def answer_for(item):
        retrieval = RetrievalOutput(
            query=item["query"],
            strategy="whole_passage",
            passages=[
                RetrievedPassage(text=t, source_passage=t, is_selected=False, score=0.9)
                for t in item["context"]
            ],
        )
        result = generate_answer(query=item["query"], retrieval=retrieval)
        return result.value.answer if result.ok else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(answer_for, items))
    for item, answer in zip(items, answers):
        item["answer"] = answer
    Path(WORK).write_text(json.dumps(items))
    usable = sum(1 for a in answers if a)
    print(f"generated {usable}/{len(items)} answers")


def _auc(positive: list[float], negative: list[float]) -> float:
    """Probability a random grounded example outscores a random ungrounded one."""
    if not positive or not negative:
        return 0.0
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative
    )
    return wins / (len(positive) * len(negative))


def _phase_score() -> None:
    """torch only."""
    from app.groundedness import sentence_support, unsupported_numbers
    from app.schemas import RetrievalOutput, RetrievedPassage

    items = [i for i in json.loads(Path(WORK).read_text()) if i.get("answer")]

    def retrieval_of(context):
        return RetrievalOutput(
            query="q",
            strategy="whole_passage",
            passages=[
                RetrievedPassage(text=t, source_passage=t, is_selected=False, score=0.9)
                for t in context
            ],
        )

    aggregations = {
        "mean": statistics.mean,
        "min": min,
        "p25": lambda v: sorted(v)[max(0, int(0.25 * len(v)) - 1)],
        "median": statistics.median,
    }
    grounded: dict[str, list[float]] = {k: [] for k in aggregations}
    ungrounded: dict[str, list[float]] = {k: [] for k in aggregations}
    number_flags_grounded = 0

    for i, item in enumerate(items):
        own = sentence_support(item["answer"], retrieval_of(item["context"]))
        other = items[(i + len(items) // 2) % len(items)]["context"]
        mismatched = sentence_support(item["answer"], retrieval_of(other))
        if not own or not mismatched:
            continue
        for name, fn in aggregations.items():
            grounded[name].append(fn(own))
            ungrounded[name].append(fn(mismatched))
        if unsupported_numbers(item["answer"], " ".join(item["context"])):
            number_flags_grounded += 1

    print(f"\nscored {len(grounded['mean'])} answer pairs")
    print(
        f"{'aggregation':<12}{'grounded p05':>14}{'ungrounded p95':>16}{'AUC':>8}{'gap':>8}"
    )
    print("-" * 58)
    best = None
    for name in aggregations:
        pos, neg = sorted(grounded[name]), sorted(ungrounded[name])
        p05 = pos[max(0, int(0.05 * len(pos)) - 1)]
        p95 = neg[min(len(neg) - 1, int(0.95 * len(neg)))]
        auc = _auc(pos, neg)
        print(f"{name:<12}{p05:>14.3f}{p95:>16.3f}{auc:>8.3f}{p05 - p95:>8.3f}")
        if best is None or auc > best[1]:
            best = (name, auc, p05, p95)

    print(f"\nbest discriminator: {best[0]} (AUC {best[1]:.3f})")
    print(f"  grounded p05 {best[2]:.3f} vs ungrounded p95 {best[3]:.3f}")
    if best[2] > best[3]:
        print(
            f"  a threshold anywhere in ({best[3]:.3f}, {best[2]:.3f}) separates cleanly"
        )
        print(f"  suggested: {(best[2] + best[3]) / 2:.2f}")
    else:
        print("  ! distributions overlap; no clean cut exists")
    print(
        f"\nliteral number check flagged {number_flags_grounded}/"
        f"{len(items)} genuinely grounded answers (false positives)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("retrieve", "generate", "score"))
    parser.add_argument("--sample", type=int, default=40)
    args = parser.parse_args()

    if args.phase:
        {
            "retrieve": lambda: _phase_retrieve(args.sample),
            "generate": _phase_generate,
            "score": _phase_score,
        }[args.phase]()
        sys.exit(0)

    for phase in ("retrieve", "generate", "score"):
        cmd = [sys.executable, "-m", "scripts.calibrate_groundedness", "--phase", phase]
        if phase == "retrieve":
            cmd += ["--sample", str(args.sample)]
        if subprocess.run(cmd, env={**os.environ}, check=False).returncode != 0:
            sys.exit(1)
