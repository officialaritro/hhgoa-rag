"""Trains and validates the off-topic classifier, and compares it honestly.

The guard it replaces is a single threshold on top-1 cosine, measured per index.
That signal provably cannot separate the two classes -- across all nine indices
the in-corpus and off-topic distributions overlap -- so the threshold has always
been a trade between refusing real questions and admitting unanswerable ones.

This fits a logistic regression over the score-profile features in
app/offtopic_features.py, using scale-invariant ones so a single model serves
every index rather than nine hand-measured thresholds.

Validation is grouped and cross-validated. Off-topic probes come in categories
(private data, commands, gibberish, ...) and probes within a category are near
duplicates, so a random split would put near-twins on both sides and report a
score that will not survive an unseen *kind* of off-topic input. Splitting by
category answers the question that matters: does this generalise to a failure
mode it has never seen?

Two phases, as everywhere here: embedding needs torch, searching needs faiss.

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.train_offtopic
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

from app.offtopic_probes import labelled_probes
from app.strategies import chunk_paths, served_names

WORK = "data/offtopic_training.json"
MODEL_PATH = "data/offtopic_classifier.pkl"
CORPUS_PATH = "data/corpus.jsonl"


def _phase_embed(sample: int, seed: int) -> None:
    """torch only."""
    import numpy as np

    from app.embeddings import embed_batch

    rows = []
    with open(CORPUS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))
    in_corpus = [r["query"] for r in picked]
    probes = labelled_probes()

    queries = in_corpus + [q for q, _ in probes]
    vectors = np.asarray(embed_batch(queries), dtype="float32")
    Path("data/offtopic_queries.f32").write_bytes(
        np.ascontiguousarray(vectors).tobytes()
    )
    Path(WORK).write_text(
        json.dumps(
            {
                "count": len(queries),
                "dimension": int(vectors.shape[1]),
                "in_corpus": in_corpus,
                "probes": [{"query": q, "category": c} for q, c in probes],
            }
        )
    )
    print(f"embedded {len(in_corpus)} in-corpus queries + {len(probes)} probes")


def _phase_train(strategy: str) -> None:
    """faiss only."""
    import faiss
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from app.offtopic_features import FEATURE_NAMES, extract_features

    meta = json.loads(Path(WORK).read_text())
    count, dim = meta["count"], meta["dimension"]
    vectors = np.frombuffer(
        Path("data/offtopic_queries.f32").read_bytes(), dtype="float32"
    ).reshape(count, dim)
    in_corpus = meta["in_corpus"]
    probes = meta["probes"]

    index_path, metadata_path = chunk_paths(strategy)
    index = faiss.read_index(index_path)
    rows = pickle.loads(Path(metadata_path).read_bytes())

    def profile(i: int) -> list[float]:
        scores, ids = index.search(vectors[i : i + 1], 20)
        seen: set[int] = set()
        kept: list[float] = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0:
                continue
            parent = rows[row_id]["parent_id"]
            if parent in seen:
                continue
            seen.add(parent)
            kept.append(float(score))
        return kept

    features, labels, groups = [], [], []
    for i, query in enumerate(in_corpus):
        features.append(extract_features(profile(i), query))
        labels.append(0)
        groups.append("in_corpus")
    for j, probe in enumerate(probes):
        features.append(extract_features(profile(len(in_corpus) + j), probe["query"]))
        labels.append(1)
        groups.append(probe["category"])

    X = np.asarray(features, dtype="float64")
    y = np.asarray(labels)
    categories = sorted({g for g in groups if g != "in_corpus"})

    print(
        f"\ntrained on {strategy}: {int((y == 0).sum())} in-corpus, "
        f"{int((y == 1).sum())} off-topic across {len(categories)} categories"
    )

    # Leave-one-category-out. A random split would put near-duplicate probes on
    # both sides and flatter the model.
    print("\nleave-one-category-out (does it generalise to an UNSEEN kind?)")
    print(f"{'held-out category':<18}{'probes':>7}{'caught':>8}{'false refuse':>14}")
    print("-" * 47)
    caught_total = held_total = 0
    for category in categories:
        test_mask = np.asarray([g == category for g in groups])
        train_mask = ~test_mask
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_mask], y[train_mask])
        predicted = model.predict(X[test_mask])
        caught = int(predicted.sum())
        held = int(test_mask.sum())
        caught_total += caught
        held_total += held
        # False refusals measured on in-corpus queries this fold did train on,
        # which is the operating point that ships.
        in_mask = np.asarray([g == "in_corpus" for g in groups])
        refused = float(model.predict(X[in_mask]).mean())
        print(f"{category:<18}{held:>7}{caught:>8}{100 * refused:>13.1f}%")
    print(f"{'TOTAL':<18}{held_total:>7}{caught_total:>8}")

    # Baseline: the single manifest threshold on top-1, same data.
    manifest = json.loads(Path(index_path).with_suffix(".manifest.json").read_text())
    threshold = manifest["offtopic_threshold"]
    top1 = X[:, FEATURE_NAMES.index("top1")]
    in_mask = y == 0
    off_mask = y == 1
    baseline_refuse = float((top1[in_mask] < threshold).mean())
    baseline_caught = int((top1[off_mask] < threshold).sum())

    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    final.fit(X, y)
    clf_refuse = float(final.predict(X[in_mask]).mean())
    clf_caught = int(final.predict(X[off_mask]).sum())

    print(f"\n{'approach':<34}{'off-topic caught':>18}{'false refusals':>16}")
    print("-" * 68)
    print(
        f"{'top-1 threshold ' + f'({threshold:.3f})':<34}"
        f"{baseline_caught:>10}/{int(off_mask.sum())}{100 * baseline_refuse:>15.1f}%"
    )
    print(
        f"{'classifier, leave-one-category-out':<34}"
        f"{caught_total:>10}/{held_total}{'':>16}"
    )
    print(
        f"{'classifier, refit on all':<34}"
        f"{clf_caught:>10}/{int(off_mask.sum())}{100 * clf_refuse:>15.1f}%"
    )

    weights = final[-1].coef_[0]
    print("\nfeature weights (positive pushes toward off-topic):")
    for name, weight in sorted(zip(FEATURE_NAMES, weights), key=lambda p: -abs(p[1])):
        print(f"  {name:<18}{weight:+.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {"model": final, "features": list(FEATURE_NAMES), "strategy": strategy},
            f,
            protocol=4,
        )
    print(f"\nwrote {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("embed", "train"))
    parser.add_argument("--strategy", default="whole_passage")
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if args.phase == "embed":
        _phase_embed(args.sample, args.seed)
        sys.exit(0)
    if args.phase == "train":
        _phase_train(args.strategy)
        sys.exit(0)

    assert args.strategy in served_names(), f"{args.strategy} is not served"
    for phase in ("embed", "train"):
        cmd = [sys.executable, "-m", "scripts.train_offtopic", "--phase", phase]
        cmd += (
            ["--sample", str(args.sample), "--seed", str(args.seed)]
            if phase == "embed"
            else ["--strategy", args.strategy]
        )
        if subprocess.run(cmd, env={**os.environ}, check=False).returncode != 0:
            sys.exit(1)
