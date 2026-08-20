"""Measures each index's off-topic threshold and writes it into its manifest.

Replaces a hand-maintained dict of two thresholds. That dict is how a value
verified against the semantic index shipped as the fixed_size default and
refused 38.5% of real in-corpus questions; with eight strategies it would be
eight chances to forget one. After this runs, an uncalibrated index cannot be
served -- `app.guardrails.offtopic_threshold` raises instead of borrowing
another strategy's number.

The statistic is the one scripts/tune_thresholds.py already established: the
in-corpus p05 (which decides false refusals) against the off-topic p95 (which
decides leaks). The threshold is placed at the in-corpus p05, tuned for low
false refusal because a refused real question is the damaging, user-visible
failure, and off-topic input that slips through still faces the generation
prompt's decline sentinel and the groundedness guard.

Two processes, like the index build: embedding uses torch, searching uses faiss,
and on macOS arm64 a process holding both segfaults once MPS is used. Queries
are embedded ONCE and reused across all eight indices, which is both faster and
strictly more comparable than re-embedding per strategy.

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.calibrate_thresholds
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from app.strategies import chunk_paths, dense_names

CORPUS_PATH = "data/corpus.jsonl"
QUERY_VECTORS = "data/calibration_queries.f32"

# Deliberately far from a corpus of MS MARCO web-search results. Kept identical
# to scripts/tune_thresholds.py so the two measurements stay comparable.
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

# The p05 of the in-corpus distribution. Set here rather than inline so the
# choice is visible: a lower percentile refuses fewer real questions and leaks
# more, and the leak path has two further guards behind it.
IN_CORPUS_PERCENTILE = 5


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


def _phase_embed(sample: int, seed: int) -> None:
    """torch only, never imports faiss."""
    import numpy as np

    from app.embeddings import embed_batch

    rows = []
    with open(CORPUS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))
    queries = [r["query"] for r in picked] + OFF_TOPIC_QUERIES
    query_ids = [r["query_id"] for r in picked]

    vectors = np.asarray(embed_batch(queries), dtype="float32")
    Path(QUERY_VECTORS).write_bytes(np.ascontiguousarray(vectors).tobytes())
    Path(QUERY_VECTORS + ".meta.json").write_text(
        json.dumps(
            {
                "count": len(queries),
                "dimension": int(vectors.shape[1]),
                "in_corpus": len(picked),
                "off_topic": len(OFF_TOPIC_QUERIES),
                # Needed to exclude a query's own row when calibrating a
                # query-enriched index -- see _held_out_scores.
                "query_ids": query_ids,
            }
        )
    )
    print(f"embedded {len(picked)} corpus queries + {len(OFF_TOPIC_QUERIES)} probes")


def _held_out_scores(index, chunk_rows, passages, vectors, query_ids, depth=64):
    """Best score per query, ignoring chunks enriched with that same query.

    query_aware bakes each passage's own gold query into its vector, so an
    in-corpus query matches its own row almost perfectly. Calibrating on that
    measures self-reference, not retrieval: the resulting threshold (0.722,
    against 0.554-0.574 for every other strategy) would refuse most genuinely
    unseen questions in production.

    Skipping the query's own row measures the case that actually ships -- a
    question whose passages were enriched with *some other* question -- without
    rebuilding the index.
    """
    scores, ids = index.search(vectors, depth)
    out = []
    for row_scores, row_ids, query_id in zip(scores, ids, query_ids):
        for score, row_id in zip(row_scores, row_ids):
            if row_id < 0:
                continue
            if passages[chunk_rows[row_id]["parent_id"]]["query_id"] == query_id:
                continue
            out.append(float(score))
            break
    return out


def _phase_measure(write: bool) -> int:
    """faiss only, never imports torch."""
    import faiss
    import numpy as np

    meta = json.loads(Path(QUERY_VECTORS + ".meta.json").read_text())
    count, dim = meta["count"], meta["dimension"]
    n_corpus = meta["in_corpus"]
    vectors = np.frombuffer(Path(QUERY_VECTORS).read_bytes(), dtype="float32").reshape(
        count, dim
    )
    in_corpus_q, off_topic_q = vectors[:n_corpus], vectors[n_corpus:]

    print(
        f"\n{'strategy':<17}{'p05':>7}{'median':>8}{'off p95':>9}{'off max':>9}"
        f"{'chosen':>8}{'refuse%':>9}{'leaks':>7}"
    )
    print("-" * 74)
    failures = 0
    for name in dense_names():
        index_path, _ = chunk_paths(name)
        if not Path(index_path).exists():
            print(f"{name:<17}  index missing -- build it first")
            failures += 1
            continue
        index = faiss.read_index(index_path)
        off_scores = [float(s[0]) for s in index.search(off_topic_q, 1)[0]]

        # A query-enriched index must be calibrated against held-out queries or
        # the threshold measures self-reference and refuses real traffic.

        # Only the enriched index needs a search-time exclusion. The control
        # index (query_aware_heldout) is held out by construction -- its
        # evaluated rows carry no query -- so filtering it as well would drop
        # the very passages the measurement is about.
        held_out = name == "query_aware"
        if held_out:
            import pickle

            from app.passages import load_passage_store

            _, metadata_path = chunk_paths(name)
            chunk_rows = pickle.loads(Path(metadata_path).read_bytes())
            passages = load_passage_store("data/passages.pkl")
            in_scores = _held_out_scores(
                index, chunk_rows, passages, in_corpus_q, meta["query_ids"]
            )
        else:
            in_scores = [float(s[0]) for s in index.search(in_corpus_q, 1)[0]]

        chosen = percentile(in_scores, IN_CORPUS_PERCENTILE)
        refused = sum(1 for s in in_scores if s < chosen) / len(in_scores)
        leaks = sum(1 for s in off_scores if s >= chosen)
        print(
            f"{name:<17}{chosen:>7.3f}{statistics.median(in_scores):>8.3f}"
            f"{percentile(off_scores, 95):>9.3f}{max(off_scores):>9.3f}"
            f"{chosen:>8.3f}{100 * refused:>8.1f}%{leaks:>7}"
            + ("  held-out" if held_out else "")
        )
        if write:
            manifest = Path(index_path).with_suffix(".manifest.json")
            data = json.loads(manifest.read_text())
            data["offtopic_threshold"] = round(chosen, 4)
            data["calibration"] = {
                "held_out": held_out,
                "in_corpus_queries": len(in_scores),
                "in_corpus_p05": round(chosen, 4),
                "in_corpus_median": round(statistics.median(in_scores), 4),
                "off_topic_p95": round(percentile(off_scores, 95), 4),
                "off_topic_max": round(max(off_scores), 4),
                "false_refusal_rate": round(refused, 4),
                "off_topic_leaks": leaks,
            }
            manifest.write_text(json.dumps(data, indent=2))
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("embed", "measure"))
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="measure, do not write")
    args = parser.parse_args()

    if args.phase == "embed":
        _phase_embed(args.sample, args.seed)
        sys.exit(0)
    if args.phase == "measure":
        sys.exit(1 if _phase_measure(not args.dry_run) else 0)

    env = {**os.environ}
    for phase in ("embed", "measure"):
        cmd = [sys.executable, "-m", "scripts.calibrate_thresholds", "--phase", phase]
        if phase == "embed":
            cmd += ["--sample", str(args.sample), "--seed", str(args.seed)]
        elif args.dry_run:
            cmd += ["--dry-run"]
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"\n{phase} phase failed", file=sys.stderr)
            sys.exit(result.returncode)
    Path(QUERY_VECTORS).unlink(missing_ok=True)
    Path(QUERY_VECTORS + ".meta.json").unlink(missing_ok=True)
    print("\nthresholds written into each index manifest.")
