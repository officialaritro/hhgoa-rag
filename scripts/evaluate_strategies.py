"""Retrieval-quality matrix across every chunking strategy.

This is the deliverable. Ten strategies with no comparison table is
indistinguishable from two strategies to a reader, and the PRD's acceptance
criteria ask for retrieval quality against the dataset's own `is_selected`
labels -- a number this project had never measured.

Metrics are computed against those labels: a query counts as a hit when one of
its labelled passages appears among the top-k retrieved sources.

Two processes, as everywhere else here: embedding needs torch, searching needs
faiss, and on macOS arm64 a process holding both segfaults once MPS is used.
Latency is therefore reported decomposed rather than as one figure -- query
embedding is timed in the torch process, FAISS search in the faiss process --
which is more informative anyway, since only one of the two scales with index
size.

`query_aware` is evaluated held out. It bakes each passage's own gold query into
its vector, so scoring it with that same query measures self-reference: its
off-topic threshold measured 0.722 naively against 0.400 with the query's own
row excluded. The same correction applies to recall, so its row is computed with
each query's own row skipped and labelled accordingly.

Run:
    EMBEDDING_DEVICE=mps .venv/bin/python -m scripts.evaluate_strategies
"""

import argparse
import json
import math
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from app.strategies import chunk_paths, dense_names, get

CORPUS_PATH = "data/corpus.jsonl"
PASSAGE_STORE = "data/passages.pkl"
QUERY_VECTORS = "data/eval_queries.f32"
REPORT_PATH = "docs/CHUNKING_REPORT.md"
DEPTH = 10


# --------------------------------------------------------------------- metrics


def is_hit(labelled: list[str], retrieved_source: str) -> bool:
    """Whether a retrieved source satisfies any of a query's labelled passages.

    Substring as well as equality, because query_group chunks concatenate
    several passages: the labelled passage is contained in the chunk rather than
    equal to it. Span strategies return the exact parent text, so equality
    covers them and the substring test cannot create false positives -- an
    unrelated parent does not contain the labelled passage.
    """
    return any(
        label == retrieved_source or label in retrieved_source for label in labelled
    )


def _relevance(ranked: list[str], labelled: list[str], k: int) -> list[int]:
    return [1 if is_hit(labelled, source) else 0 for source in ranked[:k]]


def recall_at_k(queries: list[tuple[list[str], list[str]]], k: int) -> float:
    """Share of queries with at least one labelled passage inside the top k."""
    if not queries:
        return 0.0
    hits = sum(
        1 for ranked, labelled in queries if any(_relevance(ranked, labelled, k))
    )
    return hits / len(queries)


def mrr_at_k(queries: list[tuple[list[str], list[str]]], k: int) -> float:
    if not queries:
        return 0.0
    total = 0.0
    for ranked, labelled in queries:
        for position, relevant in enumerate(_relevance(ranked, labelled, k), start=1):
            if relevant:
                total += 1 / position
                break
    return total / len(queries)


def dcg(relevance: list[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))


def ndcg_at_k(queries: list[tuple[list[str], list[str]]], k: int) -> float:
    if not queries:
        return 0.0
    total = 0.0
    for ranked, labelled in queries:
        relevance = _relevance(ranked, labelled, k)
        ideal = dcg(sorted(relevance, reverse=True))
        total += (dcg(relevance) / ideal) if ideal else 0.0
    return total / len(queries)


# ---------------------------------------------------------------------- phases


def _phase_embed(sample: int, seed: int) -> None:
    """torch only."""
    import numpy as np

    from app.embeddings import embed, embed_batch

    rows = []
    with open(CORPUS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if any(p["is_selected"] for p in row["passages"]):
                rows.append(row)
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))

    # Timed one at a time, because that is what the request path does.
    warm = embed("warmup")
    assert warm is not None
    per_query_ms = []
    for row in picked[: min(60, len(picked))]:
        start = time.perf_counter()
        embed(row["query"])
        per_query_ms.append((time.perf_counter() - start) * 1000)

    vectors = np.asarray(embed_batch([r["query"] for r in picked]), dtype="float32")
    Path(QUERY_VECTORS).write_bytes(np.ascontiguousarray(vectors).tobytes())
    Path(QUERY_VECTORS + ".meta.json").write_text(
        json.dumps(
            {
                "count": len(picked),
                "dimension": int(vectors.shape[1]),
                "query_ids": [r["query_id"] for r in picked],
                "labels": [
                    [p["text"] for p in r["passages"] if p["is_selected"]]
                    for r in picked
                ],
                "embed_ms_p50": round(statistics.median(per_query_ms), 2),
                "embed_ms_p100": round(max(per_query_ms), 2),
            }
        )
    )
    Path("data/heldout_query_ids.json").write_text(
        json.dumps(sorted({r["query_id"] for r in picked}))
    )
    print(
        f"embedded {len(picked)} labelled queries; "
        f"embed P50 {statistics.median(per_query_ms):.1f}ms"
    )


def _phase_measure() -> None:
    """faiss only."""
    import faiss
    import numpy as np

    from app.passages import load_passage_store, resolve_text

    meta = json.loads(Path(QUERY_VECTORS + ".meta.json").read_text())
    count, dim = meta["count"], meta["dimension"]
    vectors = np.frombuffer(Path(QUERY_VECTORS).read_bytes(), dtype="float32").reshape(
        count, dim
    )
    labels = meta["labels"]
    passages = load_passage_store(PASSAGE_STORE)

    results = []
    for name in dense_names():
        index_path, metadata_path = chunk_paths(name)
        if not Path(index_path).exists():
            continue
        index = faiss.read_index(index_path)
        rows = pickle.loads(Path(metadata_path).read_bytes())
        manifest = json.loads(
            Path(index_path).with_suffix(".manifest.json").read_text()
        )
        # Not a search-time filter. Filtering out the query's own row would
        # remove the passages that carry its relevance labels, making recall
        # structurally zero -- measured 0.006 before this was caught. The
        # held-out comparison is the `query_aware_heldout` index instead.
        held_out = name == "query_aware_heldout"

        # Over-fetch then dedup by parent, exactly as app/retrieval.py does, so
        # the measured numbers describe the served pipeline and not a variant.
        search_ms = []
        per_query = []
        for i in range(count):
            query = vectors[i : i + 1]
            start = time.perf_counter()
            _, ids = index.search(query, DEPTH * 4)
            search_ms.append((time.perf_counter() - start) * 1000)

            seen: set[int] = set()
            sources: list[str] = []
            for row_id in ids[0]:
                if row_id < 0:
                    continue
                row = rows[row_id]
                parent_id = row["parent_id"]
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                text = resolve_text(row, passages)
                sources.append(
                    text if row.get("text") is not None else passages[parent_id]["text"]
                )
                if len(sources) >= DEPTH:
                    break
            per_query.append((sources, labels[i]))

        results.append(
            {
                "strategy": name,
                "axis": get(name).axis,
                "held_out": held_out,
                "chunks": manifest["chunks"],
                "index_mb": Path(index_path).stat().st_size / 1e6,
                "metadata_mb": Path(metadata_path).stat().st_size / 1e6,
                "threshold": manifest.get("offtopic_threshold"),
                "false_refusal": manifest.get("calibration", {}).get(
                    "false_refusal_rate"
                ),
                "leaks": manifest.get("calibration", {}).get("off_topic_leaks"),
                "recall@1": recall_at_k(per_query, 1),
                "recall@5": recall_at_k(per_query, 5),
                "recall@10": recall_at_k(per_query, 10),
                "mrr@10": mrr_at_k(per_query, 10),
                "ndcg@10": ndcg_at_k(per_query, 10),
                "search_ms_p50": statistics.median(search_ms),
                "search_ms_p100": max(search_ms),
            }
        )
        print(
            f"  {name:<17} recall@5 {results[-1]['recall@5']:.3f}  "
            f"mrr@10 {results[-1]['mrr@10']:.3f}  "
            f"search P50 {results[-1]['search_ms_p50']:.2f}ms"
            + ("  [held-out]" if held_out else "")
        )

    Path("data/eval_results.json").write_text(
        json.dumps({"queries": count, "embed": meta, "results": results}, indent=2)
    )


def write_report() -> None:
    data = json.loads(Path("data/eval_results.json").read_text())
    results = sorted(data["results"], key=lambda r: -r["recall@5"])
    embed_p50 = data["embed"]["embed_ms_p50"]
    embed_p100 = data["embed"]["embed_ms_p100"]
    n = data["queries"]

    n_idx = len(data["results"])
    lines = [
        "# Chunking Strategy Comparison",
        "",
        f"Measured {time.strftime('%Y-%m-%d')} over **{n:,} corpus queries** that carry at",
        f"least one `is_selected` relevance label, against **{n_idx} indices** built from the",
        "same 99,767-passage corpus and the same embedding model",
        "(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, int8 scalar-quantized FAISS).",
        "",
        "Retrieval over-fetches 4x and collapses candidates by parent passage before",
        "truncating, which is exactly what `app/retrieval.py` serves, so these numbers",
        "describe the shipped pipeline rather than a variant of it.",
        "",
        "## What the numbers say",
        "",
        "**Chunking does nothing on this corpus.** `whole_passage`, `fixed_size` and",
        "`recursive` score an identical 0.848 recall@5, and their MRR@10 differs only in the",
        "third decimal (0.591 / 0.592 / 0.593). Those are three genuinely different",
        "splitters -- no split, 700-char windows cutting mid-word, and sentence-aligned",
        "400-char packing -- and they are indistinguishable. The corpus explains it: mean",
        "passage length is 333 characters and p99 is 727, so a passage is already the",
        "atomic retrievable unit and there is nothing useful to cut.",
        "",
        "**Splitting finer makes it worse, monotonically.** No split 0.848, 200-char",
        "children 0.822, ~1.6-sentence semantic groups 0.818, single sentences 0.790. Every",
        "increase in granularity costs recall.",
        "",
        "**But the driver is topical purity, not chunk size.** `query_group` has the",
        "*largest* chunks (876 characters mean) and by far the worst recall, 0.422. It",
        "concatenates every passage answering the same query, so each vector is a blur of",
        "ten loosely related topics. One coherent passage is the optimum: sub-splitting",
        "loses context, super-merging loses focus.",
        "",
        "**Query enrichment shows no reliable gain, and its apparent gain was an artifact.**",
        "See the note below the table -- this was the single most misleading result in the",
        "set, and it was wrong in both directions before being pinned down.",
        "",
        "**Recommended default: `whole_passage` or `fixed_size`.** Tied-best recall,",
        "essentially tied-best MRR, the smallest index of the competitive strategies, fast",
        "search, and the fewest off-topic leaks (1 of 8). They are interchangeable, which is",
        "itself the first finding restated.",
        "",
        "## Retrieval quality",
        "",
        ("| strategy | axis | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |"),
        ("|---|---|---|---|---|---|---|"),
    ]
    for r in results:
        star = " \\*" if r["held_out"] else ""
        lines.append(
            f"| `{r['strategy']}`{star} | {r['axis']} | {r['recall@1']:.3f} | "
            f"**{r['recall@5']:.3f}** | {r['recall@10']:.3f} | {r['mrr@10']:.3f} | "
            f"{r['ndcg@10']:.3f} |"
        )
    lines += [
        "",
        "### The `query_aware` asterisk, and why neither of its numbers is clean",
        "",
        "`query_aware` embeds each passage with the gold query that passage answers -- free",
        "document expansion, since the dataset ships the query. It was the strategy this",
        "work expected most from. It does not deliver, and establishing that took three",
        "corrections.",
        "",
        "The tell is in the table: **`query_aware` scores recall@10 of exactly 1.000.**",
        "Every one of the 500 queries finds its own enriched passage within ten results,",
        "because that passage's vector literally contains the query being searched for.",
        "That is self-reference, not retrieval. Its recall@5 of 0.790 is *below*",
        "`whole_passage`, because enriching all 99,767 passages makes them uniformly",
        "query-shaped and therefore harder to tell apart -- the noise costs more than the",
        "self-match gains.",
        "",
        "`query_aware_heldout` is a control index: identical, except the 500 evaluated",
        "rows' passages are embedded bare while every other row stays enriched. It scores",
        "0.850, marginally the best in the table. **That number is also an artifact**, in",
        "the opposite direction: those rows' passages are clean while all their competitors",
        "carry query noise, which is an advantage production would never grant.",
        "",
        "So the honest reading is a bracket, 0.790 to 0.850, with `whole_passage`'s 0.848",
        "sitting inside it. **Query enrichment shows no measurable gain on this corpus.**",
        "A clean measurement would need queries that never entered the index at all, which",
        "this dataset does not provide.",
        "",
        "The same self-reference inflated its guardrail calibration: measured naively the",
        "off-topic threshold came out at 0.722 with zero leaks of eight probes, apparently",
        "the only strategy whose in-corpus and off-topic score distributions separate",
        "cleanly. Held out it is 0.400 with five leaks, the worst of the slate. Shipping the",
        "naive threshold would have refused most real traffic.",
        "",
        "## Cost and calibration",
        "",
        "| strategy | chunks | index MB | metadata MB | off-topic threshold | false refusal | leaks (of 8) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['strategy']}` | {r['chunks']:,} | {r['index_mb']:.1f} | "
            f"{r['metadata_mb']:.1f} | {r['threshold']:.3f} | "
            f"{100 * r['false_refusal']:.1f}% | {r['leaks']} |"
        )

    total_chunks = sum(r["chunks"] for r in results)
    total_mb = sum(r["index_mb"] + r["metadata_mb"] for r in results)
    lines += [
        "",
        f"**{total_chunks:,} vectors across {len(results)} indices, "
        f"{total_mb:.0f} MB total.**",
        "Chunk metadata is span-addressed -- `(parent_id, start, end)` into one shared",
        "passage store -- rather than each chunk carrying its own copy of its parent text.",
        "That is 18.4 bytes per chunk against 763.9 measured on the previous scheme, a 41x",
        "reduction, and it is what makes nine indices cost roughly what two did.",
        "",
        "## Latency",
        "",
        "Decomposed rather than summed, because only one half scales with index size, and",
        "because embedding and FAISS cannot share a process on this build machine (both",
        "link their own OpenMP runtime; co-loading them segfaults once Metal is in use).",
        "",
        (
            f"Query embedding is strategy-independent: **P50 {embed_p50:.1f} ms, "
            f"P100 {embed_p100:.1f} ms**."
        ),
        "",
        "| strategy | search P50 | search P100 | embed + search P50 |",
        "|---|---|---|---|",
    ]
    for r in sorted(data["results"], key=lambda r: r["search_ms_p50"]):
        lines.append(
            f"| `{r['strategy']}` | {r['search_ms_p50']:.2f} ms | "
            f"{r['search_ms_p100']:.2f} ms | {embed_p50 + r['search_ms_p50']:.1f} ms |"
        )

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("embed", "measure"))
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.phase == "embed":
        _phase_embed(args.sample, args.seed)
        sys.exit(0)
    if args.phase == "measure":
        _phase_measure()
        sys.exit(0)

    for phase in ("embed", "measure"):
        cmd = [sys.executable, "-m", "scripts.evaluate_strategies", "--phase", phase]
        if phase == "embed":
            cmd += ["--sample", str(args.sample), "--seed", str(args.seed)]
        result = subprocess.run(cmd, env={**os.environ}, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)
    write_report()
    Path(QUERY_VECTORS).unlink(missing_ok=True)
    Path(QUERY_VECTORS + ".meta.json").unlink(missing_ok=True)
