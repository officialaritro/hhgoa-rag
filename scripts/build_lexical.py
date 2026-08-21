"""Builds the BM25 index over the passage store.

One document per passage, matching `whole_passage`'s unit. That is deliberate:
RRF merges rankings, so the two halves of a hybrid must rank the *same* items or
the fusion is comparing incomparable things. Chunk-level lexical documents would
also duplicate text across overlapping chunks and let one passage occupy several
lexical ranks.

Neither torch nor faiss is needed, so this runs in one process.

Run:
    .venv/bin/python -m scripts.build_lexical
"""

import time
from pathlib import Path

from app.lexical import BM25Index
from app.passages import load_passage_store

PASSAGE_STORE = "data/passages.pkl"
BM25_PATH = "data/bm25_passages.pkl"

if __name__ == "__main__":
    passages = load_passage_store(PASSAGE_STORE)
    print(f"building BM25 over {len(passages):,} passages ...")
    started = time.perf_counter()
    index = BM25Index.build([p["text"] for p in passages])
    build_s = time.perf_counter() - started
    index.save(BM25_PATH)
    size_mb = Path(BM25_PATH).stat().st_size / 1e6
    print(
        f"  vocabulary {len(index.vocabulary):,} terms, "
        f"nnz {index.weights.nnz:,}, {size_mb:.1f} MB, built in {build_s:.1f}s"
    )
    # A latency sanity check: this runs on the request path.
    probes = [
        "what are the symptoms of dehydration",
        "how much does a passport cost",
        "who invented the telephone",
    ]
    for probe in probes:
        start = time.perf_counter()
        top = index.top_k(probe, k=20)
        ms = (time.perf_counter() - start) * 1000
        print(f'  "{probe[:34]}..." {ms:6.1f} ms  top score {top[0][1]:.2f}')
