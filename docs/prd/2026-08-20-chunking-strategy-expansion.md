# Chunking Strategy Expansion

Created: 2026-08-20
Agent: Claude Code
Category: Feature
Status: Approved
Research: Measured (live instance + full corpus)

## Problem Statement

`task 2_ hhg.md` requires that "chunking strategy should be **vast** — don't submit a
single naive fixed-size chunking approach," and asks for "real thought put into how the
dataset is split, indexed, and retrieved."

The shipped system has two strategies, `fixed_size` and `semantic`. Measured against the
full ingested corpus on 2026-08-20, neither does what its name claims, and together they
occupy only the two extreme ends of the design space:

**Finding 1 — `fixed_size` barely chunks.** The corpus holds **99,767 passages** whose
length distribution is mean 333 chars, p50 299, p75 377, p90 536, **p99 727**, max 1233.
The strategy's window is 700 chars, so only **1.4%** of passages exceed it. It produces
101,131 chunks from 99,767 passages: **98.6% of its output is the unmodified passage**.
It is within rounding distance of not chunking at all.

**Finding 2 — `semantic` is a sentence splitter.** The corpus holds **349,983 sentences**
under the regex boundary in `scripts/chunk_semantic.py`. The strategy produces **338,544
chunks — 96.7% of the sentence count**. Its similarity threshold of 0.8 is high enough
that adjacent sentences almost never merge, so it pays one embedding call per sentence at
index time to approximate `re.split`.

The two strategies are therefore "do not split" and "split at every sentence." The entire
middle of the space is empty, no strategy varies the *unit returned* to generation, no
strategy uses the dataset's own per-row query text, and no candidate fusion happens at
retrieval time.

This document specifies an expansion to eight chunking strategies plus two retrieval-time
fusion modes, the refactor that makes that affordable, and the comparison report that
makes the claim verifiable rather than asserted.

## Measured Baseline

All figures measured on the live instance `i-09e157bfae9bb82a6` (`m7i-flex.large`,
2 vCPU / 7.6 GB, `ap-south-1b`) and the full `data/corpus.jsonl`, on 2026-08-20.

### Corpus

| Property | Value |
|---|---|
| Rows ingested | 10,000 |
| Passages | 99,767 (10.0 per row) |
| Total passage text | 33.2M chars |
| Passage chars | mean 333, p50 299, p75 377, p90 536, p99 727, max 1233 |
| Passages > 400 chars | 22.7% |
| Passages > 700 chars | 1.4% |
| Sentences | 349,983 (3.51 per passage) |
| Query text | mean 34 chars, one per row, currently unused outside evaluation |

### Current on-disk and resident cost

| Artifact | Size |
|---|---|
| `corpus.jsonl` | 38.6 MB |
| `chunks_fixed_size.jsonl` (intermediate) | 78.6 MB |
| `chunks_semantic.jsonl` (intermediate) | 192.2 MB |
| `index_fixed_size.faiss` (101,131 vectors) | 38.8 MB |
| `index_semantic.faiss` (338,544 vectors) | 130.0 MB |
| `metadata_fixed_size.pkl` | 77.3 MB |
| `metadata_semantic.pkl` | 187.5 MB |
| **`data/` total** | **709 MB** |
| uvicorn RSS, both indices + MiniLM resident | 1.29 GB |
| Free disk after 2026-08-20 cleanup | 11 GB of 19 GB |
| Available memory | 6.0 GB of 7.6 GB |

FAISS costs **384 bytes per vector** (int8 scalar-quantized, 384-dim). Metadata costs
**555-765 bytes per chunk**, because every chunk row stores its own `text` *and* a full
copy of its `source_passage` (`scripts/chunk_fixed_size.py`). Metadata is roughly twice
the size of the index it describes.

Embedding throughput is approximately **330 texts/sec** on this instance, derived from the
implementation plan's measured ~40 min for the current 790k combined chunk-and-sentence
embeds, consistent with the ~354/sec recorded in `app/embeddings.py`.

## Goals

1. Eight chunking strategies that differ in **method**, spanning four distinct axes, not
   one axis with varied parameters.
2. Two retrieval-time fusion modes, exposed through the existing strategy parameter.
3. A single registry as the sole source of strategy identity.
4. Storage normalized so ten strategies cost no more disk than today's two.
5. Off-topic thresholds calibrated automatically at build time, making an uncalibrated
   strategy structurally impossible to ship.
6. A published comparison report with retrieval quality, size and latency per strategy.

## Non-Goals

- **Cross-encoder reranking.** Considered and deliberately deferred. It would require
  reworking the groundedness guard to free latency headroom, which reopens the already
  published `docs/LATENCY_REPORT.md`. Recorded in Deferred Work below.
- **Changing the 200 ms critical path.** Retrieval stays at one embed plus one or more
  FAISS searches. No new model call is added to the request path.
- **Re-ingesting or enlarging the corpus.** The 10,000-row cap set by the implementation
  plan's Global Constraints stands.
- **A larger or different embedding model.** `all-MiniLM-L6-v2` stays; it was already
  benchmarked at 11.8 ms/query against `bge-small` at 38.9 ms.
- **Text-to-speech, multi-turn, auth.** Out of scope per the PRD, unchanged.
- **Replacing FAISS or changing the index type.** `IndexScalarQuantizer` int8 stays; its
  99.6% top-5 overlap versus exact search is already measured.

## The Eight Strategies

Four axes. The axis column is the point: a slate that varies only chunk size is a
parameter sweep, not a strategy slate.

| # | Strategy | Axis | Method | Est. vectors |
|---|---|---|---|---|
| 1 | `whole_passage` | split | No split. One vector per passage. | 99,767 |
| 2 | `fixed_size` | split | Unchanged. 700-char window, 20% overlap. | 101,131 |
| 3 | `recursive` | split | Separator hierarchy ¶ → line → sentence → word, target ~400 chars, overlap of one whole sentence. | ~140,000 |
| 4 | `semantic` | split | Embedding-breakpoint split, **threshold retuned** (see below). | ~150,000-200,000 |
| 5 | `parent_child` | unit | Embed a ~200-char child; **return the parent passage**. | ~166,000 |
| 6 | `sentence_window` | unit | Embed one sentence; return that sentence ±1 neighbour. | 349,983 |
| 7 | `query_aware` | enrichment | Embed `Q: {row.query}\n{passage}`; return the bare passage. | 99,767 |
| 8 | `query_group` | aggregation | Concatenate all passages sharing a `query_id`, then split at ~1000 chars. | ~40,000 |

Counts for 1, 2, 6 are exact (measured). Counts for 3, 4, 5, 8 are estimates derived from
the 33.2M-char total and the measured length distribution.

### Notes on individual strategies

**`whole_passage` is the control, and it is expected to tie `fixed_size`.** That is its
purpose. Finding 1 currently rests on a chunk-count ratio; this strategy converts it into
a retrieval-quality measurement judges can read off the report.

**`semantic`'s threshold changes from 0.8 to ~0.55**, chosen by measuring the adjacent-
sentence similarity distribution over a corpus sample and placing the threshold at its
median rather than by intuition. At 0.8 it merges almost nothing (Finding 2), making it a
duplicate of `sentence_window`'s granularity at higher build cost. The existing docstring
in `scripts/chunk_semantic.py` argues 0.8 on the grounds that unrelated sentences sit at
0.3-0.6 — that reasoning describes *unrelated* sentences, but adjacent sentences within a
single web passage are usually related, which is why the observed merge rate is near zero.
The retune is a correction to that constant, and its chunk count and guardrail threshold
both move as a result.

**`query_aware` is the highest-value strategy in the slate.** Every corpus row carries the
natural-language query its passages answer (`scripts/ingest_dataset.py`), currently read
only to produce evaluation labels. Prepending it at embed time is document expansion —
the doc2query / HyDE pattern — with the synthetic-query generation step replaced by
ground truth the dataset already ships. It costs no LLM calls and directly targets the
query/passage vocabulary gap that MS MARCO exists to measure.

**A correctness note that applies to `query_aware`'s evaluation.** Because the gold query
is baked into the vector, evaluating this strategy with the same query as the retrieval
input measures a partially self-referential match. The evaluation must therefore report
`query_aware` results with this caveat stated inline in the report, and must additionally
report a **held-out variant**: for the evaluated rows only, the index is built without
that row's own query. Without this, the strategy will post an inflated recall number that
does not survive contact with an unseen question. This is a required part of the
evaluation, not an optional refinement.

## The Two Fusion Modes

Both are exposed as ordinary `strategy=` values, so `/api/ask`, `/api/compare`, the
`/api/strategies` listing, and the frontend radio group need no new plumbing.

| Mode | Method |
|---|---|
| `hybrid` | BM25 lexical shortlist + dense shortlist over `recursive`, merged by reciprocal rank fusion (RRF, k=60). |
| `fusion` | RRF across the ranked lists of `whole_passage`, `recursive`, `semantic`, and `query_aware`. |

`hybrid` exists because MS MARCO is a keyword-heavy web-search corpus where BM25 is a
strong baseline that fails on different queries than a dense bi-encoder does. `fusion` is
the payoff for `/api/compare` already existing: the endpoint currently displays two
strategies side by side, and fusing their rankings is the natural next step.

**BM25 implementation:** `bm25s` (sparse, numpy-backed). If measured latency over ~140,000
documents does not fit the budget, fall back to a scikit-learn TF-IDF sparse matmul and
**state the substitution in the report** rather than dropping `hybrid` silently.

**Fusion cost:** `fusion` runs four FAISS searches against one query embedding. The query
is embedded once and reused. Searches run concurrently. Budget impact is assessed in
Latency Impact below.

## Architecture

### A. Strategy registry — `app/strategies.py` (new)

Strategy identity is currently duplicated across four locations: `_CHUNKS_PATHS`
(`scripts/build_index.py`), `INDEX_PATHS` (`app/retrieval.py`), `_STRATEGIES`
(`app/main.py`), and `_DEFAULT_OFFTOPIC_THRESHOLDS` (`app/guardrails.py`). At two
strategies that is eight entries to keep in sync; at ten it is forty, and a missed entry
is exactly the class of defect that put a miscalibrated threshold into production.

One frozen record per strategy:

```python
@dataclass(frozen=True)
class Strategy:
    name: str
    kind: Literal["dense", "hybrid", "fusion"]
    chunker: Callable[[Row], Iterator[Chunk]] | None  # None for composed kinds
    members: tuple[str, ...] = ()                     # fusion members
    description: str = ""                             # surfaced by /api/strategies
```

Index paths, the build loop, `/api/strategies`, threshold lookup and the evaluation loop
all derive from this registry. A coupling test asserts that every registered strategy has
either a `chunker` or non-empty `members`, has a calibrated threshold in its manifest, and
appears in `/api/strategies`.

### B. Parent store and span-addressed chunks

Today each chunk row carries both its own text and a full copy of its parent passage. At
the projected ~1.15M vectors that duplication would cost roughly 700 MB of metadata alone,
and it is pure redundancy.

New on-disk shape:

- **`data/passages.pkl`** — built once, shared by every strategy:
  `passage_id → {text, is_selected, query_id, query}`. ~99,767 entries, ~40 MB.
- **`data/index_<name>.faiss`** — unchanged format.
- **`data/chunks_<name>.pkl`** — per-strategy chunk rows. Each row is either a **span**,
  `(parent_id, start, end)`, or carries `text` explicitly when the chunk is not a
  contiguous substring of one parent.

Seven of the eight strategies are span-addressable. Only `query_group` needs stored text,
because its chunks deliberately cross passage boundaries. `semantic` is span-addressable
by storing a sentence-index range and re-splitting the parent on read; the re-split is
cheap and keeps its metadata in the same shape as the others.

**Intermediate `chunks_*.jsonl` files stop being produced.** Chunkers become generators
consumed directly by the embedder. This removes 271 MB of current intermediates and
avoids ~1 GB more at the new scale. The two existing intermediate files are retained
until the new pipeline is verified, then deleted.

### C. Retrieval dispatch — `app/retrieval.py`

`retrieve()` gains a dispatch on `Strategy.kind`:

- **dense** — embed query, FAISS search, resolve parents, dedup, truncate to `k`.
- **hybrid** — BM25 shortlist and dense shortlist, RRF merge, resolve, dedup, truncate.
- **fusion** — one query embedding, concurrent FAISS searches across members, RRF merge,
  resolve, dedup, truncate.

All three paths **over-fetch `k * 4`, collapse candidates sharing a `parent_id`, then
truncate to `k`.** Today a 20%-overlap strategy can spend several of its five slots on
near-identical text from one passage, silently shrinking the distinct context handed to
generation. Dedup is not an enhancement; it is a correctness fix that the parent store
makes trivial to implement.

### D. Schema and generation

`RetrievedPassage` keeps both `text` (the matched chunk) and `source_passage` (the parent),
so the frontend continues to work unchanged. **Generation switches from consuming `text`
to consuming `source_passage`.** Without this change `parent_child` and `sentence_window`
have no observable effect, since their entire premise is that the returned unit is wider
than the embedded one.

The groundedness guard continues to score the answer against the text actually shown to
the model, which after this change is the parent text. Its threshold is re-measured
accordingly; the existing 0.40 default was calibrated against chunk text.

### E. Build-time threshold calibration

`scripts/tune_thresholds.py` already computes the right statistic — in-corpus p05 against
off-topic p95, with an explicit warning when the distributions overlap — and nothing
consumes its output. Wiring it in:

1. `build_index` calls the tuner after building each index and writes
   `offtopic_threshold` into that index's `.manifest.json`.
2. `check_off_topic` reads the threshold from the manifest. An explicit environment
   variable still wins, preserving the documented retune-without-deploy path.
3. `load_index` raises when `offtopic_threshold` is absent, matching the posture of the
   existing `IndexModelMismatch` guard.

A strategy therefore cannot reach the serving path uncalibrated. This is the structural
fix for the defect recorded in `tests/test_guardrail_calibration.py`, where a threshold
verified against `semantic` shipped as the `fixed_size` default and refused 38.5% of real
in-corpus questions.

Calibration cost is ~200 retrievals per strategy, roughly 40 seconds total at build time.
`tests/test_guardrail_calibration.py` keeps its recorded distributions for the two
existing strategies and gains the registry coupling test; the measured tables are updated
in the same commit as the retune, per its own instructions.

## Projected Cost

| Item | Projection | Basis |
|---|---|---|
| Total vectors, 8 indices | ~1.15M | measured + estimated per-strategy counts |
| FAISS on disk | ~450 MB | 384 B/vector, measured |
| Chunk metadata | ~150 MB | span rows, ~100 B/chunk |
| `passages.pkl` | ~40 MB | 99,767 entries |
| `corpus.jsonl` | 38.6 MB | unchanged |
| **`data/` total** | **~674 MB** | **below the current 709 MB, for 10 strategies instead of 2** |
| uvicorn RSS, all resident | ~2.0 GB | 1.29 GB today + ~450 MB FAISS + unpickled metadata |
| Free disk headroom | 11 GB | measured after cleanup |
| Available memory | 6.0 GB | measured |

Disk and memory are both comfortable. Lazy loading is available for free if needed —
`_load_cached` is already `@cache`-decorated, so narrowing the startup warm loop in
`app/main.py` to the UI-exposed strategies defers the rest until first use — but it is an
optimization, not a requirement.

### Latency impact

The 200 ms target applies to Boundary A of `docs/LATENCY_REPORT.md`: retrieval plus
guardrails, measured at P50 129.3 ms / P100 147.3 ms, of which retrieval is 19.6 ms P50.

- **dense strategies** — unchanged shape. Larger indices add a sublinear amount to search;
  `semantic` at 338,544 vectors already sits inside 19.6 ms.
- **`fusion`** — one embed (~11.8 ms, shared) plus four concurrent searches. The embed
  dominates, so the expected cost is close to a single dense retrieval, not four times it.
- **`hybrid`** — adds one BM25 scoring pass. This is the only genuinely new cost and the
  one component with real budget risk.
- **dedup** — over-fetching `k*4` = 20 instead of 5 is negligible in FAISS.

**Every strategy is re-benchmarked and the latency report updated.** If `hybrid` or
`fusion` exceeds the budget, the report states the measured number rather than the mode
being quietly withdrawn. No strategy that misses 200 ms becomes the UI default.

## Evaluation Deliverable

`scripts/evaluate_strategies.py` (new, extending `scripts/evaluate_retrieval.py`) emits
`docs/CHUNKING_REPORT.md`: one row per strategy and fusion mode, with

- recall@1, recall@5, recall@10 against the corpus `is_selected` labels
- MRR@10 and nDCG@10
- vector count, index MB, metadata MB
- build seconds
- retrieval P50 / P100 ms

plus the `query_aware` held-out variant and the two findings above stated with their
numbers.

**This report is the deliverable, not a supplement to it.** Ten strategies with no
comparison table is indistinguishable from two strategies to a reader. The table is what
converts "vast" from a claim into a measurement.

## Build and Deploy Plan

Building ~1.15M chunk embeddings plus ~349,983 sentence embeddings for semantic boundary
detection is ~1.5M embeds. At the measured ~330/sec that is **~77 minutes on the serving
instance's 2 vCPU** — an hour-plus of CPU starvation on the box running the live demo,
two days before a no-resubmission deadline. So the build happens off the serving box.

### Chosen: build locally on Apple Silicon, ship the artifacts

Measured on the development machine (MacBook Air M1, 8 GB, macOS arm64) on 2026-08-21:

| Device | Throughput | 1.5M embeds |
|---|---|---|
| Local M1, MPS (Metal) | **506 texts/sec** | **~50 min** |
| Local M1, CPU | 326 texts/sec | ~78 min |
| EC2 `m7i-flex.large`, 2 vCPU | ~330 texts/sec | ~77 min |
| EC2 `c7i.4xlarge`, 16 vCPU (not used) | ~2,600 texts/sec est. | ~10 min |

MPS is chosen: it is free, needs no provisioning, and the serving box keeps its CPU. The
~50 min figure is a short-burst measurement; the M1 Air is fanless and will thermal
throttle over a sustained run, so **plan for 65-90 min** and run it overnight.

**Artifact portability is verified, not assumed.** A `IndexScalarQuantizer` index built on
arm64 macOS / Python 3.14 / numpy 2.5.2 / faiss 1.15.0 was transferred to the instance and
read back on x86_64 Linux / Python 3.12.3 / numpy 2.5.2 / faiss 1.15.0:

- FAISS search results are **bit-identical** across the two architectures.
- Pickle protocols 4 and 5 both load on the instance (its `HIGHEST_PROTOCOL` is 5, so the
  local 3.14 default of protocol 5 is safe). `.npy` span arrays load unchanged.
  Even so, **local Python is pinned down from 3.14 to 3.12 to match the instance**, and
  `.python-version` is changed accordingly. The pickle question was never the reason: the
  reason is that 3.12 is what ships, so a green local test run should prove the deploy
  target works, and the index build now happens locally — building artifacts on a
  different interpreter than the one serving them is a variable worth deleting rather
  than verifying. `uv sync --locked --python 3.12` resolves the identical lock (torch
  2.13.0 included) and CI has passed on 3.12 continuously, so the switch carries no
  dependency risk. `.github/workflows/ci.yml` keeps testing both versions: `pyproject.toml`
  declares `requires-python = ">=3.12"`, so the matrix now covers that declared contract
  as a forward-compatibility canary instead of papering over a local/production split.
  Upgrading the *instance* to 3.14 was rejected as the wrong direction — swapping the
  production interpreter under a live demo days before a no-resubmission deadline trades
  real risk for a cosmetic gain. Metadata pickles are additionally written with an
  **explicit** `protocol=4` rather than the interpreter default, so the artifact cannot
  silently change if either side's Python moves again. Bulk span data uses `.npy`, which
  is protocol-independent.
- numpy and faiss versions match exactly on both sides, so no version skew to manage.

**MPS-versus-CPU numerics are verified.** The index is built from MPS-computed vectors
while queries are embedded on the instance's x86 CPU. Measured over 256 texts: cosine
agreement **1.000000** (min and mean), max elementwise delta **1.9e-07**, and **100%
identical top-1 and top-5** for MPS queries against a CPU-built index. The device split
between build time and query time is a non-issue.

### Required prerequisite: `build_index` must stream

`app/indexing.py`'s `build_index` currently embeds *every* chunk into one array before
calling `index.add()`. For the largest strategy that peaks at roughly:

| Component | Size |
|---|---|
| 349,983 × 384 × 4 B float32 | 0.54 GB |
| transient 2× at concatenation | 1.08 GB |
| torch + MiniLM resident (MPS shares system RAM) | ~2.0 GB |
| macOS baseline | ~3.0 GB |
| **peak** | **~6.1 GB on an 8 GB machine** |

That does not crash — macOS swaps — but swapping on a fanless machine compounds with
thermal throttling and makes an overnight run unpredictable. `build_index` therefore
changes to a streaming form:

1. Embed a training sample (the first ~50,000 chunks) and call `index.train()` on it.
   `IndexScalarQuantizer` requires training before adding, so the sample cannot be skipped.
2. Embed the remaining chunks in batches, calling `index.add()` per batch and discarding
   each batch's vectors.
3. Write the index, the span-addressed chunk metadata, and the manifest as today.

Peak memory becomes one batch of vectors rather than the whole corpus. This benefits the
instance too, and is a prerequisite for step 1 of the implementation order rather than an
optimization.

### Sequence

1. `scp` `data/corpus.jsonl` (38.6 MB) down from the instance. Do **not** re-run
   `scripts/ingest_dataset.py` locally: the corpus must be byte-identical to the one the
   measured findings and chunk counts were derived from, and re-ingesting also re-downloads
   the source parquet unnecessarily.
2. Build all indices locally on MPS, overnight.
3. Verify locally: `scripts/evaluate_strategies.py` plus a manifest check on every index.
4. Upload `data/` (~674 MB projected) to S3.
5. Pull onto `i-09e157bfae9bb82a6`, restart the service, re-run the evaluation **on the
   instance** and confirm the numbers match the local run before publishing the report.
   The evaluation is needed for the report regardless, so this verification is free.
6. `deploy/setup.sh` changes its index-bootstrap step from "build if missing" to "fetch
   from S3 if missing, build only as fallback," so a fresh deploy no longer implies an
   hour of embedding.

Renting a `c7i.4xlarge` (~$1, ~10 min, 32 GB RAM, no thermal limit, and an S3 upload that
never touches home bandwidth) remains the faster option and is the fallback if the local
build proves unreliable. It is not chosen because local is free and now measured to work.

## Risks

| Risk | Mitigation |
|---|---|
| `hybrid` BM25 pass misses the 200 ms budget | Measure first; fall back to TF-IDF sparse matmul; report the measured number either way. Never the UI default unless it fits. |
| `semantic` retune changes an already-reported strategy's numbers | Intended. Both the old and retuned configurations are reported, so Finding 2 stays visible rather than being quietly corrected. |
| `query_aware` posts an inflated, self-referential recall | The held-out variant is a required evaluation output, and the caveat is stated inline in the report. |
| Parent-store refactor breaks the live service mid-window | Feature branch; both index generations are on disk simultaneously; the manifest check already refuses a mismatched index rather than serving nonsense. |
| Local overnight build throttles or swaps on the fanless 8 GB M1 | `build_index` streams, capping peak at one batch instead of ~6.1 GB. Plan 65-90 min, not the 50 min burst figure. `c7i.4xlarge` is the fallback. |
| ~674 MB upload from home bandwidth stalls | Upload to S3 with a resumable multipart client; `deploy/setup.sh` fetches from S3 rather than the laptop, so a failed upload never leaves the instance half-updated. |
| Locally built artifacts unreadable on the instance | Verified end to end on 2026-08-21: bit-identical FAISS search across arm64→x86_64, pickle protocols 4 and 5 both load, numpy and faiss versions match. Re-verified by the post-transfer evaluation run. |
| Chunk-count estimates for strategies 3, 4, 5, 8 prove wrong | They affect only build time and disk, both of which have wide headroom (11 GB free, ~674 MB projected). |
| Ten strategies is too many to finish before Aug 22 | The implementation order below is a strict priority sequence; every prefix of it is a complete, honest submission. |

## Implementation Order

Strict priority. Any prefix ships.

1. Registry (`app/strategies.py`), parent store, and the streaming `build_index`.
   Everything depends on these three; the streaming change is what makes the local
   overnight build fit in 8 GB.
2. `recursive`, `parent_child`, `query_aware`, `whole_passage`.
3. Dedup-by-parent, and generation switched to `source_passage`.
4. Build-time threshold calibration and the coupling test.
5. `scripts/evaluate_strategies.py` and `docs/CHUNKING_REPORT.md`.
6. `semantic` retune.
7. `hybrid`.
8. `sentence_window`, `query_group`.
9. `fusion`.
10. Re-benchmark, update `docs/LATENCY_REPORT.md`.

Steps 1-5 produce five strategies plus the report that proves both findings. That is the
minimum defensible submission; each further step widens it.

## Deferred Work

- **Cross-encoder reranking over a shortlist.** The highest-quality option available.
  Requires first reclaiming latency headroom from the groundedness guard, which at 110.7 ms
  P50 is the largest locally-owned cost in the pipeline. That guard re-embeds the
  concatenated retrieved context on every request, and those chunk vectors are already
  resident in FAISS — `index.reconstruct()` plus mean-pooling would replace the long-string
  embed with an approximate lookup, plausibly dropping the guard to ~15 ms and funding a
  reranker inside the existing 200 ms budget. Deferred because it reopens a published
  latency report; recorded because it is the correct next move after this work.
- **Matryoshka two-stage retrieval** (low-dimension shortlist, full-dimension rescore).
- **HNSW or IVF-PQ** in place of the flat scalar-quantized index. Only becomes relevant at
  a corpus size well beyond the current 10,000-row cap.

## Non-Engineering Requirements

Unchanged and still outside this build: the two videos and the `#RAGInGoa` posts on
Instagram, X and LinkedIn by every team member, per `task 2_ hhg.md`.
