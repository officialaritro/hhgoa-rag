# Voice-Enabled RAG Pipeline

A user speaks a question, the pipeline transcribes it, retrieves grounded context from an
indexed corpus, and answers — end to end, with guardrails that know when *not* to answer.

**Live:** https://ragingoa.duckdns.org

```
Voice → Speech-to-text (ElevenLabs) → Unsafe-input guard → Retrieval (FAISS)
      → Cross-encoder rerank → Off-topic guard → Answer generation (Claude Haiku)
      → Claim-level groundedness guard → Answer
```

Every external call runs through a typed harness (`app/harness.py`): one retry on failure,
then a structured `StageResult` instead of a raised exception — no bare try/except scattered
through the pipeline.

## Chunking — eight strategies across four axes, measured

The task calls for "vast" chunking, and for real thought about how the corpus is split,
indexed **and retrieved**. Eight strategies are built, calibrated, served and measured. The
axis matters more than the count: a slate that only varies chunk size is a parameter sweep.

| Axis | Strategy | What it varies |
|---|---|---|
| split | `whole_passage` | no split — one vector per passage |
| split | `fixed_size` | 700-char windows, 20% overlap, cutting mid-word |
| split | `recursive` | sentence-aligned ~400 chars, one sentence of overlap |
| split | `semantic` | embedding-breakpoint merging of adjacent sentences |
| unit | `parent_child` | embeds a 200-char child, **returns the whole parent** |
| unit | `sentence_window` | embeds one sentence, **returns it ±1** |
| enrichment | `query_aware` | embeds the passage with its gold query prepended |
| aggregation | `query_group` | concatenates every passage sharing a query, then splits |

Plus two retrieval-time fusion modes: `hybrid` (BM25 + dense, reciprocal rank fusion) and
`fusion` (RRF across three granularities).

**The measured result is that chunking does not help on this corpus.** Over 500 labelled
queries with paired 95% confidence intervals — full matrix in
[`docs/CHUNKING_REPORT.md`](docs/CHUNKING_REPORT.md):

| strategy | recall@5 | vs `whole_passage` |
|---|---|---|
| `whole_passage` | 0.848 | *baseline* |
| `fixed_size` | 0.848 | no difference |
| `recursive` | 0.848 | no difference |
| `fusion` | 0.854 | no difference |
| `parent_child` | 0.822 | no difference |
| `semantic` | 0.818 | **worse** |
| `sentence_window` | 0.790 | **worse** |
| `query_aware` | 0.790 | **worse** |
| `hybrid` | 0.740 | **worse** |
| `query_group` | 0.422 | **worse** |

Four strategies are significantly worse than not chunking at all, and **none is
significantly better**. Passages here average 333 characters, so a passage is already the
atomic retrievable unit — `fixed_size`'s 700-char window exceeds only 1.4% of them. It took
the full slate to establish that, and `scripts/audit_strategies.py` separately verifies that
each strategy does what its name claims rather than degenerating into another.

## Reranking — what the chunking matrix pointed at

The matrix answered a different question than it was asked. `whole_passage` reaches
recall@10 of **0.960** but recall@1 of **0.410**: the right passage is nearly always
retrieved and simply not ranked first. That is an *ordering* problem, and no chunking
strategy touches ordering — a bi-encoder embeds query and passage separately and never
compares them.

A cross-encoder does:

| | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| dense order | 0.410 | 0.848 | 0.591 |
| **+ reranking (depth 10)** | **0.504** | **0.916** | **0.671** |

recall@5 +6.8 points, paired 95% CI **+0.038 to +0.098** — larger than everything the
chunking slate achieved, and the only change here whose interval excludes zero.

## Guardrails — three checks, calibrated from measured distributions

No second LLM call: every check is a threshold on a similarity score or a pattern match,
set from measured data rather than a guessed constant.

- **Unsafe input** — regex on the transcript, before retrieval runs (~0ms).
- **Off-topic** — top retrieval similarity against a threshold **measured per index at
  build time and written into that index's manifest**. This mattered: a value verified
  against the semantic index once shipped as the `fixed_size` default and refused **38.5%**
  of real in-corpus questions. `app/guardrails.py` now raises `MissingCalibration` rather
  than borrow another index's number, so an uncalibrated strategy cannot be served at all.
  Calibrated against 50 off-topic probes across 8 categories (`app/offtopic_probes.py`).
- **Groundedness** — per-answer-**sentence** support against each retrieved passage
  individually. The previous version concatenated all five passages into one string, which
  exceeds MiniLM's 256-token limit for **94%** of real contexts — so it was silently scoring
  answers against a truncated context. A literal number check runs alongside it, because a
  fabricated figure is invisible to cosine: "founded in 1987" against a context saying 1897
  is one digit apart and semantically identical.

Calibrated against 40 real generated answers: grounded minimum **0.524**, ungrounded maximum
**0.271** — a clean gap where the distributions previously overlapped, catching 100% of
ungrounded answers at zero false refusals.

## Latency — three boundaries, not one number

The 200ms target covers "chunking + vector DB retrieval + everything through to final
output." Two stages are third-party calls whose latency isn't ours, so one number would hide
where time goes. Measured against the **live deployment**, 30 real spoken-audio requests
(method and per-stage breakdown in [`docs/LATENCY_REPORT.md`](docs/LATENCY_REPORT.md)):

| Boundary | Covers | P50 | P70 | P100 | Under 200ms |
|---|---|---|---|---|---|
| **A** | Retrieval, reranking, all guardrails | **95.9ms** | **99.7ms** | **121.6ms** | **Yes** |
| B | A + answer generation (Claude) | 1,425ms | 1,585ms | 3,196ms | No |
| C | Full voice-to-answer, incl. speech-to-text | 2,655ms | 2,830ms | 4,369ms | No |

**Boundary A clears the target at every percentile with 39% headroom — while carrying a
cross-encoder it did not have before.** It is 26% faster than the previous measurement,
because the groundedness guard stopped re-embedding vectors retrieval had already computed
(110.7ms → 12.1ms) and answers got shorter (generation 2,108ms → 1,329ms).

These are `/api/ask` figures. `/api/compare` runs the pipeline four times over one
transcription and does **not** meet the target — 120–290ms per strategy, since reranking is
CPU-bound and the endpoint contends with itself. `docs/LATENCY_REPORT.md` explains what was
done about that and what deliberately was not.

B and C don't clear it and can't: Claude's generation floor is ~650ms to first token and
ElevenLabs' STT floor ~1.1s, both measured independently and provider-bound.

## Frontend

- **Compare strategies** — one spoken question answered against four granularities from a
  single transcription, side by side. The default set is a granularity ladder, so the
  report's central finding is visible in the demo itself.
- **Citations** — every answer shows the exact retrieved passages that grounded it.
- **Latency chart** — the per-stage breakdown, rendered live per query.
- Hold the mic button *or* hold **Space** to talk; **Esc** drops an in-progress recording.

## Running locally

```bash
uv sync
cp .env.example .env   # fill in ELEVENLABS_API_KEY and ANTHROPIC_API_KEY
set -a && source .env && set +a
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` — `localhost` counts as a secure context, so microphone capture
works over plain HTTP locally (production requires TLS, see `deploy/AWS_SETUP.md`).

Building the indices from scratch (~70 minutes; see `docs/CHUNKING_REPORT.md` for sizes):

```bash
uv run python -m scripts.ingest_dataset        # corpus -> data/corpus.jsonl
uv run python -m scripts.build_all             # every registered strategy
uv run python -m scripts.build_lexical         # BM25, for the hybrid mode
uv run python -m scripts.calibrate_thresholds  # per-index off-topic thresholds
```

`build_all` is resumable, retries with backoff, and re-execs under `caffeinate` so an idle
machine cannot suspend it mid-build.

```bash
uv run pytest -q                  # 292 tests
uv run ruff check . && uv run ruff format --check .
```

Tests that need built indices skip when `data/` is absent, which is the case in CI.

## Repo layout

```
app/            FastAPI backend — stt, retrieval, reranking, generation, guardrails, harness
  chunkers.py     the eight chunking strategies, as pure functions
  strategies.py   the registry: one source of truth for strategy identity
  passages.py     shared parent-passage store; chunks are spans into it
  vectors.py      embedding half of the build (never imports faiss)
  indexing.py     FAISS half of the build (never imports torch)
scripts/        ingest, build, calibrate, evaluate, benchmark
static/         frontend (vanilla HTML/CSS/JS — no build step)
tests/          pytest suite, mocked at the external-call boundary
deploy/         AWS EC2 + Caddy setup, systemd unit, CI deploy
docs/           chunking report, latency report, environment reference
```

The build runs as **two processes** — one embedding, one indexing — because faiss and torch
each bundle their own OpenMP runtime and a process holding both segfaults once Metal is in
use. `app/vectors.py` never imports faiss, `app/indexing.py` never imports torch, and a
subprocess test guards each direction.

Deploys automatically to the live link on every push to `main`
(`.github/workflows/ci.yml`), after tests, lint and format checks pass on Python 3.12 and
3.14, with a health gate and automatic rollback.
