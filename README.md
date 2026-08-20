# Voice-Enabled RAG Pipeline

A user speaks a question, the pipeline transcribes it, retrieves grounded context from an
indexed corpus, and answers out loud through the browser — end to end, with guardrails that
know when *not* to answer.

**Live:** https://ragingoa.duckdns.org

```
Voice → Speech-to-text (ElevenLabs) → Unsafe-input guard → Retrieval (FAISS) →
Off-topic guard → Answer generation (Claude Haiku) → Groundedness guard → Answer
```

Every stage runs through a typed harness (`app/harness.py`): one retry on failure, then a
structured `StageResult` instead of a raised exception — no bare try/except scattered
through the pipeline.

## Chunking — two distinct strategies, measured

The task calls for "vast" chunking, not a single naive fixed-size split. This build indexes
the corpus twice, with genuinely different splitting logic, and lets a query be answered
against either index (or both, side by side — see **Compare strategies** below):

| Strategy | Method | Chunks | Recall@5 |
|---|---|---|---|
| `fixed_size` | 700-char windows, 20% overlap (`scripts/chunk_fixed_size.py`) | ~101k | **0.790** |
| `semantic` | Sentence-embedding boundary detection, similarity threshold 0.8 (`scripts/chunk_semantic.py`) | ~338k | **0.661** |

Recall@5 is measured against the dataset's own `is_selected` relevance labels
(`scripts/evaluate_retrieval.py`), not a hand-derived expectation. Fixed-size chunks
outperform here because they're larger and preserve more complete passage context per
chunk; semantic chunking splits into ~3.3x more, smaller pieces, each carrying less context
for this particular ground truth. That's a real empirical result, not a bug — the two
strategies score differently because they *are* different, which is the point.

Both indices are FAISS (`IndexScalarQuantizer`, int8, inner-product) over local
`sentence-transformers/all-MiniLM-L6-v2` embeddings — no hosted embedding API on the
critical path. Retrieval over both indices combined (~440k chunks) runs in **~20ms at P50**.

## Guardrails — three checks, calibrated against measured score distributions

No second LLM call for guardrails — every check is a threshold on a similarity score or a
pattern match, chosen from real measured data rather than a guessed constant
(`app/guardrails.py`):

- **Unsafe input** — regex match on the transcript before retrieval ever runs (free, ~0ms).
- **Off-topic** — the retrieved top similarity score against a threshold calibrated
  *per chunking strategy*. This mattered in practice: semantic chunks are shorter, so every
  cosine score runs higher (in-corpus median 0.779 vs 0.741 for fixed-size). A single shared
  threshold, verified only against the semantic index, shipped against a `fixed_size`
  default and refused **38.5%** of real in-corpus questions before this was caught. Each
  threshold now sits at that index's own measured p05 tail.
- **Groundedness** — cosine similarity between the generated answer and its retrieved
  context. Real answers score 0.756–0.902 against their context; answers paired with
  unrelated context score below 0.11 — the 0.40 threshold sits well clear of both.

`tests/test_guardrail_calibration.py` pins these thresholds to a false-refusal budget, so a
future retune can't silently regress them.

## Latency — three boundaries, not one number

The task's 200ms target covers "chunking + vector DB retrieval + everything through to
final output." This pipeline also makes two third-party network calls (speech-to-text,
generation) whose latency isn't ours to control, so one number would hide where time
actually goes. Measured against the **live deployment**, 30 real spoken-audio requests
(full method and per-stage breakdown in [`docs/LATENCY_REPORT.md`](docs/LATENCY_REPORT.md)):

| Boundary | Covers | P50 | P70 | P100 | Under 200ms |
|---|---|---|---|---|---|
| **A** | Retrieval + all three guardrails | **129.3ms** | **133.0ms** | **147.3ms** | **Yes** |
| B | A + answer generation (Claude) | 2,239ms | 2,412ms | 3,658ms | No |
| C | Full voice-to-answer, incl. speech-to-text | 3,356ms | 3,547ms | 4,682ms | No |

**Boundary A — retrieval and every guardrail this build owns — clears the 200ms target at
every percentile, with ~26% headroom even at P100.** Boundaries B and C don't, and can't:
Claude Haiku's own generation floor is ~650ms to first token, and ElevenLabs' speech-to-text
floor is ~1.1s, both measured independently of this pipeline and both provider-bound, not a
symptom of unoptimized code. `docs/LATENCY_REPORT.md` quantifies exactly where the time
goes and what's already identified (streaming generation, a persistent STT socket) as
headroom not yet spent.

## Frontend

- **Compare strategies** — one spoken question, answered independently against both chunking
  strategies from a single transcription, side by side.
- **Citations** — every answer shows the exact retrieved passages that grounded it.
- **Latency chart** — the per-stage breakdown above, rendered live per query.
- Hold the mic button *or* hold **Space** to talk (Wispr Flow-style); **Esc** drops an
  in-progress recording so you can redo it without submitting a bad take.

## Running locally

```bash
uv sync
cp .env.example .env   # fill in ELEVENLABS_API_KEY and ANTHROPIC_API_KEY
set -a && source .env && set +a
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` — `localhost` counts as a secure context, so microphone
capture works over plain HTTP locally (production requires TLS, see
`deploy/AWS_SETUP.md`).

```bash
uv run pytest -q          # test suite
uv run ruff check .       # lint
uv run ruff format --check .
```

## Repo layout

```
app/            FastAPI backend — stt, retrieval, generation, guardrails, harness
scripts/        Dataset ingest, chunking, indexing, benchmarking, threshold tuning
static/         Frontend (vanilla HTML/CSS/JS — no build step)
tests/          pytest suite, fully mocked at the external-call boundary
deploy/         AWS EC2 + Caddy setup, systemd unit, CI deploy
docs/           Latency reports, environment variable reference, implementation plans
```

Deploys automatically to the live link on every push to `main` (`.github/workflows/ci.yml`),
after tests, lint, and format checks pass.
