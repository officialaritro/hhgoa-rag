# Voice-Enabled RAG Pipeline Implementation Plan

Created: 2026-08-18
Agent: Claude Code
Status: PENDING
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** A user opens the live link, speaks a question in English, and receives a text answer generated from passages retrieved out of the MSMARCO-XI dataset — end to end through speech-to-text, chunked vector retrieval, grounded generation, and guardrails — with the pipeline's P50/P70/P100 latency measured and reported.

## Out of Scope

- **Text-to-speech / spoken answers.** The answer is displayed as text only (Answer-output decision, this planning session).
- **More than two chunking strategies.** Exactly two, per the Chunking-depth decision.
- **A model-based (LLM-judge) groundedness check.** The groundedness, off-topic, and unsafe-input checks are all rule-based/local — no second LLM call on the critical path (Groundedness-check decision).
- **Multi-turn conversation, chat history, or follow-up questions.** Single question, single answer, per the PRD.
- **User accounts, login, or per-user data.** The live link is a public, unauthenticated demo.
- **Support for more than one MSMARCO-XI language.** English only.
- **Mid-recording cancellation or abort.** The user can stop speaking or reload the page; no dedicated cancel control is built.
- **High-concurrency serving.** The single EC2 instance is sized for a handful of simultaneous demo viewers/judges, not production load.
- **Serverless or multi-instance deployment, autoscaling, load balancing.** One long-running process on one EC2 instance.

## Approach

**Chosen:** Single Python (FastAPI) service serving both a minimal static frontend and the RAG API, deployed as one long-running process on a single AWS EC2 instance, with a custom lightweight harness instead of an orchestration framework.

**Why:** Python carries the strongest ecosystem for every ML-heavy piece of this build — dataset loading, chunking, embeddings, and evaluating retrieval against the dataset's own relevance labels — and a single in-process service avoids the extra network hop a separate frontend service would add against an already-tight latency target. The `.nvmrc` Node 22 pin goes unused; per the PRD's own Technical Context, that file constrains only what runs on Node, not the whole stack. A custom harness (typed input/output per external call, one retry, a defined error path) is chosen over a framework like LangChain because framework overhead is a real cost against a 200ms target, not a convenience worth its abstraction here.

## Global Constraints

- Speech-to-text: ElevenLabs Scribe v2 Realtime, **streaming** (WebSocket) mode — not the blocking REST mode. A blocking call adds the full recording length to measured latency (PRD Engineering Risk 2).
- Dataset: `ai4bharat/MSMARCO-XI` (Hugging Face), English content only, drawn from the `Eng_Query` / `Eng_Answer` / `passages.English_passages` fields of the `train/hintrain.parquet` file specifically (confirmed real schema — see Context for Implementer). Corpus is capped at the **first 10,000 rows** (~100k passages) of that file's 778,638 total rows. This was revised down from an initial 50,000-row cap once semantic chunking's embedding cost was measured: semantic chunking needs one embed call per *sentence* for boundary detection (not just per chunk), which at 50,000 rows projected to ~3.4 hours combined with indexing both strategies — impractical for fast iteration against the Aug 22 deadline. At 10,000 rows the same pipeline is ~40 minutes.
- Chunking: exactly 2 distinct chunking strategies, differing in method (not just parameters).
- Answer output: text only. No text-to-speech.
- Off-topic, unsafe-input, and groundedness checks: rule-based/local (embedding-similarity thresholds, keyword/pattern matching). No second LLM call for guardrails.
- Generation: Anthropic Claude API, `claude-haiku-4-5` (confirmed current). Measured TTFT from the instance is **~650–780 ms** depending on context size — see Measured Infrastructure Facts. Reuse one client process-wide and prewarm it; a fresh connection per call costs ~200 ms more.
- Embeddings: a local, in-process embedding model — `sentence-transformers` with `all-MiniLM-L6-v2`, measured **11.8 ms/query** on the instance. (Benchmarked against `fastembed`+`bge-small-en-v1.5` at 38.9 ms; MiniLM-L6 wins because it is a 6-layer model against bge-small's 12.) No hosted embeddings API call on the critical path.
- **torch must resolve from the CPU-only index**, pinned in `pyproject.toml` via `[tool.uv.sources]`. The default wheel pulls ~4.6 GB of CUDA packages (`nvidia/*` 2.7 GB, `triton` 691 MB) that are unusable on this CPU instance and exhausted its 20 GiB volume mid-install. The pin takes the venv from 5.5 GB to 1.8 GB.
- Vector index: FAISS, in-process, in-memory — one index per chunking strategy.
- Hosting: a single always-on AWS EC2 `m7i-flex.large` instance (not serverless, not auto-scaled), fronted by Caddy terminating TLS on 443. The app binds `127.0.0.1` only.
- **TLS is mandatory, not cosmetic.** `getUserMedia`/`MediaRecorder` are secure-context-only APIs: served over plain HTTP, `navigator.mediaDevices` is `undefined` and the demo silently has no microphone — it fails with no error to debug. Let's Encrypt will not issue for `*.compute.amazonaws.com`, hence the DuckDNS hostname.
- Latency target: under 200ms at P50, P70, and P100, measured and reported at **three explicitly labelled boundaries** (Task 9). Measurement makes the shape clear: retrieval-core work is ~12 ms, while generation alone floors at ~650 ms and STT adds ~350 ms. Boundary A clears the target with wide margin; B and C cannot, for reasons outside our control. The report states all three plainly — per PRD Engineering Risk 1 and Acceptance Criteria 4, the requirement is real numbers, not flattering ones.
- Python tooling: `uv` for all Python operations (`uv run pytest`, `uv sync`, `uv run python -m ...`), not bare `pip`/`python3` — project standard, added mid-implementation. Dependencies live in `pyproject.toml`; `requirements.txt` is kept only as a plain reference list. Lint/format via `ruff check` / `ruff format`; type-check via `basedpyright` (errors must be zero; the large `reportUnknown*`/`reportAny` warning count from third-party stub gaps in pyarrow/huggingface_hub/faiss is accepted as-is, not chased to zero).

## Measured Infrastructure Facts

Measured on the provisioned instance, 2026-08-19. Do not re-derive; do not assume they generalise to another region or instance type.

**Instance:** `i-09e157bfae9bb82a6` · `ap-south-1b` (Mumbai) · Elastic IP `13.234.228.244` · Ubuntu 24.04 · **`m7i-flex.large`** — 2 vCPU Intel Xeon Platinum 8488C (Sapphire Rapids), **7.6 GB RAM**, 20 GiB gp3, 2 GB swap at `vm.swappiness=10`. M-family flex has no T-family CPU-credit model, so there is no burst exhaustion to distort P100.

The account is on the **AWS Free Plan**, which permits only free-tier-eligible instance types — both to launch and to resize into. `t3.medium` is rejected; `m7i-flex.large` (8 GB) and `c7i-flex.large` (4 GB) are permitted and are strictly better. Check `describe-instance-types --filters Name=free-tier-eligible,Values=true` before changing type: the rejection message ("this operation is not available for free plan accounts") misleadingly implicates the operation rather than the type.

**Network round-trips from the instance:**

| Endpoint | TCP | TLS | TTFB |
|---|---|---|---|
| `api.anthropic.com` | 2 ms | 22 ms | **25 ms** |
| `api.elevenlabs.io` | 3 ms | 73 ms | **350 ms** |

Anthropic has a Mumbai edge, so generation latency is model time, not network. ElevenLabs' origin is ~300 ms away — **the WebSocket must be opened before end-of-speech**, or the handshake alone exceeds the entire budget.

**Generation (Claude Haiku 4.5, streaming):**

| Retrieved context | TTFT mean | TTFT min |
|---|---|---|
| 1 passage (~300 chars) | 666 ms | 660 ms |
| 5 passages (~1.8k chars) | 707 ms | 627 ms |
| 10 passages (~3.8k chars) | 780 ms | 743 ms |

Two consequences: generation floors around **650 ms to first token**, 3× the whole budget; and **context is nearly free** — 12× more text costs ~115 ms, so Task 4 should choose `k` for retrieval quality, not latency.

**Speech-to-text (measured end to end, real audio through `app/stt.py`):** **~1.1–1.3 s per query**, not the ~350 ms suggested by raw API TTFB. The difference is the WebSocket handshake, audio upload, and the commit round-trip — all of which the raw TTFB probe excluded. Five real clips transcribed verbatim and correctly. This is the single largest stage in the pipeline; a client-side socket opened before the user speaks is the only structural way to reduce it.

**Guardrail thresholds — measured against the built indices, 2026-08-19:**

| Signal | In-corpus / grounded | Off-topic / ungrounded | Threshold set |
|---|---|---|---|
| Top retrieval similarity | 0.773 – 0.842 | 0.499 – 0.649 | **0.70** |
| Answer vs retrieved context | 0.756 – 0.902 | below 0.11 | **0.40** |

The original 0.3 off-topic threshold sat below the *lowest* off-topic score observed (0.386), so that guard never fired at all. `scripts/tune_thresholds.py` recommends a much lower groundedness value (~0.02) because it scores the dataset's terse `Eng_Answer` rather than real model output; generated answers quote the retrieved passages and score far higher, so the measured pipeline overrides that proxy. All three E2E scenarios verified live against these values: in-corpus answered, off-topic refused as `off-topic`, unsafe refused as `unsafe input`.

**Embedding:** `all-MiniLM-L6-v2` at **11.8 ms/query** (p50 11.8, max 13.3). Import + model load is ~17.5 s — startup cost, not per-query; the systemd unit allows `TimeoutStartSec=300`.

**HuggingFace throughput:** **74 MB/s from the instance** versus 4.8 MB/s from a residential laptop — 15× faster. Ingestion and index building therefore run *on the instance*; nothing GB-scale is uploaded. Measured: `scripts/ingest_dataset.py` completed 10,000 rows in **35 s including the 3.7 GB download**, peak RSS **3.82 GB**.

> That 3.82 GB peak comes from projecting the whole `passages` struct, which carries `Translated_passages`. It fits in 7.6 GB and works, but projecting `passages.English_passages` + `passages.is_selected` as nested paths would cut it substantially. Note the same whole-struct projection **OOM-kills** when reading over `HfFileSystem` instead of a downloaded file — download first, as the script already does.

**ElevenLabs keys — two exist.** The production key is IP-restricted to `13.234.228.244`: verified minting `sutkn_…` from the instance and correctly refused (`403`) elsewhere. It therefore **cannot be used for local development** — that needs the separate unrestricted key. Scope is Speech to Text only; no other scope is required. Minting costs ~405 ms, so fetch the token at page load, not on record-press. Regional residency endpoints (`api.sg.residency…` at 166 ms TTFB) are faster but **reject this account's key** — they need an Enterprise plan. Use the default host.

**Security group** `sg-01967e366d79ce0c8`: 22 from `152.58.139.215/32`, 80 and 443 from `0.0.0.0/0`, plus 8000 from `0.0.0.0/0` pending revocation at go-live. The SSH rule pins one residential IP, which rotates — if SSH starts timing out, re-authorise rather than debugging the instance.

**Public hostname:** `ragingoa.duckdns.org` → `13.234.228.244`, Let's Encrypt certificate valid to 16 Nov 2026, auto-renewing.

## Context for Implementer

**Corrected during implementation** (the PRD's and this plan's original wording, based on the HF dataset card's example code, was wrong — `load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")` does not work; `datasets` only exposes one `"default"` builder config that concatenates every language's training file together, too large to fit this machine's disk). The dataset's real structure, confirmed via `huggingface_hub.HfApi.list_repo_files` and `pyarrow.parquet.ParquetFile` schema inspection: each language has its own parquet file directly (`train/hintrain.parquet`, `train/bentrain.parquet`, etc., each ~3.7-4GB, 778,638 rows for Hindi), downloadable individually via `huggingface_hub.hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset", filename="train/hintrain.parquet")`. There is no cross-language comparison needed — the project uses exactly one file (`train/hintrain.parquet`) directly, not a merge across languages.

Confirmed real schema of that file: `Eng_Query` (string), `Eng_Answer` (string), and `passages` (struct with `English_passages: list<string>`, `Translated_passages: list<string>`, `is_selected: list<int64>` — `is_selected` is a list of 0/1 flags parallel-indexed to `English_passages`, not a per-passage nested field). Measured: ~10 passages per row on average.

## Runtime Environment

- **Start command:** `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`
- **Port:** 8000, **loopback only** — Caddy terminates TLS on 443 and reverse-proxies to it. Binding `0.0.0.0` would serve the app unencrypted and break the microphone.
- **Public URL:** `https://ragingoa.duckdns.org` · preflight diagnostic at `/preflight`
- **Health check:** `GET /health` — returns 200 once the vector indices and embedding/generation clients are loaded and ready
- **Restart procedure:** re-run the start command (or restart the systemd unit created in Task 10); the service is stateless aside from the on-disk FAISS indices it loads at startup

## Assumptions

- The judges/demo audience generates load in the range of a handful of concurrent requests, not a real production traffic pattern — Task 10's EC2 sizing depends on this.
- The first 10,000 rows of `train/hintrain.parquet` are a representative-enough sample for a demo corpus and for a meaningful recall@k measurement (Task 4) — not a random sample, since the file has no query_id ordering guarantee documented, but adequate for MVP purposes.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The <200ms target cannot be met end-to-end | **Confirmed by measurement** | Submission must report a missed target at the outer boundaries | Generation floors at ~650 ms TTFT and STT adds ~350 ms — both provider-bound. Task 9 reports three labelled boundaries so the ~12 ms retrieval core is visible separately from third-party time, per PRD Engineering Risk 1 and Acceptance Criteria 4 |
| Browser blocks the microphone because the live link is not HTTPS | **Resolved** | Demo would have no microphone at all — silent total failure | TLS is a Global Constraint; Caddy + Let's Encrypt live on `ragingoa.duckdns.org`, and `/preflight` verifies the browser preconditions independently of the app |
| Dependency install fills the disk | **Occurred, fixed** | Deploy fails with `No space left on device` | Default torch wheel pulls 4.6 GB of CUDA packages; `pyproject.toml` pins the CPU-only index (venv 5.5 GB → 1.8 GB) |
| Production ElevenLabs key is IP-locked to the instance | By design | Local development gets `403` on every token mint | A second unrestricted dev key exists; do not add rotating residential IPs to the production key |
| EC2 cold start / first-query model and index warm-up skews the P100 figure | Medium | P100 measurement misleading, target judged unfairly | Task 9's benchmark script runs and discards a stated number of warm-up queries before the measured batch, and states this explicitly in the report |
| A rule-based groundedness check may pass a subtly ungrounded answer or reject a well-grounded one | Medium | Guardrail under- or over-triggers | Task 7 sets and documents a similarity threshold; Task 7's Definition of Done requires demonstrating both a pass case and a refusal case |
| A 10,000-row subsample may under-represent the full corpus's topic diversity | Low-Medium | Off-topic guardrail or recall@k evaluation less representative than a full-corpus run | Documented explicitly in the benchmark/evaluation output (Task 4, Task 9) as a known scope reduction, not silently presented as full-corpus results |

## E2E Test Scenarios

### TS-001: Ask a question by voice and get a grounded answer
**Priority:** Critical
**Preconditions:** Live link loaded in browser; microphone permission granted; vector indices built and loaded
**Mapped Tasks:** Task 5, Task 6, Task 8

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to the live link | Page loads with a "record" control and an empty answer area |
| 2 | Click record, speak a question whose answer exists in the ingested corpus, stop recording | Transcript of the spoken question appears |
| 3 | Wait for the response | A text answer appears, drawn from retrieved passages; the per-stage and total latency for this query are shown or logged |

### TS-002: Off-topic question is refused, not guessed
**Priority:** High
**Preconditions:** Same as TS-001
**Mapped Tasks:** Task 7, Task 8

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Record a question clearly unrelated to the ingested corpus's subject matter | Transcript appears as usual |
| 2 | Wait for the response | The system responds that it cannot answer, instead of generating a guess; the refusal reason is visible or logged as "off-topic" |

### TS-003: Unsafe input is refused
**Priority:** High
**Preconditions:** Same as TS-001
**Mapped Tasks:** Task 7, Task 8

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Record a deliberately unsafe/inappropriate question | Transcript appears as usual |
| 2 | Wait for the response | The system refuses to answer; the refusal reason is visible or logged as "unsafe input" |

### TS-004: Empty or unintelligible speech does not crash the pipeline
**Priority:** Medium
**Preconditions:** Same as TS-001
**Mapped Tasks:** Task 5, Task 8

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click record, stay silent, stop recording | An empty or near-empty transcript is produced |
| 2 | Wait for the response | The system returns a clear "could not understand" response, not an error page or a hang |

### TS-005: Latency report reflects a real end-to-end run
**Priority:** Medium
**Preconditions:** Corpus ingested, indices built, service deployed to the live link
**Mapped Tasks:** Task 9

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run the latency benchmark script against the deployed service with its stated batch of test queries | The script completes and writes a report |
| 2 | Open the generated report | P50, P70, and P100 latency figures are present, along with whether the under-200ms target was met at each |

## Progress Tracking

- [x] Task 1: Project scaffolding and harness core
- [x] Task 2: Dataset ingestion and chunking strategy A (fixed-size)
- [x] Task 3: Chunking strategy B and vector indexing for both strategies
- [x] Task 4: Retrieval and retrieval-quality evaluation
- [x] Task 5: Speech-to-text integration (ElevenLabs streaming)
- [x] Task 6: Answer generation (Anthropic Claude)
- [x] Task 7: Guardrails (off-topic, unsafe-input, groundedness)
- [ ] Task 8: API wiring and minimal frontend
- [ ] Task 9: Latency benchmark harness
- [~] Task 10: AWS EC2 deployment — infrastructure, TLS, systemd and corpus/index build done; awaiting Task 8/9 completion and port-8000 revocation

## File Structure

- `pyproject.toml` (create) — dependencies and `uv`/ruff/pytest config (authoritative; `requirements.txt` kept as a plain reference list)
- `app/main.py` (create) — FastAPI app, routes, health check, static file serving
- `app/harness.py` (create) — typed call wrapper: retry-once + structured error path, used by every external call
- `app/schemas.py` (create) — Pydantic input/output models for every pipeline stage
- `app/embeddings.py` (create) — loads and runs the local embedding model
- `app/indexing.py` (create) — builds and loads FAISS indices, one per chunking strategy
- `app/retrieval.py` (create) — embeds a query and searches a chosen index
- `app/stt.py` (create) — ElevenLabs Scribe v2 Realtime streaming client
- `app/generation.py` (create) — Anthropic Claude call, grounded prompt construction
- `app/guardrails.py` (create) — off-topic, unsafe-input, and groundedness checks
- `scripts/ingest_dataset.py` (create) — loads MSMARCO-XI, extracts English fields, writes a corpus file
- `scripts/chunk_fixed_size.py` (create) — chunking strategy A
- `scripts/chunk_semantic.py` (create) — chunking strategy B
- `scripts/build_index.py` (create) — runs embeddings + FAISS index build for both strategies
- `scripts/evaluate_retrieval.py` (create) — recall@k against `is_selected` labels
- `scripts/generate_test_audio.py` (create) — synthesizes real speech (via ElevenLabs TTS) for the latency benchmark's test batch
- `scripts/benchmark_latency.py` (create) — runs the test batch against the live `/api/ask` endpoint, computes P50/P70/P100, writes the report
- `scripts/tune_thresholds.py` (create) — measures the score distributions the guardrail thresholds separate, and recommends values from data
- `static/index.html` (create) — minimal page: record control, transcript, answer, latency display
- `static/app.js` (create) — microphone capture, streams audio to the backend, renders the response
- `deploy/setup.sh` (create) — EC2 provisioning/systemd unit setup
- `deploy/README.md` (create) — deployment steps for the live link
- `tests/` (create) — one test module per `app/` and `scripts/` module listed above

## Implementation Tasks

### Task 1: Project scaffolding and harness core

**Objective:** Stand up the FastAPI project skeleton and the shared harness every external call (speech-to-text, generation) will use: a typed input/output schema per stage, a single-retry wrapper around transient failures, and a defined error result instead of an unhandled exception.

**Files:**

- Create: `pyproject.toml`
- Create: `app/main.py`
- Create: `app/harness.py`
- Create: `app/schemas.py`
- Test: `tests/test_harness.py`

**Key Decisions / Notes:**

- `app/harness.py` exposes one function, e.g. `run_stage(fn, input, retries=1) -> StageResult`, where `StageResult` is a Pydantic model with `ok: bool`, `value`, and `error: str | None`. Every later task (5, 6) wraps its external call with this, not with ad-hoc try/except.
- `app/schemas.py` defines one Pydantic model per stage boundary (STT input/output, retrieval input/output, generation input/output) — later tasks import from here rather than redefining shapes.
- `GET /health` in `app/main.py` returns 501/"not ready" until Task 8 wires it to real readiness checks; this task just establishes the route.

**Definition of Done:**

- [ ] `uvicorn app.main:app` starts without error and `GET /health` returns a response (not yet meaningful, just reachable)
- [ ] `run_stage` retries exactly once on a simulated transient failure, then returns an `ok=False` `StageResult` with an error message on a second failure, without raising
- [ ] Verify: `pytest tests/test_harness.py -q`

### Task 2: Dataset ingestion and chunking strategy A (fixed-size)

**Objective:** Download `train/hintrain.parquet` directly, extract the first 10,000 rows' English-only fields as the corpus, and implement the first chunking strategy — fixed-size splitting with overlap — over that corpus.

**Files:**

- Create: `scripts/ingest_dataset.py`
- Create: `scripts/chunk_fixed_size.py`
- Test: `tests/test_ingest_dataset.py`
- Test: `tests/test_chunk_fixed_size.py`

**Key Decisions / Notes:**

- Download via `huggingface_hub.hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset", filename="train/hintrain.parquet")`, then read only the first 10,000 rows via `pyarrow.parquet.ParquetFile.iter_batches` (do not load the full 778,638-row file into memory) — per the Global Constraints row cap.
- Per row, extract `Eng_Query`, `Eng_Answer`, `passages.English_passages` (list of strings) and `passages.is_selected` (parallel list of 0/1 flags, same index order as `English_passages` — confirmed real schema, see Context for Implementer). Deduplicate identical passage text across rows before chunking.
- Corpus output: the extracted English-only rows written to a single on-disk file (e.g. JSONL) that Task 3's indexing step reads.
- Fixed-size chunking: split each passage into token- or character-bounded windows with a stated overlap (e.g. 20%); record the source passage and its `is_selected` flag as metadata on every chunk, since Task 4's evaluation needs it.
- **Revised during implementation, per a review against the `chunking-strategy` skill:** `DEFAULT_CHUNK_SIZE` is 700 characters, not an earlier 200-character value. Fixed-size chunking should bound maximum chunk size, not fragment already-coherent units — MS MARCO passages average ~333 chars and max out at ~1233 chars (measured), so 700 chars lets most passages stay intact as one chunk while still capping the longer outliers. The skill's own guidance (256-1024 *tokens*, i.e. roughly 1000+ characters) confirmed 200 characters was well below even its smallest recommended tier.

**Definition of Done:**

- [ ] Running `scripts/ingest_dataset.py` produces a corpus file containing only the first 10,000 rows' English content
- [ ] Running `scripts/chunk_fixed_size.py` against that corpus produces chunks that each carry their source passage's `is_selected` metadata
- [ ] Verify: `pytest tests/test_ingest_dataset.py tests/test_chunk_fixed_size.py -q`

### Task 3: Chunking strategy B and vector indexing for both strategies

**Objective:** Implement a second chunking strategy that differs in method from Task 2's fixed-size splitting (semantic or metadata-aware splitting), then build a FAISS vector index for each of the two chunking strategies using a local embedding model.

**Files:**

- Create: `scripts/chunk_semantic.py`
- Create: `app/embeddings.py`
- Create: `app/indexing.py`
- Create: `scripts/build_index.py`
- Test: `tests/test_chunk_semantic.py`
- Test: `tests/test_indexing.py`

**Key Decisions / Notes:**

- `app/embeddings.py` loads one local embedding model at import time (module-level singleton) so later per-query calls (Task 4) don't reload it — this is a hot path per `development-practices.md`'s performance rule.
- `app/indexing.py` builds one FAISS index per chunking strategy and persists both to disk (e.g. `data/index_fixed_size.faiss`, `data/index_semantic.faiss`), alongside a parallel metadata store (chunk text, source passage, `is_selected`) keyed by FAISS row id.
- Semantic chunking (Task 3) differs from fixed-size (Task 2) in method, not parameters — implemented as embedding-similarity breakpoints between sentences (see `scripts/chunk_semantic.py`).
- **Added during implementation:** both indices persist as int8 scalar-quantized (`faiss.IndexScalarQuantizer`, `QT_8bit`) rather than raw float32 `IndexFlatIP` — ~4x smaller in memory, measured at 99.6% top-5 retrieval agreement with the uncompressed index, needed to fit both strategies' indices resident on the `t3.small` EC2 instance chosen for Task 10 (see `deploy/AWS_SETUP.md`).
- **Revised during implementation, per the `chunking-strategy` skill:** `DEFAULT_SIMILARITY_THRESHOLD` in `scripts/chunk_semantic.py` is 0.8, not an earlier 0.5. General sentence-embedding similarity between unrelated sentences commonly sits in the 0.3-0.6 range (embedding-space anisotropy), so 0.5 under-split — merging sentences that weren't actually thematically connected. The skill recommends 0.8 for embedding-based semantic boundary detection. **Deferred, not applied:** the skill also recommends comparing sentences via a 3-5 sentence buffer window rather than adjacent pairs only, and an iterate-until-target validation loop (cohesion score, retrieval precision/recall) — both skipped given the Aug 22 deadline; the PRD only requires multiple *distinct* chunking strategies, not a specific quality bar.
- **Performance fix during implementation:** both `app/indexing.py`'s `build_index` and `scripts/chunk_semantic.py`'s sentence-boundary detection call `app/embeddings.py`'s new `embed_batch()` (one call for a whole list of texts) instead of `embed()` in a per-item loop. The per-item version made a real `build_index` run take far longer than the batched-throughput estimate (~354 texts/sec) projected, since per-call model overhead dominates when texts aren't batched — discovered mid-run on the 10,000-row corpus (~486k total chunks).

**Definition of Done:**

- [ ] `scripts/chunk_semantic.py` produces chunks whose boundaries differ from `chunk_fixed_size.py`'s output on the same corpus (not just re-parameterized)
- [ ] `scripts/build_index.py` produces two loadable FAISS indices, one per strategy, each with a matching metadata store
- [ ] Verify: `pytest tests/test_chunk_semantic.py tests/test_indexing.py -q`

### Task 4: Retrieval and retrieval-quality evaluation

**Objective:** Given a text query, embed it and search a chosen chunking strategy's FAISS index to return a ranked set of passages, then measure retrieval quality against the dataset's own `is_selected` relevance labels.

**Files:**

- Create: `app/retrieval.py`
- Create: `scripts/evaluate_retrieval.py`
- Test: `tests/test_retrieval.py`

**Key Decisions / Notes:**

- `app/retrieval.py` exposes `retrieve(query: str, strategy: str, k: int) -> RetrievalOutput`, where `RetrievalOutput` (defined in Task 1's `app/schemas.py`) wraps a list of retrieved-passage items. Task 6 consumes this same `RetrievalOutput` — do not introduce a second name for it.
- `scripts/evaluate_retrieval.py` computes recall@k (does the top-k retrieved set include a passage flagged `is_selected=1` for that query) for at least one strategy, over a sample of ingested queries — this is PRD Acceptance Criteria 3.
- **Added during implementation:** `app/retrieval.py` caches each strategy's loaded index+metadata per-process via `functools.cache` on a private `_load_cached` helper (public `INDEX_PATHS` dict), rather than reloading from disk on every call — reloading (and re-unpickling metadata) on every request cost both latency budget and memory churn.
- **Real measured recall@5** (10,000-row corpus, 30-query sample, `scripts/evaluate_retrieval.py`): fixed_size = 0.790, semantic = 0.661. Fixed-size's larger 700-char chunks apparently preserve more complete passage context per chunk, which helps recall against the passage-level `is_selected` ground truth here — semantic chunking's smaller sub-passage chunks (338,544 vs 101,131 total, ~3.3x more) each carry less complete context. This is a genuine empirical result for the submission's "vast chunking" comparison, not a bug.

**Definition of Done:**

- [ ] `retrieve()` returns passages ranked by similarity for a sample query, for both chunking strategies
- [ ] `scripts/evaluate_retrieval.py` reports a recall@k number for at least one strategy, computed from the dataset's own `is_selected` labels (not a hand-derived expectation)
- [ ] Verify: `pytest tests/test_retrieval.py -q`

### Task 5: Speech-to-text integration (ElevenLabs streaming)

**Objective:** Wire the ElevenLabs Scribe v2 Realtime streaming API into the harness from Task 1, converting a spoken audio question into a text transcript, and define the behavior for empty or unintelligible audio.

**Files:**

- Create: `app/stt.py`
- Test: `tests/test_stt.py`

**Key Decisions / Notes:**

- `app/stt.py` uses ElevenLabs' streaming (WebSocket) mode per the Global Constraints — not the blocking REST mode.
- Wrapped via `app/harness.py`'s `run_stage`: a transient connection failure retries once; a persistent failure returns an `ok=False` `StageResult` that Task 8 turns into a user-facing "could not process audio" response, not a crash.
- Empty or near-empty transcript (silence, unintelligible speech) is a valid `ok=True` result with an empty/short string — Task 8 (not this task) decides what response the user sees for it (TS-004).
- Tests mock the ElevenLabs WebSocket client; no live API calls in the test suite.

**Definition of Done:**

- [ ] Given a mocked streaming response, `app/stt.py` returns the assembled transcript text
- [ ] Given a mocked connection failure that succeeds on the second attempt, the call succeeds via the harness's one retry
- [ ] Given a persistent mocked failure, the call returns an `ok=False` result rather than raising
- [ ] Verify: `pytest tests/test_stt.py -q`

### Task 6: Answer generation (Anthropic Claude)

**Objective:** Wire an Anthropic Claude API call into the harness to generate an answer from a query and its retrieved passages, with the prompt explicitly instructing the model to answer only from the supplied context.

**Files:**

- Create: `app/generation.py`
- Test: `tests/test_generation.py`

**Key Decisions / Notes:**

- Confirm the current Claude model ID via the `claude-api` skill or Anthropic's docs before writing the API call — do not hardcode a guessed model ID (Global Constraints).
- Prompt construction takes the `RetrievalOutput` from Task 4 and instructs the model to answer strictly from the provided passages and to say so explicitly if the passages don't contain an answer — this drafted-answer signal is what Task 7's groundedness check inspects.
- Wrapped via `app/harness.py`'s `run_stage`, matching Task 5's retry/error-path pattern.
- Tests mock the Anthropic client; no live API calls in the test suite.

**Definition of Done:**

- [ ] Given a mocked model response and a set of retrieved passages, `app/generation.py` returns an answer string grounded in those passages
- [ ] Given a persistent mocked API failure, the call returns an `ok=False` result rather than raising
- [ ] Verify: `pytest tests/test_generation.py -q`

### Task 7: Guardrails (off-topic, unsafe-input, groundedness)

**Objective:** Implement the three rule-based guardrail checks: an off-topic check using retrieval relevance, an unsafe-input check using pattern matching, and a groundedness check comparing the generated answer against the retrieved passages — each producing a "cannot answer" result with a stated reason on failure.

**Files:**

- Create: `app/guardrails.py`
- Test: `tests/test_guardrails.py`

**Key Decisions / Notes:**

- Off-topic check: if the top retrieved passage's similarity score (from Task 4) falls below a stated threshold, treat the query as off-topic — reuses retrieval infrastructure instead of adding a separate model call.
- Unsafe-input check: a keyword/pattern-based filter over the transcript text, checked before retrieval runs (cheapest check first).
- Groundedness check: embedding-similarity between the generated answer's embedding (via Task 3's `app/embeddings.py`) and the retrieved context's embedding; below a stated threshold, the answer is treated as ungrounded and replaced with a refusal.
- Every threshold used here is a literal constant defined in this file with a one-line comment explaining its origin — Task 9's benchmark run is a further signal on whether they need retuning.
- **`scripts/tune_thresholds.py` replaces guessing with measurement.** It samples in-corpus queries against clearly off-topic ones, reports both top-score distributions, and places the off-topic threshold between the off-topic p95 and the in-corpus p05 — the two values that actually govern false accepts and false rejects. Groundedness needs no LLM call: an `Eng_Answer` paired with its own query's retrieved context is grounded by construction, and the same answer against a different query's context is not. Run it once the indices exist, then set the constants from its output.
- The untuned failure mode that matters is **refusing a valid question live on stage**, which is worse than answering a marginal one — if the distributions overlap, bias the off-topic threshold low.

**Definition of Done:**

- [ ] A query with low retrieval relevance is flagged off-topic and produces a "cannot answer" result (TS-002)
- [ ] A transcript matching the unsafe-input pattern list is flagged and produces a "cannot answer" result before retrieval runs (TS-003)
- [ ] A generated answer with low similarity to its retrieved context is flagged ungrounded and replaced with a refusal; a well-grounded answer passes through unchanged
- [ ] Verify: `pytest tests/test_guardrails.py -q`

### Task 8: API wiring and minimal frontend

**Objective:** Wire Tasks 5–7 into a single FastAPI endpoint that takes spoken audio and returns a text answer (or a refusal), serve a minimal static page for microphone capture and display, and make `GET /health` reflect real readiness.

**Files:**

- Modify: `app/main.py`
- Create: `static/index.html`
- Create: `static/app.js`
- Test: `tests/test_api.py`

**Key Decisions / Notes:**

- Request flow in the new endpoint: STT (Task 5) → unsafe-input check (Task 7) → retrieval (Task 4) → off-topic check (Task 7) → generation (Task 6) → groundedness check (Task 7) → response. Any `ok=False` `StageResult` along the way short-circuits to a user-facing error/refusal, never a 500 with a stack trace.
- `GET /health` now checks that both FAISS indices are loaded and the embedding model is initialized; returns 200 only when ready.
- `static/app.js` uses the Web Audio API (`getUserMedia` + `AudioContext`/`ScriptProcessorNode`) to capture raw 16kHz PCM and stream it to the backend, not `MediaRecorder` — `MediaRecorder` produces compressed WebM/Opus, which does not match what Task 5's STT client expects (`AudioFormat.PCM_16000`). `static/index.html` shows the record control, transcript, answer text, and per-query latency.
- Empty/unintelligible transcript (Task 5) is handled here as a distinct "could not understand" response, satisfying TS-004 — this is the first task that decides that behavior, per Task 5's note.

**Definition of Done:**

- [ ] TS-001 (ask a question, get a grounded answer) passes via browser automation against a locally running instance
- [ ] TS-004 (silent/empty audio produces a clear response, not a crash) passes
- [ ] `GET /health` returns 200 only after indices and models are loaded, and a non-200 status before that
- [ ] Verify: `pytest tests/test_api.py -q`

### Task 9: Latency benchmark harness

**Objective:** Build a script that runs a stated batch of test queries through the full deployed pipeline, records end-to-end latency for each, discards a stated warm-up window, and computes and reports P50, P70, and P100 latency plus whether the under-200ms target was met at each.

**Files:**

- Create: `scripts/benchmark_latency.py`
- Create: `scripts/generate_test_audio.py`
- Test: `tests/test_benchmark.py`
- Test: `tests/test_generate_test_audio.py`

**Key Decisions / Notes:**

- Batch size: at least 30 test queries (PRD Acceptance Criteria 5's "a reasonable number") — `DEFAULT_BATCH_SIZE = 30`.
- Warm-up handling: the first N queries (`DEFAULT_WARMUP_QUERIES = 3`) run and are discarded before the measured batch begins, mitigating the cold-start risk (PRD Engineering Risk 3 / this plan's Risks table).
- **Added during implementation — how "full pipeline including speech-to-text" gets measured without real human recordings:** `scripts/generate_test_audio.py` synthesizes a small fixed set of representative questions into real 16kHz PCM audio via ElevenLabs' own TTS API (`output_format="pcm_16000"`, matching what Task 5's STT client expects), caching the clips to `data/benchmark_audio/`. `scripts/benchmark_latency.py`'s `run_benchmark` cycles through this small clip set to reach the requested batch size and sends each through the real, live `/api/ask` endpoint via `run_batch` — real STT, retrieval, generation, and guardrails on every call, timed end to end. A latency benchmark does not need query diversity (unlike retrieval-quality evaluation), only enough real round trips for a stable percentile, so cycling a handful of clips is sufficient.
- **Scope reduction from the original per-stage-timing ambition:** this measures end-to-end latency only, not a per-stage (STT/retrieval/generation/guardrails) breakdown — that would need timing instrumentation added inside `app/main.py`'s endpoint itself, cut for time. PRD Acceptance Criteria 4/5 only require the end-to-end P50/P70/P100 numbers, which this satisfies.
- `run_benchmark(audio_paths, base_url=...)`: `base_url=None` benchmarks the app in-process (local dev, via `TestClient`); a real URL benchmarks the deployed instance over the network (Task 10) — the submission's numbers must come from the latter, per PRD Open Decision 4.
- Report output states, for each of P50/P70/P100: the measured value and whether it is under 200ms — matching PRD Acceptance Criteria 4's requirement to state this plainly regardless of outcome.
- **The cut per-stage breakdown is still available from infrastructure measurement.** Because the harness reports end-to-end only, the report should cite the independently measured per-stage figures from Measured Infrastructure Facts alongside it, so a ~1.2 s end-to-end number is legible rather than looking like an unexplained miss:

  | Stage | Measured | In our control? |
  |---|---|---|
  | Query embedding | 11.8 ms | Yes |
  | FAISS search | low single-digit ms | Yes |
  | Generation (TTFT, Haiku 4.5) | ~650–780 ms | No — provider floor |
  | Speech-to-text round trip | ~350 ms | No — provider, ~300 ms from Mumbai |

  The honest framing for the submission: the retrieval core — the part the brief describes as *"chunking + vector DB retrieval"* — runs in roughly 12 ms, an order of magnitude inside the target. The remainder is third-party model and network time that no amount of local optimisation removes. State that plainly; a transparent breakdown reads far better than a single unexplained figure.

**Definition of Done:**

- [ ] Running the script against a locally running instance produces a report with P50, P70, and P100 values and a pass/fail statement against 200ms for each
- [ ] The report explicitly states the warm-up queries were excluded from the measured percentiles
- [ ] Verify: `pytest tests/test_benchmark.py tests/test_generate_test_audio.py -q`

### Task 10: AWS EC2 deployment

**Objective:** Deploy the service to a single always-on AWS EC2 instance as one long-running process, produce the live link, and run Task 9's benchmark against the deployed instance to produce the submission's real latency numbers.

**Files:**

- Create: `deploy/setup.sh`
- Create: `deploy/README.md`
- Create: `deploy/AWS_SETUP.md`
- Create: `deploy/Caddyfile`
- Create: `deploy/preflight.html`
- Test: none (deployment/infrastructure task — verified by live checks below, not a unit test)

**Key Decisions / Notes:**

**Applied and verified 2026-08-19** — the instance is provisioned, TLS is live, and the service runs. See `deploy/AWS_SETUP.md` for the full change log.

- Instance resized to `m7i-flex.large` (7.6 GB), root volume grown 8 → 20 GiB, 2 GB swapfile added, ports 80/443 opened. Elastic IP survived the stop/start, so the live link never moved.
- `deploy/setup.sh` installs `uv` and Caddy, runs `uv sync`, builds the corpus and both FAISS indices when `data/` is empty, writes the `voice-rag` systemd unit, and configures TLS. Idempotent.
- **The app binds `127.0.0.1:8000`, never `0.0.0.0`.** Caddy terminates TLS on 443 and reverse-proxies to it. Serving the app directly on 8000 over plain HTTP would disable the microphone (secure-context requirement).
- `deploy/Caddyfile` also serves `/preflight` from `/opt/hhgoa-rag-preflight` independently of the app, so the browser-precondition check works even when the backend is down.
- **Corpus and indices are built on the instance, not shipped to it.** Measured 74 MB/s to HuggingFace there against 4.8 MB/s residential; ingestion of 10,000 rows including the 3.7 GB download took 35 s. Only code moves over `git pull`; `data/` is gitignored.
- After deployment, re-run `scripts/benchmark_latency.py` against the live HTTPS URL (not localhost) — PRD Open Decision 4 means the submitted P50/P70/P100 numbers must reflect the real deployed path.
- **Revoking public port 8000 is the last step**, only once 443 serves the app — it is currently the only fallback route in.

**Definition of Done:**

- [x] Instance provisioned, resized, TLS live — certificate valid to 16 Nov 2026
- [x] `voice-rag.service` enabled and running; survives reboot
- [x] Corpus ingested and chunked on the instance
- [ ] Both FAISS indices built and loaded
- [ ] The live URL serves the frontend and answers a real spoken question end to end (TS-001 re-run against the live link), with the microphone actually granted by the browser
- [ ] `/preflight` is all-green on the machine the demo will be given from
- [ ] `GET https://ragingoa.duckdns.org/health` returns 200
- [ ] Public port 8000 revoked; `curl --max-time 5 http://13.234.228.244:8000/health` times out
- [ ] `scripts/benchmark_latency.py` has been run against the live HTTPS URL and its report is saved under `deploy/` or `docs/` for the submission
