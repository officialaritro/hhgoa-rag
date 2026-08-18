# Voice-Enabled RAG Pipeline Implementation Plan

Created: 2026-08-18
Agent: Claude Code
Status: PENDING
Approved: No
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
- Dataset: `ai4bharat/MSMARCO-XI` (Hugging Face), English content only, drawn from the `Eng_Query` / `Eng_Answer` / `English_passages` fields — the dataset has no dedicated `en` config; English content lives embedded inside each per-language config.
- Chunking: exactly 2 distinct chunking strategies, differing in method (not just parameters).
- Answer output: text only. No text-to-speech.
- Off-topic, unsafe-input, and groundedness checks: rule-based/local (embedding-similarity thresholds, keyword/pattern matching). No second LLM call for guardrails.
- Generation: Anthropic Claude API. Confirm the current model ID via the `claude-api` skill or Anthropic's docs at implementation time — do not hardcode a guessed model ID.
- Embeddings: a local, in-process embedding model. No hosted embeddings API call on the critical path.
- Vector index: FAISS, in-process, in-memory — one index per chunking strategy.
- Hosting: a single always-on AWS EC2 instance (not serverless, not auto-scaled).
- Latency target: the full pipeline (audio in → answer out, including speech-to-text) under 200ms at P50, P70, and P100. Per PRD Engineering Risk 1, this is a stretch target the pipeline may not fully clear — the benchmark report (Task 9) must state the actual measured numbers regardless of outcome.

## Context for Implementer

`ai4bharat/MSMARCO-XI` is loaded per-language, e.g. `load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")` — there is no `"en"` config. Each row, regardless of which language config it comes from, carries the original English content in `Eng_Query`, `Eng_Answer`, and `passages.English_passages`, alongside that config's translated fields (which this project ignores). It is not yet confirmed whether every per-language config covers the same underlying set of English source rows or different subsets — Task 2 must check this against the actual loaded data (e.g. compare `query_id` coverage across two configs) before committing to one config as the corpus source, rather than assuming.

## Runtime Environment

- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Port:** 8000
- **Health check:** `GET /health` — returns 200 once the vector indices and embedding/generation clients are loaded and ready
- **Restart procedure:** re-run the start command (or restart the systemd unit created in Task 10); the service is stateless aside from the on-disk FAISS indices it loads at startup

## Assumptions

- Every per-language MSMARCO-XI config exposes the same English source rows (same `query_id` space) — Task 2 depends on this and must verify it before ingestion proceeds; if false, Task 2 picks the config with the fullest English coverage instead.
- The judges/demo audience generates load in the range of a handful of concurrent requests, not a real production traffic pattern — Task 10's EC2 sizing depends on this.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The full-pipeline <200ms target at all three percentiles is very unlikely to be met even with every latency-favoring choice made in this plan | High | Submission may have to report a missed target | Task 9's benchmark report states the actual P50/P70/P100 numbers transparently regardless of outcome, per PRD Engineering Risk 1 and Acceptance Criteria 4 |
| EC2 cold start / first-query model and index warm-up skews the P100 figure | Medium | P100 measurement misleading, target judged unfairly | Task 9's benchmark script runs and discards a stated number of warm-up queries before the measured batch, and states this explicitly in the report |
| A rule-based groundedness check may pass a subtly ungrounded answer or reject a well-grounded one | Medium | Guardrail under- or over-triggers | Task 7 sets and documents a similarity threshold; Task 7's Definition of Done requires demonstrating both a pass case and a refusal case |
| Per-language MSMARCO-XI configs may not share the same English row set | Low-Medium | Wrong or incomplete corpus ingested | Task 2 verifies row consistency across configs before committing to one, per its Key Decisions |

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

- [ ] Task 1: Project scaffolding and harness core
- [ ] Task 2: Dataset ingestion and chunking strategy A (fixed-size)
- [ ] Task 3: Chunking strategy B and vector indexing for both strategies
- [ ] Task 4: Retrieval and retrieval-quality evaluation
- [ ] Task 5: Speech-to-text integration (ElevenLabs streaming)
- [ ] Task 6: Answer generation (Anthropic Claude)
- [ ] Task 7: Guardrails (off-topic, unsafe-input, groundedness)
- [ ] Task 8: API wiring and minimal frontend
- [ ] Task 9: Latency benchmark harness
- [ ] Task 10: AWS EC2 deployment

## File Structure

- `requirements.txt` (create) — pinned Python dependencies
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
- `scripts/benchmark_latency.py` (create) — runs the test batch, computes P50/P70/P100, writes the report
- `static/index.html` (create) — minimal page: record control, transcript, answer, latency display
- `static/app.js` (create) — microphone capture, streams audio to the backend, renders the response
- `deploy/setup.sh` (create) — EC2 provisioning/systemd unit setup
- `deploy/README.md` (create) — deployment steps for the live link
- `tests/` (create) — one test module per `app/` and `scripts/` module listed above

## Implementation Tasks

### Task 1: Project scaffolding and harness core

**Objective:** Stand up the FastAPI project skeleton and the shared harness every external call (speech-to-text, generation) will use: a typed input/output schema per stage, a single-retry wrapper around transient failures, and a defined error result instead of an unhandled exception.

**Files:**

- Create: `requirements.txt`
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

**Objective:** Load the MSMARCO-XI dataset, resolve which per-language config's English fields to use as the corpus, and implement the first chunking strategy — fixed-size splitting with overlap — over that corpus.

**Files:**

- Create: `scripts/ingest_dataset.py`
- Create: `scripts/chunk_fixed_size.py`
- Test: `tests/test_ingest_dataset.py`
- Test: `tests/test_chunk_fixed_size.py`

**Key Decisions / Notes:**

- Per Context for Implementer above: before committing to one language config, `scripts/ingest_dataset.py` must load at least two configs (e.g. `"hi"` and `"bn"`) and compare `query_id` coverage. If they match, pick either; if they diverge, pick the config with the fullest English coverage and note the finding in a code comment.
- Corpus output: `Eng_Query`, `Eng_Answer`, and each row's `English_passages` (deduplicated across rows), written to a single on-disk file (e.g. JSONL) that Task 3's indexing step reads.
- Fixed-size chunking: split each passage into token- or character-bounded windows with a stated overlap (e.g. 20%); record the source passage and `is_selected` flag as metadata on every chunk, since Task 4's evaluation needs it.

**Definition of Done:**

- [ ] Running `scripts/ingest_dataset.py` produces a corpus file containing only English content, with a logged decision on which language config was used and why
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
- `app/indexing.py` builds one FAISS index per chunking strategy and persists both to disk (e.g. `data/index_fixed.faiss`, `data/index_semantic.faiss`), alongside a parallel metadata store (chunk text, source passage, `is_selected`) keyed by FAISS row id.
- Semantic chunking (Task 3) differs from fixed-size (Task 2) in method, not parameters — e.g. splitting on embedding-similarity breakpoints between sentences, or splitting on passage/metadata boundaries rather than a fixed window.

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
- Retrieval latency (embed + FAISS search) is logged per call here — Task 9 reuses this instrumentation rather than re-adding it.

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
- Every threshold used here is a literal constant defined in this file with a one-line comment explaining its origin (chosen empirically during this task, not derived from a formula) — Task 9's benchmark run is the first real signal on whether thresholds need retuning.

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
- `static/app.js` uses the browser `MediaRecorder`/`getUserMedia` APIs to capture microphone audio and stream it to the backend; `static/index.html` shows the record control, transcript, answer text, and per-query latency.
- Empty/unintelligible transcript (Task 5) is handled here as a distinct "could not understand" response, satisfying TS-004 — this is the first task that decides that behavior, per Task 5's note.

**Definition of Done:**

- [ ] TS-001 (ask a question, get a grounded answer) passes via browser automation against a locally running instance
- [ ] TS-004 (silent/empty audio produces a clear response, not a crash) passes
- [ ] `GET /health` returns 200 only after indices and models are loaded, and a non-200 status before that
- [ ] Verify: `pytest tests/test_api.py -q`

### Task 9: Latency benchmark harness

**Objective:** Build a script that runs a stated batch of test queries through the full deployed pipeline, records per-stage and total latency for each, discards a stated warm-up window, and computes and reports P50, P70, and P100 latency plus whether the under-200ms target was met at each.

**Files:**

- Create: `scripts/benchmark_latency.py`
- Test: `tests/test_benchmark.py`

**Key Decisions / Notes:**

- Batch size: at least 30 test queries (PRD Acceptance Criteria 5's "a reasonable number").
- Warm-up handling: the first N queries (stated explicitly in the report, e.g. 3) run and are discarded before the measured batch begins, mitigating the cold-start risk (PRD Engineering Risk 3 / this plan's Risks table).
- Per-query timing captures each stage (STT, retrieval, generation, guardrails) plus the end-to-end total, so the report can show where time actually goes, not just the final number.
- Report output states, for each of P50/P70/P100: the measured value and whether it is under 200ms — matching PRD Acceptance Criteria 4's requirement to state this plainly regardless of outcome.

**Definition of Done:**

- [ ] Running the script against a locally running instance with a mocked STT/generation backend produces a report with P50, P70, and P100 values and a pass/fail statement against 200ms for each
- [ ] The report explicitly states the warm-up queries were excluded from the measured percentiles
- [ ] Verify: `pytest tests/test_benchmark.py -q`

### Task 10: AWS EC2 deployment

**Objective:** Deploy the service to a single always-on AWS EC2 instance as one long-running process, produce the live link, and run Task 9's benchmark against the deployed instance to produce the submission's real latency numbers.

**Files:**

- Create: `deploy/setup.sh`
- Create: `deploy/README.md`
- Test: none (deployment/infrastructure task — verified by live checks below, not a unit test)

**Key Decisions / Notes:**

- `deploy/setup.sh` provisions the instance (installs Python, project dependencies, systemd unit running the Task 1 start command) so the service restarts automatically if the instance reboots.
- `deploy/README.md` documents the exact steps and the resulting live URL, for the submission's "Live working link" requirement.
- After deployment, re-run `scripts/benchmark_latency.py` against the live URL (not localhost) — the PRD's Open Decision 4 (hosting/network conditions) means the submitted P50/P70/P100 numbers should reflect the real deployed path, not just a local run.

**Definition of Done:**

- [ ] The live URL serves the frontend and answers a real spoken question end to end (TS-001 re-run against the live link)
- [ ] `GET /health` on the live URL returns 200
- [ ] `scripts/benchmark_latency.py` has been run against the live URL and its report is saved under `deploy/` or `docs/` for the submission
- [ ] Verify: manual check — `curl https://<live-url>/health` returns 200, and the saved latency report exists
