# Voice-Enabled RAG Pipeline (HH Goa 2026, Task 2)

Created: 2026-08-17
Agent: Claude Code
Category: Feature
Status: Final
Research: Standard

## Problem Statement

HH Goa 2026 sets a shortlisting task. The task requires one team to build a voice-enabled Retrieval-Augmented Generation (RAG) system. A user speaks a question. The system must convert the speech to text. The system must retrieve relevant passages from a fixed dataset. The system must generate an answer from those passages. The team must submit the system by August 22, 2026, 11:59 PM. The rules do not allow resubmission. The scope of this PRD must fit inside that fixed deadline.

The source requirements document is `task 2_ hhg.md` in the project root. This PRD treats that document as the single source of truth. This PRD does not add requirements the source document does not state. Where the source document leaves a decision open, this PRD names the decision. This PRD does not silently pick an answer for an open decision.

## Core User Flows

### Flow 1: Ask a question by voice

1. The user speaks a question into a microphone.
2. The speech-to-text (STT) stage converts the audio to a text transcript.
3. The retrieval stage searches the vector index and returns the most relevant passages from the ingested dataset.
4. The generation stage writes an answer, using only the retrieved passages as source material.
5. The system shows the answer to the user. (Open Decision 1 covers whether the system must also speak the answer.)
6. The system records the time each stage took.

### Flow 2: Off-topic or unsafe question

1. The user speaks a question that the dataset does not cover, or a question that is unsafe or inappropriate.
2. The STT stage converts the audio to text, the same as Flow 1.
3. A guardrail stage checks the question, the retrieved passages, or the drafted answer.
4. If the guardrail stage finds a problem, the system tells the user it cannot answer. The system does not guess.
5. The system records why it refused to answer.

### Flow 3: Produce the latency report

1. An evaluator runs the pipeline over a batch of test questions.
2. The harness records the time for each pipeline stage, for every test question.
3. The harness computes the P50, P70, and P100 latency across the full batch.
4. The team includes these three numbers in the submission.

## Scope

### In Scope (see MVP section for the minimum build)

- Voice input capture from a microphone.
- Speech-to-text conversion, using the ElevenLabs Scribe v2 Realtime API.
- Ingestion of the MSMARCO-XI dataset, English-language content.
- At least two distinct chunking strategies over the ingested text.
- A vector index, built from the chunked text, that supports similarity search.
- Retrieval of relevant passages for a given text query.
- Answer generation, grounded in the retrieved passages.
- A harness layer around every model or API call: a defined input/output shape, at least one retry on failure, and a defined error path.
- Guardrails: a check for off-topic questions, a check for unsafe input, and a groundedness or hallucination check on the generated answer.
- Latency instrumentation, per stage and end to end, across a batch of test queries, with P50/P70/P100 reported.
- A live, working, deployed version of the system.
- A GitHub repository holding the code.

### Explicitly Out of Scope

- **Multi-turn conversation.** The source document describes one question and one answer. It does not describe follow-up questions or chat history.
- **User accounts, login, or per-user data.** The live link is a public demonstration. The source document does not ask for access control.
- **Support for all 14 MSMARCO-XI languages at once.** The source document does not require multi-language support. Building and testing against every language would not fit the deadline.
- **Spoken (text-to-speech) answer output, as a required MVP feature.** The source document's own pipeline diagram stops at "Answer generation." See Open Decision 1.
- **Production-scale hosting** (autoscaling, multi-region, load balancing). This is a demonstration for judges, not a production service. A single working instance meets the "live working link" requirement.
- **Ingestion of any dataset other than MSMARCO-XI.** The source document names one dataset.
- **Billing, usage quotas, or per-user API keys.** Not requested, and not relevant to a shortlisting demo.
- **A native mobile app or browser extension.** "Live working link" points to a web-reachable system. No mobile requirement is stated.
- **Video production and social media posting.** The source document does require two videos and posts on Instagram, X, and LinkedIn with the tag `#RAGInGoa`. This is a real requirement of the task, but it is not a software requirement. `/spec` builds code; it does not record video or post to social media. This work item is tracked in the Non-Engineering Requirements section below, so it is not lost, but it stays outside the build.

## Technical Context

- **Existing code:** none. The project directory holds only the source requirements file and a `.nvmrc` file that pins Node.js version 22. This is a new build. There is no existing architecture to preserve or work around.
- **Runtime constraint:** the `.nvmrc` file pins Node.js 22. Any part of the system that runs on Node.js must run on Node.js 22. This does not force every part of the system onto Node.js — a Python-based retrieval or embedding service, for example, is not ruled out by this file.
- **Dataset facts** (confirmed from the official Hugging Face dataset card at `huggingface.co/datasets/ai4bharat/MSMARCO-XI`):
  - The dataset is the MS MARCO question-answering dataset, machine-translated into 14 Indic languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, and Urdu.
  - Each dataset record holds: a query, an answer, a set of passages, and an `is_selected` flag per passage that marks which passages are the true relevant source for the answer. Each record also keeps the original English query, answer, and passages.
  - The dataset card does not publish a total record count or a per-language record count. `/spec` must check the actual row counts against the dataset viewer or by loading the data. This PRD does not state a row count, because none was confirmed.
- **Speech-to-text vendor facts:** the source document allows exactly one of two vendors: Sarvam AI or ElevenLabs. The team has chosen **ElevenLabs Scribe v2 Realtime**. ElevenLabs publishes a stated latency of 150 milliseconds for this API. This is the vendor's own published figure; this PRD has not independently measured it. (For reference, the vendor not chosen: Sarvam AI documents three API modes — REST for files under 30 seconds, Batch for files up to 2 hours, and Realtime/WebSocket for streaming partial transcripts — but its public documentation does not state a specific millisecond latency figure for any mode.)
- **Latency shape of the other pipeline stages** (gathered only to size the risk in the Engineering Risks section, not as a specification): third-party benchmarks of vector search over a local index (for example pgvector, FAISS, or Qdrant) commonly report P50/P99 latency in the single-digit to low-tens-of-milliseconds range, at moderate dataset sizes. Third-party benchmarks of hosted large language model (LLM) APIs commonly report a time-to-first-token of 200 milliseconds or more on standard inference paths, with specialized low-latency inference hardware reporting lower figures. These numbers vary with dataset size, network path, and vendor choice. `/spec` must re-measure them against the actual chosen stack; this PRD does not treat them as a guarantee.

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| One PRD, not split into several | One PRD | The six technical requirements describe one connected pipeline for one submission. They are not independent subsystems. |
| Research tier used | Standard (targeted) | The Engineering Risks section needed real latency figures for speech-to-text, vector search, and LLM generation. Inventing numbers would violate the no-invented-values rule. |
| Video recording and social media posting excluded from the build scope | Excluded from `/spec` scope, tracked separately | `/spec` produces code. Recording a video and posting to Instagram, X, and LinkedIn is not a coding task. |
| MVP requires at least two chunking strategies | Minimum of two | The source document explicitly forbids a single naive fixed-size chunking approach. A higher count is an enhancement, not a floor. |
| Latency target scope | Full voice-to-answer pipeline — speech input through generated answer (the broad reading, formerly Open Decision 1) | Team's explicit choice, made after the infeasibility risk was disclosed (see Engineering Risk 1). |
| Latency percentile gate | All three — P50, P70, and P100 must each individually clear 200 milliseconds (formerly Open Decision 2) | Team's explicit choice; the strictest of the three options offered. |
| Speech-to-text vendor | ElevenLabs Scribe v2 Realtime (formerly Open Decision 4) | Team's explicit choice; its published 150ms figure gives the best available chance of meeting the latency target above. |
| MSMARCO-XI language | English (formerly Open Decision 3) | Team's explicit choice; strongest off-the-shelf model and vendor support, lowest technical risk. |

## Research Findings

Research tier: Standard (targeted web search and two page fetches, aimed at the latency risk analysis the source document asks for in item 7).

- **MSMARCO-XI dataset structure** — confirmed directly from the Hugging Face dataset card (`huggingface.co/datasets/ai4bharat/MSMARCO-XI`). Details are recorded in the Technical Context section above.
- **ElevenLabs Scribe v2 Realtime** — the vendor's own marketing and documentation pages state a 150 millisecond latency figure for real-time transcription, across 90-plus languages.
- **Sarvam AI speech-to-text** — the vendor's documentation (`docs.sarvam.ai`) describes REST, Batch, and Realtime API modes, tuned for Indian languages, but does not publish a specific latency number in the pages this PRD reviewed.
- **Vector database latency benchmarks** — multiple independent 2026 benchmark write-ups (covering pgvector, Qdrant, Milvus, and FAISS) report P50 through P99 query latency in the single-digit to low-tens-of-milliseconds range for local, in-process vector search at moderate corpus sizes. Latency rises with corpus size and with network calls to a remote, hosted vector database.
- **Hosted LLM latency benchmarks** — independent 2026 benchmark trackers report common hosted LLM time-to-first-token figures at 200 milliseconds or higher (one tracker recorded GPT-4o at 342 milliseconds P50 in its measurement window). Specialized inference hardware, such as Groq's LPU, is reported to reach substantially lower time-to-first-token and much higher output-token throughput than standard hosted inference.
- **Gap found, not filled:** no source found during this research pass states a total or per-language record count for MSMARCO-XI. `/spec` should confirm this directly against the dataset.

These findings ground the Engineering Risks section below. They are not implementation instructions.

## Open Decisions

The source document originally left eight items unresolved. The team has since resolved four of them directly (latency target scope, latency percentile gate, speech-to-text vendor, and MSMARCO-XI language); those four now appear in the Key Decisions table above, not here. The four items below are still unresolved. `/spec` must decide each one before or during implementation planning, because each choice changes the resulting architecture.

1. **Is a spoken answer required, or is an on-screen text answer enough?** The source document's own pipeline diagram ends at "Answer generation." Its opening sentence says the system "returns an answer," without stating the output medium. The title "voice-enabled" could describe voice input alone, or a full voice-in, voice-out experience.

2. **How many chunking strategies, and which ones, count as satisfying the "vast" chunking requirement?** The source document requires more than one strategy and lists examples — semantic splitting, fixed-size splitting, overlap handling, metadata-aware splitting — but sets no required count. This PRD sets a floor of two for the MVP (see MVP section). How far beyond two to go is a judgment call bounded by the deadline.

3. **What must the hallucination or groundedness check actually test, and what should the system do when it fails?** The source document asks for "hallucination checks, or answers not grounded in the retrieved context," and says the system should "know when not to answer." It does not name a detection method, and it does not say what the refusal response should look like — for example, a plain "I don't know," a clarifying question back to the user, or a caveated partial answer.

4. **Where will the live link run, and must the 200-millisecond target hold over real network conditions to that host, or only in a local, controlled benchmark?** The source document does not name a hosting target or say whether judges will test latency live.

## Engineering Risks

1. **The chosen 200-millisecond target — full voice-to-answer pipeline, gated on P50, P70, and P100 alike — is very likely unreachable with the chosen hosted vendor stack.** ElevenLabs' own published figure for its fastest real-time mode is 150 milliseconds, for speech-to-text alone. Hosted LLM generation commonly adds 200 milliseconds or more just for the first token, before any output tokens are produced, and vector retrieval adds further time on top. Added together, the three stages are very unlikely to land under 200 milliseconds even once, let alone at every percentile including the worst case (P100) across a full test batch. The source document phrases this requirement as "should complete" (item 3), not as a submission gate. The team should treat clearing all three percentiles as a stretch goal, keep measuring and reporting the actual achieved numbers regardless of outcome (this satisfies item 4, the separate latency-reporting requirement, independently of whether item 3's target is met), and be ready to state plainly in the submission whether the target was hit and by how much it was missed if not. Chasing this target past the point of diminishing return, at the cost of the harness or guardrail requirements, would trade a hard requirement for a soft one.

2. **Streaming speech-to-text and blocking speech-to-text produce very different latency numbers.** A blocking (record-then-send) call waits for the full spoken utterance before returning any text, which adds the length of the recording itself to the measured latency. A streaming call returns partial text while the user is still speaking, which can hide most of the speech-to-text latency behind the user's own speaking time — but only if the rest of the harness is built to consume a stream, not a single final response. This is a deliberate design choice, not a default, and it has a large effect on the reported numbers.

3. **The first one or two test queries will likely be far slower than the rest, which skews P100.** Loading a model, warming up an embedding index, and opening API connections all typically make the first query in a batch slower than later queries. The source document asks for P100 across "a reasonable number of test queries." Since the team's chosen latency gate (Key Decisions table) requires P100 itself to clear 200 milliseconds, an un-separated cold start makes that specific criterion close to impossible to pass on a literal reading — the harness should separate a warm-up phase from the measured batch, and the team should decide, and record, whether "P100" means the worst case including cold start or the worst steady-state case.

4. **A model-based guardrail check adds latency on the same critical path the 200-millisecond target measures.** A hallucination or groundedness check that itself calls an LLM (for example, "does this answer follow from this context, yes or no") adds another network round trip before the system can return an answer. Guardrail design and latency design must be solved together — a guardrail added on at the end, after a latency budget is already tight, will likely break that budget.

5. **Chunking strategy choice affects retrieval latency and retrieval accuracy at the same time, in different directions.** Smaller chunks tend to retrieve faster and more precisely, but can lose surrounding context needed for a full answer. Semantic chunking is typically more expensive to compute when the dataset is first ingested, which is a one-time cost, but it does not have to be more expensive at query time. The "vast" chunking requirement (source document item 2) must be judged on query-time latency, not only on retrieval quality.

6. **This risk is mitigated by the team's language choice.** The team chose English (Key Decisions table), which carries the strongest off-the-shelf embedding, tokenizer, and speech-to-text support of any MSMARCO-XI language, and the best realistic chance of hitting a tight latency floor. This risk would resurface if the language choice is later revisited toward a less common Indic language.

7. **The task's one-shot submission rule makes the short build window the primary risk, on top of every technical risk above.** The task launched August 13, 2026, and the deadline is August 22, 2026, 11:59 PM, with no resubmission allowed. A large, ambitious scope that is unfinished at the deadline is a worse outcome than a smaller scope that is finished, working, and demonstrable on video.

## Acceptance Criteria

| # | Requirement (source document item) | Acceptance Criterion |
|---|---|---|
| 1 | Speech-to-text (item 1) | A spoken English-language audio question, through ElevenLabs Scribe v2 Realtime, produces a text transcript. |
| 2 | Chunking (item 2) | At least two chunking strategies exist, are documented, and differ in method — not only in parameter values (for example, fixed-size plus semantic, or fixed-size plus metadata-aware). Both strategies produce a retrievable index. |
| 3 | Vector indexing and retrieval (item 2) | Given a text query, the system returns a ranked set of passages from the vector index. Retrieval quality is checked, for at least one chunking strategy, against the dataset's own `is_selected` relevance labels (for example, a recall-at-k measurement). |
| 4 | Latency target (item 3) | The full pipeline — spoken audio input through generated answer, including speech-to-text — completes in under 200 milliseconds, and this holds at P50, P70, and P100 across the test batch. Per Engineering Risk 1, this is a stretch target the team may not fully clear; the team must report the actual P50/P70/P100 numbers regardless (criterion 5) and state plainly whether the 200ms target was met. |
| 5 | Latency analytics (item 4) | P50, P70, and P100 latency are computed and reported from a run over a stated number of test queries. The source document says "a reasonable number," not an exact count; `/spec` must pick and state a concrete number (for example, 30 or more queries). |
| 6 | Harness (item 5) | Every model or API call in the pipeline (speech-to-text, embedding, generation) goes through a call path with: a defined input and output shape, at least one retry on a transient failure, and a defined error path for when retries run out — for example, a clear message to the user, not a crash. |
| 7 | Guardrails (item 6) | The system has a demonstrable "cannot answer" path for each of: (a) a question unrelated to the ingested dataset, (b) a deliberately unsafe or inappropriate input, and (c) a case where the retrieved passages do not support the drafted answer. Each path can be triggered on demand, for the demo video. |
| 8 | Submission (submission requirements) | The GitHub repository is reachable. The live link serves a working system. Both videos meet their stated length and content rules (90 seconds for the process video; full end-to-end coverage for the demo video). |

## MVP vs. Optional Enhancement

### MVP — required to submit

- ElevenLabs Scribe v2 Realtime wired end to end for speech-to-text.
- MSMARCO-XI English-language content ingested.
- At least two chunking strategies; at least one active vector index built from them.
- Vector retrieval working against that index.
- Answer generation, grounded in the retrieved passages.
- A minimal harness: a defined call schema, at least one retry, and one defined error path.
- Minimal guardrails: an off-topic check and an unsafe-input check, each ending in a "cannot answer" response.
- Latency instrumented per stage — including speech-to-text — with P50/P70/P100 computed and reported over a stated test batch.
- A live, deployed, working link.
- A GitHub repository.

### Optional enhancement — valuable, not required to submit

- More than two chunking strategies, compared side by side in the report or demo.
- A model-based hallucination or groundedness check (for example, a second LLM call acting as a judge), beyond a simpler rule-based groundedness check.
- Support for more than one MSMARCO-XI language.
- A spoken (text-to-speech) answer, if Open Decision 1 resolves toward "text is enough" for the MVP.
- Retry policies with backoff or circuit-breaking, beyond a single retry.
- A latency dashboard or observability view, beyond a static report.
- Automatic warm-up handling, to stabilize the P100 figure (see Engineering Risk 3).

## Non-Engineering Requirements (tracked here, outside `/spec`'s scope)

The source document sets these requirements. They are real requirements of the task. They are not software to build, so they are tracked here instead of in the Scope section above.

- Fill the official submission form (`forms.gle/MNvCjcv23Hn2Eeu58`).
- Record Video 1: a 90-second team/process video, showing how the team works, not the product.
- Record Video 2: a full end-to-end demo video of the working system.
- Post both videos to Instagram, X, and LinkedIn — by every individual team member, not one shared team post.
- At least one team member's Instagram account must be public.
- Every post, on every platform, by every member, must include the tag `#RAGInGoa`.
- Submit before August 22, 2026, 11:59 PM. No resubmission is allowed after submitting.
