# Environment Variables

Template: [`.env.example`](../.env.example) at the repo root. Copy it to `.env` and fill in real values. `.env` is git-ignored; `.env.example` is committed and holds no secrets.

## Required secrets

| Variable | Used by | Where to get it | Notes |
|---|---|---|---|
| `ELEVENLABS_API_KEY` | Task 5 — speech-to-text | [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys) | Confirmed from ElevenLabs' own authentication docs. Sent as the `xi-api-key` HTTP header by the SDK — never expose it client-side. |
| `ANTHROPIC_API_KEY` | Task 6 — answer generation | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | Confirmed from Anthropic's SDK authentication reference. |

Neither key is needed to run the unit test suite — Tasks 5 and 6 mock both clients in tests, per the plan.

## Optional runtime config

These have working defaults; override only if you need to.

| Variable | Default | Used by | Notes |
|---|---|---|---|
| `HOST` | `0.0.0.0` | Task 8 / Runtime Environment | |
| `PORT` | `8000` | Task 8 / Runtime Environment | Matches the plan's health-check and start-command port. |
| `CLAUDE_MODEL_ID` | `claude-haiku-4-5` | Task 6 | Recommended starting point for the low-latency generation call — the fastest/smallest current Claude tier. Task 6's plan note says to reconfirm the exact model ID against current docs at implementation time, not assume this stays correct. Swap to `claude-sonnet-5` if answer quality needs it, at the cost of latency. |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Task 3 | Local, in-process embedding model, per the plan's Global Constraints (no hosted embeddings API on the critical path). Confirm this model's license and quality are acceptable before finalizing. |
| `DATA_DIR` | `./data` | Tasks 2, 3, 9 | Where the ingested corpus, the shared passage store and every strategy's FAISS index are read/written. Already excluded from git via `.gitignore` (`data/`). |
| `OFFTOPIC_SIMILARITY_THRESHOLD_<STRATEGY>` | *from the index manifest* | guardrails | Per-strategy override, e.g. `OFFTOPIC_SIMILARITY_THRESHOLD_WHOLE_PASSAGE`. Normally unset: thresholds are **measured per index at build time** by `scripts/calibrate_thresholds.py` and written into that index's `.manifest.json`. `app/guardrails.py` raises `MissingCalibration` rather than borrow another index's number, so an uncalibrated strategy cannot be served. Setting this bypasses the budget check in `tests/test_guardrail_calibration.py`. |
| `GROUNDEDNESS_SIMILARITY_THRESHOLD` | `0.40` | guardrails | Threshold on the **mean per-answer-sentence** support, not a whole-answer cosine. Measured over 40 real generated answers: grounded minimum 0.524, ungrounded maximum 0.271 — a clean gap whose midpoint is 0.397. Re-measure with `scripts/calibrate_groundedness.py`. |
| `RERANK_ENABLED` | `1` | reranking | Cross-encoder reranking. Worth +6.8pp recall@5 (0.848 → 0.916, CI +0.038 to +0.098) for ~53ms of boundary A. Set to `0` if the host cannot afford it — on 2 vCPU it pushed boundary A past the 200ms target. |
| `RERANK_DEPTH` | `7` | reranking | Candidates reranked. The depth that fits is a property of the hardware: 7 fits 2 vCPU (170.8ms boundary A), 10 fits 8 vCPU (84.7ms) and is worth another 1.6pp. Depth 5 cannot help recall@5 — reranking the top five cannot change which five are in the top five. |
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranking | |
| `EMBEDDING_DEVICE` | *auto* | embeddings | Torch device. Unset on the instance, which has no GPU. Set to `mps` for index builds on Apple Silicon: 506 texts/sec against 326 on CPU. |
| `LATENCY_WARMUP_QUERIES` | `3` | Task 9 | Queries run and discarded before the measured latency batch, per the plan's cold-start mitigation. |
| `LATENCY_BATCH_SIZE` | `30` | Task 9 | Minimum test-query count for the P50/P70/P100 report (PRD Acceptance Criteria 5). |

## What does *not* need an environment variable

- **AWS credentials are not part of the application's `.env`.** Nothing in this pipeline calls an AWS API at runtime (FAISS is local, no S3 or other AWS service is used) — AWS credentials are only needed on the machine doing the *deployment*, not on the running server. See [`deploy/AWS_SETUP.md`](../deploy/AWS_SETUP.md).
- Dataset language/content is fixed (English, MSMARCO-XI) per the PRD and isn't configurable via env.
