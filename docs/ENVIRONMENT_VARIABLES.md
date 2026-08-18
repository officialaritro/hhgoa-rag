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
| `DATA_DIR` | `./data` | Tasks 2, 3, 9 | Where the ingested corpus and the two FAISS indices are read/written. Already excluded from git via `.gitignore` (`data/`). |
| `OFFTOPIC_SIMILARITY_THRESHOLD` | *(none — set during Task 7)* | Task 7 | Determined empirically while building the off-topic guardrail; not a value to guess in advance. |
| `GROUNDEDNESS_SIMILARITY_THRESHOLD` | *(none — set during Task 7)* | Task 7 | Same — determined empirically. |
| `LATENCY_WARMUP_QUERIES` | `3` | Task 9 | Queries run and discarded before the measured latency batch, per the plan's cold-start mitigation. |
| `LATENCY_BATCH_SIZE` | `30` | Task 9 | Minimum test-query count for the P50/P70/P100 report (PRD Acceptance Criteria 5). |

## What does *not* need an environment variable

- **AWS credentials are not part of the application's `.env`.** Nothing in this pipeline calls an AWS API at runtime (FAISS is local, no S3 or other AWS service is used) — AWS credentials are only needed on the machine doing the *deployment*, not on the running server. See [`deploy/AWS_SETUP.md`](../deploy/AWS_SETUP.md).
- Dataset language/content is fixed (English, MSMARCO-XI) per the PRD and isn't configurable via env.
