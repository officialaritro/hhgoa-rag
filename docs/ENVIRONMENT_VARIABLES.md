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
| `OFFTOPIC_SIMILARITY_THRESHOLD_FIXED_SIZE` | `0.55` | Task 7 | Measured, not guessed: sits at the fixed-size index's in-corpus p05, costing 4.5% false refusals. Leave unset — the default in `app/guardrails.py` is pinned to a false-refusal budget by `tests/test_guardrail_calibration.py`, and setting this bypasses that check. |
| `OFFTOPIC_SIMILARITY_THRESHOLD_SEMANTIC` | `0.60` | Task 7 | Separate from the fixed-size value because the indices are on different score scales — semantic chunks are shorter, so every cosine runs higher (in-corpus median 0.779 vs 0.741). A single shared threshold verified on one index refused 38.5% of real questions on the other. |
| `GROUNDEDNESS_SIMILARITY_THRESHOLD` | `0.40` | Task 7 | Measured against real generated answers (0.756–0.902 vs their context, below 0.11 against unrelated context). Note `scripts/tune_thresholds.py` recommends ~0.02 here because it scores the dataset's terse `Eng_Answer` rather than model output. |
| `LATENCY_WARMUP_QUERIES` | `3` | Task 9 | Queries run and discarded before the measured latency batch, per the plan's cold-start mitigation. |
| `LATENCY_BATCH_SIZE` | `30` | Task 9 | Minimum test-query count for the P50/P70/P100 report (PRD Acceptance Criteria 5). |

## What does *not* need an environment variable

- **AWS credentials are not part of the application's `.env`.** Nothing in this pipeline calls an AWS API at runtime (FAISS is local, no S3 or other AWS service is used) — AWS credentials are only needed on the machine doing the *deployment*, not on the running server. See [`deploy/AWS_SETUP.md`](../deploy/AWS_SETUP.md).
- Dataset language/content is fixed (English, MSMARCO-XI) per the PRD and isn't configurable via env.
