# Latency Report

**Measured 2026-08-19 against the live deployment** at `https://ragingoa.duckdns.org`
(AWS EC2 `m7i-flex.large`, 2 vCPU / 7.6 GB, `ap-south-1b` Mumbai), over HTTPS through
the production Caddy → uvicorn path. Not a local run.

**Method.** 33 real spoken-audio requests through the live `/api/ask` endpoint — the full
pipeline, including speech-to-text, on every call. **The first 3 are warm-up and are
excluded**; percentiles are over the remaining **30 measured requests**. Audio is 16 kHz
mono PCM of five representative questions (`assets/benchmark_audio/`), cycled to reach the
batch size. Reproduce with `python -m scripts.benchmark_latency --base-url https://ragingoa.duckdns.org`.

## Headline: P50 / P70 / P100

The task specifies the target as *"chunking + vector DB retrieval + everything through to
final output ... under 200 ms."* We report three boundaries rather than one number, because
the pipeline contains two third-party network calls whose latency we do not control, and a
single figure would hide where the time actually goes.

| Boundary | What it covers | P50 | P70 | P100 | Under 200 ms |
|---|---|---|---|---|---|
| **A** | Retrieval + all three guardrails | **129.3 ms** | **133.0 ms** | **147.3 ms** | **Yes — all three** |
| **B** | A + answer generation (Claude) | 2,239 ms | 2,412 ms | 3,658 ms | No |
| **C** | Full end-to-end, incl. speech-to-text | 3,356 ms | 3,547 ms | 4,682 ms | No |

**Boundary A — the retrieval pipeline this task is about — meets the 200 ms target at P50,
P70 and P100**, with ~26% headroom even at the worst case.

Boundaries B and C do not, and cannot. The reasons are external and quantified below.

## Per-stage breakdown (30 requests)

| Stage | P50 | P70 | P100 | In our control |
|---|---|---|---|---|
| Speech-to-text (ElevenLabs) | 1,117 ms | 1,212 ms | 1,560 ms | No — third-party |
| Unsafe-input guardrail | 0.0 ms | 0.0 ms | 0.0 ms | Yes |
| **Retrieval (embed + FAISS)** | **19.6 ms** | **20.3 ms** | **24.2 ms** | Yes |
| Off-topic guardrail | 0.0 ms | 0.0 ms | 0.0 ms | Yes |
| Answer generation (Claude Haiku 4.5) | 2,108 ms | 2,280 ms | 3,520 ms | No — third-party |
| Groundedness guardrail | 110.7 ms | 114.5 ms | 125.3 ms | Yes |

Retrieval over **439,675 indexed chunks** takes **19.6 ms at P50** — an order of magnitude
inside the target. The unsafe-input and off-topic guards are effectively free: one is regex,
the other reuses the retrieval score already computed.

The groundedness guard (110.7 ms) is the largest component we own. It embeds the generated
answer and the concatenated retrieved context — two model calls, the second over a long
string. It is the obvious target if boundary A ever needs to be faster.

## Why B and C cannot reach 200 ms

Measured from the instance itself, independent of this pipeline:

- **Generation floor.** Claude Haiku 4.5 returns its first token in ~650 ms and a complete
  answer in ~2.1 s. `api.anthropic.com` answers in 25 ms TTFB from Mumbai, so this is model
  time, not network. No hosted LLM completes a grounded multi-sentence answer in 200 ms.
- **Speech-to-text floor.** ~1.1 s per clip. `api.elevenlabs.io` has a ~350 ms origin
  round-trip from Mumbai; the rest is the WebSocket handshake, audio upload and commit.

Both are provider-bound. Every locally controllable stage in this pipeline sums to ~130 ms.

## Known headroom, not claimed as achieved

- Generation is **non-streaming**, so boundary B measures time-to-*complete*-answer. Measured
  time-to-first-token is ~650 ms; streaming would cut perceived latency roughly threefold.
- Answers are verbose (multi-paragraph, bulleted). Instructing brevity would reduce
  generation time materially.
- Speech-to-text opens a fresh WebSocket per request. A socket opened at record-press
  would remove the handshake from the measured window.

## Reproducing

```bash
# full-pipeline percentiles
python -m scripts.benchmark_latency --base-url https://ragingoa.duckdns.org

# per-stage percentiles (every response carries stages_ms)
curl -X POST "https://ragingoa.duckdns.org/api/ask?strategy=semantic" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @assets/benchmark_audio/question_0.pcm | jq .stages_ms
```
