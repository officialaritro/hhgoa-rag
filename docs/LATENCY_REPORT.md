# Latency Report

**Measured 2026-08-21 against the live deployment** at `https://ragingoa.duckdns.org`
(AWS EC2 `c7i.2xlarge`, 8 vCPU / 15.7 GB, Xeon Platinum 8488C, `ap-south-1b` Mumbai), over
HTTPS through the production Caddy → uvicorn path. Not a local run.

**Method.** 33 real spoken-audio requests through the live `/api/ask` endpoint — the full
pipeline, including speech-to-text, on every call. **The first 3 are warm-up and are
excluded**; percentiles are over the remaining **30 measured requests**. Audio is 16 kHz
mono PCM of five representative questions (`assets/benchmark_audio/`), cycled to reach the
batch size. Reproduce with `python -m scripts.benchmark_latency --base-url https://ragingoa.duckdns.org`.

**Configuration measured:** `whole_passage` chunking, cross-encoder reranking enabled at
depth 10, claim-level groundedness guard. That is what the live service serves.

## Headline: P50 / P70 / P100

The task specifies the target as *"chunking + vector DB retrieval + everything through to
final output ... under 200 ms."* We report three boundaries rather than one number, because
the pipeline contains two third-party network calls whose latency we do not control, and a
single figure would hide where the time actually goes.

| Boundary | What it covers | P50 | P70 | P100 | Under 200 ms |
|---|---|---|---|---|---|
| **A** | Retrieval, reranking, and all three guardrails | **84.7 ms** | **96.3 ms** | **115.7 ms** | **Yes — all three** |
| **B** | A + answer generation (Claude) | 1,328 ms | 1,483 ms | 3,165 ms | No |
| **C** | Full end-to-end, incl. speech-to-text | 2,655 ms | 2,830 ms | 4,369 ms | No |

**Boundary A — the retrieval pipeline this task is about — meets the 200 ms target at P50,
P70 and P100**, with 42% headroom even at the worst case.

Boundaries B and C do not, and cannot. The reasons are external and quantified below.

Boundary A is defined by rule, not by an enumerated list: it sums every stage named
`retrieval` or beginning with `guardrail`. A guardrail added later therefore cannot quietly
fall outside the boundary this claim is made against.

## Per-stage breakdown (30 requests, P50)

| Stage | P50 | In our control |
|---|---|---|
| Speech-to-text (ElevenLabs) | 1,112.1 ms | No — third-party |
| Unsafe-input guardrail | 0.04 ms | Yes |
| **Retrieval + reranking** | **73.4 ms** | Yes |
| Off-topic guardrail | 0.02 ms | Yes |
| Answer generation (Claude Haiku 4.5) | 1,235.2 ms | No — third-party |
| **Groundedness guardrail** | **11.8 ms** | Yes |

## What changed since the previous report

The previous measurement, on `m7i-flex.large` (2 vCPU) without reranking, is worth
comparing against directly.

| Stage P50 | Before | Now | Change |
|---|---|---|---|
| Speech-to-text | 1,117 ms | 1,112 ms | — |
| Retrieval | 19.6 ms | 73.4 ms | now includes cross-encoder reranking |
| Groundedness guard | 110.7 ms | **11.8 ms** | **9.4× faster** |
| Answer generation | 2,108 ms | **1,235 ms** | **41% faster** |
| **Boundary A P50** | 129.3 ms | **84.7 ms** | **34% faster** |
| **Boundary A P100** | 147.3 ms | **115.7 ms** | **21% faster** |

**Boundary A got faster while gaining a cross-encoder reranker**, which is the part worth
explaining. Three separate changes:

**The groundedness guard was re-embedding work retrieval had already done.** It scored the
answer against the five retrieved passages by embedding them on every request. For a chunk
whose embedded span is also its returned span — which `whole_passage`, the default, always
satisfies — that vector is already in the FAISS index. Retrieval now hands it over and the
guard embeds only the answer's sentences. On the old instance that alone took the guard from
111–190 ms to 19–30 ms and stopped boundary A from intermittently breaching 200 ms.

**Answers got shorter.** The generation prompt now caps the answer at three sentences and
forbids prefacing. Measured over 40 real answers: median 79 words to 46, and roughly 32
seconds of synthesised speech down to 18 — which matters more than the millisecond count
for something read aloud. Generation fell 41%, less than the 42% word reduction, because
~650 ms of time-to-first-token is fixed overhead that brevity cannot touch.

**The instance was resized** from `m7i-flex.large` (2 vCPU, burstable) to `c7i.2xlarge`
(8 vCPU, fixed performance). Burstable was the wrong family: a cross-encoder on every
request is sustained CPU, which is precisely what those instances throttle.

## Why reranking is worth 73 ms

It buys **recall@5 of 0.848 → 0.916** (paired 95% CI +0.038 to +0.098), measured over 500
labelled corpus queries. See `docs/CHUNKING_REPORT.md`. For comparison, the best result
from eight chunking strategies and two fusion modes was 0.854, whose interval against no
chunking at all contains zero.

Depth was chosen against this budget rather than for quality alone. On the previous 2 vCPU
instance, depth 10 measured 162 ms in isolation but **124–227 ms in the live service**, and
boundary A breached 200 ms on 3 of 5 requests. Real passages vary from ~100 to 1,233
characters where an isolated probe used one short document repeated, and the reranker
shares CPU with the embedding model and the web server. On 8 vCPU depth 10 fits with room:
retrieval and reranking together are 73.4 ms at P50.

`RERANK_ENABLED=0` disables it and returns boundary A to ~50 ms P50, at the cost of the
recall gain. `RERANK_DEPTH` is also environment-settable, because the depth that fits is a
property of the hardware rather than of the code.

## Why B and C cannot reach 200 ms

Measured from the instance itself, independent of this pipeline:

- **Generation floor.** Claude Haiku 4.5 returns its first token in ~650 ms.
  `api.anthropic.com` answers in 25 ms TTFB from Mumbai, so this is model time, not network.
  No hosted LLM completes a grounded multi-sentence answer in 200 ms.
- **Speech-to-text floor.** ~1.1 s per clip. `api.elevenlabs.io` has a ~350 ms origin
  round-trip from Mumbai; the rest is the WebSocket handshake, audio upload and commit.

Both are provider-bound. Every locally controllable stage sums to ~85 ms.

## Known headroom, not claimed as achieved

- Generation is **non-streaming**, so boundary B measures time-to-*complete*-answer.
  Time-to-first-token is ~650 ms, so streaming would cut perceived latency roughly
  twofold. It is deliberately not implemented: the groundedness guard runs on the complete
  answer, and streaming tokens would show the user text that has not been verified. Doing
  it properly means streaming into a provisional UI state and committing or replacing on
  verification — a design change, not a flag.
- Speech-to-text opens a fresh WebSocket per request. A socket opened at record-press would
  remove the handshake from the measured window.
- Retrieval could be cut further with a two-stage matryoshka shortlist, but at 73 ms inside
  a 200 ms boundary there is no pressure to.

## A note on how these numbers were obtained

Three latency claims in this project's history were wrong because they compared
measurements taken on different hardware, or in isolation rather than in the running
service. Each was corrected only after being measured again in place:

- the groundedness guard was reported as 2.1× faster when it was 1.25× *slower* on matched
  hardware (the earlier figure compared instance CPU against a laptop GPU);
- reranking was sized from an 8-core laptop, then from an isolated probe on the instance,
  before the live service showed it costing 1.9× the probe's estimate.

Every figure in this report is from the live deployment, through HTTPS, with the stage
breakdown the service itself reports.

## Reproducing

```bash
# per-boundary and per-stage percentiles
python -m scripts.benchmark_latency --base-url https://ragingoa.duckdns.org

# a single request's stage breakdown
curl -X POST "https://ragingoa.duckdns.org/api/ask?strategy=whole_passage" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @assets/benchmark_audio/question_0.pcm | jq .stages_ms
```
