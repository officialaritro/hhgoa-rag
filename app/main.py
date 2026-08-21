"""FastAPI app entry point -- wires speech-to-text, guardrails, retrieval,
reranking and generation into one endpoint.

Run: uvicorn app.main:app --host 127.0.0.1 --port 8000

Binds loopback in production: Caddy terminates TLS on 443 and proxies here.
Binding 0.0.0.0 would expose the app unencrypted and break the microphone,
since getUserMedia is a secure-context-only API.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.generation import generate_answer
from app.guardrails import check_groundedness, check_off_topic, check_unsafe_input
from app.retrieval import retrieve
from app.strategies import get, served_names
from app.stt import transcribe

logger = logging.getLogger("uvicorn.error")

# Derived from the registry rather than restated. Strategy identity used to live
# in four places; a strategy present in one and missing from another is exactly
# how a threshold calibrated against one index shipped as another's default.
_STRATEGIES = served_names()

# whole_passage, on the evidence now available (docs/CHUNKING_REPORT.md).
# Nothing in the slate measurably beats it -- every paired 95% interval against
# it contains zero or is negative -- and it is the cheapest of the strategies
# that tie it: 38.3 MB against fixed_size's 38.8, 6.7ms search, and the fewest
# off-topic leaks measured over 50 probes (6, tied with fixed_size, against 27
# for semantic and 32 for query_aware).
_DEFAULT_STRATEGY = "whole_passage"

# /api/compare fans out one transcript across several strategies concurrently,
# and every strategy in the fan-out costs its own generation call, so comparing
# all eight would issue eight Claude calls per request.
#
# The default is a granularity ladder rather than one-per-axis: 1.00, 1.31, 2.20
# and 3.51 chunks per passage. That makes the report's central finding visible in
# the demo itself -- coarser is better on this corpus, and the finest split is
# measurably the worst of the four. query_aware was in this list and is removed:
# it is the weakest strategy measured (significantly worse recall, 32 of 50
# off-topic probes leaked), so showcasing it would misrepresent the system.
COMPARE_DEFAULT = ("whole_passage", "recursive", "semantic", "sentence_window")

# Only the default is warmed at startup. The rest load on first use via the
# cache in app/retrieval.py -- warming all eight would read 483 MB of indices
# and unpickle every span store before the service reports ready.
_WARM_AT_STARTUP = (_DEFAULT_STRATEGY,)

# Set True once both indices and the embedding model load and answer a
# warm-up query successfully. False means /health reports not-ready instead
# of the app crashing on boot -- data/ may not be populated yet in every
# environment this runs in (e.g. before Task 10's first deploy).
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _ready
    try:
        for strategy in _WARM_AT_STARTUP:
            retrieve(query="warmup", strategy=strategy, k=1)
        _ready = True
        logger.info("startup: indices loaded, service ready")
    except Exception:
        _ready = False
        # Log the traceback. Swallowing it silently left /health returning 503
        # with no way to tell whether the cause was a missing index, an
        # unreadable model cache, or a model mismatch -- all of which happened.
        logger.exception("startup: warm-up failed, service will report not-ready")
    yield


app = FastAPI(title="Voice-Enabled RAG Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> JSONResponse:
    if _ready:
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=503, content={"status": "not ready"})


def _refusal(
    reason: str | None,
    latency_ms: float,
    stages: dict[str, float],
    strategy: str,
    transcript: str | None = None,
    top_score: float | None = None,
) -> dict[str, Any]:
    # Transcript is echoed back even on refusal: TS-002/TS-003 both expect the
    # user to see what was heard before being told it cannot be answered.
    #
    # top_score is reported on refusals too, not only on the success path. An
    # off-topic refusal that shows no score is undiagnosable from the browser:
    # a miscalibrated threshold and a genuinely off-topic question produce the
    # identical response. That is exactly how a threshold refusing 38.5% of
    # real in-corpus questions reached the live service unnoticed.
    #
    # passages is always [] here: _refusal() is only used for the stt/unsafe
    # refusals that happen before retrieve() ever runs, so there is nothing to
    # cite yet.
    return {
        "transcript": transcript,
        "answer": None,
        "refusal_reason": reason,
        "latency_ms": latency_ms,
        "stages_ms": stages,
        "strategy": strategy,
        "top_score": top_score,
        "passages": [],
    }


async def _run_strategy_pipeline(transcript: str, strategy: str) -> dict[str, Any]:
    """Retrieval -> off-topic guard -> generation -> groundedness guard for one
    strategy, given an already-transcribed, already-safety-checked query.

    Returns the per-strategy result shape shared by /api/ask (merged with its
    own stt/guardrail_unsafe stages) and /api/compare (merged per strategy
    into a `results` object) -- both endpoints need the identical shape, so
    it lives here once rather than being built twice.
    """
    stages: dict[str, float] = {}

    mark = time.perf_counter()
    retrieval = await run_in_threadpool(retrieve, query=transcript, strategy=strategy)
    stages["retrieval"] = _elapsed_ms(mark)
    passages = [p.model_dump() for p in retrieval.passages]

    mark = time.perf_counter()
    # max, not passages[0]: the reranker reorders by cross-encoder relevance, so
    # the first passage is no longer necessarily the highest cosine. The guard
    # asks "is anything in the corpus close enough to this question", which is a
    # max over the candidates -- reading position zero would let a reranker
    # promotion cause a false refusal.
    top_score = max((p.score for p in retrieval.passages), default=0.0)
    off_topic = check_off_topic(top_similarity_score=top_score, strategy=strategy)
    stages["guardrail_off_topic"] = _elapsed_ms(mark)
    if not off_topic.passed:
        return {
            "answer": None,
            "refusal_reason": off_topic.reason,
            "stages_ms": stages,
            "top_score": top_score,
            "passages": passages,
        }

    mark = time.perf_counter()
    generation_result = await run_in_threadpool(
        generate_answer, query=transcript, retrieval=retrieval
    )
    stages["generation"] = _elapsed_ms(mark)
    if not generation_result.ok:
        return {
            "answer": None,
            "refusal_reason": "could not generate an answer",
            "stages_ms": stages,
            "top_score": top_score,
            "passages": passages,
        }

    answer = generation_result.value.answer

    # A decline is a refusal, not an answer. It cannot be left for the
    # groundedness guard below: the model quotes the passages when explaining
    # why it cannot answer, so a decline scores as well against the context as
    # a real answer (measured 0.533 fixed_size / 0.795 semantic vs a 0.40
    # threshold) and would be rendered to the user as an answer with no
    # refusal reason at all.
    if generation_result.value.insufficient_context:
        return {
            "answer": None,
            "refusal_reason": "not in the retrieved passages",
            "stages_ms": stages,
            "top_score": top_score,
            "passages": passages,
        }

    mark = time.perf_counter()
    grounded = await run_in_threadpool(
        check_groundedness, answer=answer, retrieval=retrieval
    )
    stages["guardrail_groundedness"] = _elapsed_ms(mark)
    if not grounded.passed:
        return {
            "answer": None,
            "refusal_reason": grounded.reason,
            "stages_ms": stages,
            "top_score": top_score,
            "passages": passages,
        }

    return {
        "answer": answer,
        "refusal_reason": None,
        "stages_ms": stages,
        "top_score": top_score,
        "passages": passages,
    }


@app.post("/api/ask")
async def ask(
    request: Request,
    strategy: str = Query(
        _DEFAULT_STRATEGY,
        description="Which chunking strategy's index to retrieve from.",
    ),
) -> JSONResponse:
    """Voice question in (raw PCM body) -> grounded answer or a stated refusal.

    Every blocking call below runs via `run_in_threadpool`. This endpoint is
    `async`, so a bare blocking call would occupy the event loop and stall
    *every* concurrent request, not just this one -- with a ~1s pipeline and
    several judges clicking at once, that serialises the whole demo.
    """
    start = time.perf_counter()
    stages: dict[str, float] = {}

    if strategy not in _STRATEGIES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"unknown strategy {strategy!r}",
                "valid_strategies": list(_STRATEGIES),
            },
        )

    audio_bytes = await request.body()

    mark = time.perf_counter()
    stt_result = await run_in_threadpool(transcribe, audio_chunks=[audio_bytes])
    stages["stt"] = _elapsed_ms(mark)
    if not stt_result.ok:
        return JSONResponse(
            _refusal("could not process audio", _elapsed_ms(start), stages, strategy)
        )

    transcript = stt_result.value.transcript
    if not transcript.strip():
        return JSONResponse(
            _refusal("could not understand audio", _elapsed_ms(start), stages, strategy)
        )

    # Cheapest check first: pure regex, no model call, so it costs nothing to
    # run before retrieval and rejects unsafe input without touching the index.
    mark = time.perf_counter()
    unsafe = check_unsafe_input(transcript)
    stages["guardrail_unsafe"] = _elapsed_ms(mark)
    if not unsafe.passed:
        return JSONResponse(
            _refusal(unsafe.reason, _elapsed_ms(start), stages, strategy, transcript)
        )

    result = await _run_strategy_pipeline(transcript, strategy)
    stages.update(result["stages_ms"])

    return JSONResponse(
        {
            "transcript": transcript,
            "answer": result["answer"],
            "refusal_reason": result["refusal_reason"],
            "latency_ms": _elapsed_ms(start),
            "stages_ms": stages,
            "strategy": strategy,
            "top_score": result["top_score"],
            "passages": result["passages"],
        }
    )


async def _run_strategy_pipeline_safe(transcript: str, strategy: str) -> dict[str, Any]:
    """Wraps `_run_strategy_pipeline` so one strategy's unexpected failure
    inside the `/api/compare` fan-out reports a refusal for that strategy
    only, instead of raising out of `asyncio.gather` and 500ing the whole
    request -- which would also discard the sibling strategy's real result.
    """
    try:
        return await _run_strategy_pipeline(transcript, strategy)
    except Exception:
        logger.exception("compare: strategy %r raised", strategy)
        return {
            "answer": None,
            "refusal_reason": "internal error",
            "stages_ms": {},
            "top_score": None,
            "passages": [],
        }


@app.post("/api/compare")
async def compare(
    request: Request,
    strategies_param: str = Query(
        ",".join(COMPARE_DEFAULT),
        alias="strategies",
        description="Comma-separated strategies to compare side by side.",
    ),
) -> JSONResponse:
    """Voice question in -> both chunking strategies answered from the same
    transcription, concurrently. Exists so a side-by-side comparison never
    has to transcribe the same audio twice (extra STT latency/cost, and a
    real risk of two different transcripts for what's supposed to be one
    question)."""
    start = time.perf_counter()
    shared_stages: dict[str, float] = {}

    compared = tuple(s.strip() for s in strategies_param.split(",") if s.strip())
    unknown = [s for s in compared if s not in _STRATEGIES]
    if not compared or unknown:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"unknown strategies {unknown}"
                if unknown
                else "no strategies",
                "valid_strategies": list(_STRATEGIES),
            },
        )

    audio_bytes = await request.body()

    mark = time.perf_counter()
    stt_result = await run_in_threadpool(transcribe, audio_chunks=[audio_bytes])
    shared_stages["stt"] = _elapsed_ms(mark)
    if not stt_result.ok:
        return JSONResponse(
            {
                "transcript": None,
                "latency_ms": _elapsed_ms(start),
                "shared_stages_ms": shared_stages,
                "refusal_reason": "could not process audio",
                "results": None,
            }
        )

    transcript = stt_result.value.transcript
    if not transcript.strip():
        return JSONResponse(
            {
                "transcript": transcript,
                "latency_ms": _elapsed_ms(start),
                "shared_stages_ms": shared_stages,
                "refusal_reason": "could not understand audio",
                "results": None,
            }
        )

    mark = time.perf_counter()
    unsafe = check_unsafe_input(transcript)
    shared_stages["guardrail_unsafe"] = _elapsed_ms(mark)
    if not unsafe.passed:
        return JSONResponse(
            {
                "transcript": transcript,
                "latency_ms": _elapsed_ms(start),
                "shared_stages_ms": shared_stages,
                "refusal_reason": unsafe.reason,
                "results": None,
            }
        )

    strategy_results = await asyncio.gather(
        *(_run_strategy_pipeline_safe(transcript, s) for s in compared)
    )

    return JSONResponse(
        {
            "transcript": transcript,
            "latency_ms": _elapsed_ms(start),
            "shared_stages_ms": shared_stages,
            "refusal_reason": None,
            "results": dict(zip(compared, strategy_results)),
        }
    )


@app.get("/api/strategies")
def strategies() -> JSONResponse:
    """Chunking strategies available to retrieve from, with what each one does.

    The frontend reads this rather than hard-coding names. Eight bare strategy
    names mean nothing in a radio group, and the registry already carries a
    description and the axis each strategy varies, so they are served here
    rather than restated in the UI.
    """
    return JSONResponse(
        {
            "strategies": list(_STRATEGIES),
            "default": _DEFAULT_STRATEGY,
            "compare_default": list(COMPARE_DEFAULT),
            "details": {
                name: {
                    "description": get(name).description,
                    "axis": get(name).axis,
                    "kind": get(name).kind,
                }
                for name in _STRATEGIES
            },
        }
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
