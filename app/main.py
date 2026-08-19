"""FastAPI app entry point -- wires speech-to-text, guardrails, retrieval,
and generation into one endpoint (plan Task 8).

Run: uvicorn app.main:app --host 0.0.0.0 --port 8000
(see docs/plans/2026-08-18-voice-enabled-rag-pipeline.md -> Runtime Environment)
"""

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
from app.stt import transcribe

logger = logging.getLogger("uvicorn.error")

_DEFAULT_STRATEGY = "fixed_size"
_STRATEGIES = ("fixed_size", "semantic")

# Set True once both indices and the embedding model load and answer a
# warm-up query successfully. False means /health reports not-ready instead
# of the app crashing on boot -- data/ may not be populated yet in every
# environment this runs in (e.g. before Task 10's first deploy).
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _ready
    try:
        for strategy in _STRATEGIES:
            retrieve(query="warmup", strategy=strategy, k=1)
        _ready = True
        logger.info("startup: indices loaded, service ready")
    except Exception:  # noqa: BLE001 -- any startup failure means not-ready, not a crash
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
) -> dict[str, Any]:
    # Transcript is echoed back even on refusal: TS-002/TS-003 both expect the
    # user to see what was heard before being told it cannot be answered.
    return {
        "transcript": transcript,
        "answer": None,
        "refusal_reason": reason,
        "latency_ms": latency_ms,
        "stages_ms": stages,
        "strategy": strategy,
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
            _refusal(
                "could not understand audio", _elapsed_ms(start), stages, strategy
            )
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

    mark = time.perf_counter()
    retrieval = await run_in_threadpool(retrieve, query=transcript, strategy=strategy)
    stages["retrieval"] = _elapsed_ms(mark)

    mark = time.perf_counter()
    top_score = retrieval.passages[0].score if retrieval.passages else 0.0
    off_topic = check_off_topic(top_similarity_score=top_score)
    stages["guardrail_off_topic"] = _elapsed_ms(mark)
    if not off_topic.passed:
        return JSONResponse(
            _refusal(off_topic.reason, _elapsed_ms(start), stages, strategy, transcript)
        )

    mark = time.perf_counter()
    generation_result = await run_in_threadpool(
        generate_answer, query=transcript, retrieval=retrieval
    )
    stages["generation"] = _elapsed_ms(mark)
    if not generation_result.ok:
        return JSONResponse(
            _refusal(
                "could not generate an answer",
                _elapsed_ms(start),
                stages,
                strategy,
                transcript,
            )
        )

    answer = generation_result.value.answer
    mark = time.perf_counter()
    grounded = await run_in_threadpool(
        check_groundedness, answer=answer, retrieval=retrieval
    )
    stages["guardrail_groundedness"] = _elapsed_ms(mark)
    if not grounded.passed:
        return JSONResponse(
            _refusal(grounded.reason, _elapsed_ms(start), stages, strategy, transcript)
        )

    return JSONResponse(
        {
            "transcript": transcript,
            "answer": answer,
            "refusal_reason": None,
            "latency_ms": _elapsed_ms(start),
            "stages_ms": stages,
            "strategy": strategy,
            "top_score": top_score,
        }
    )


@app.get("/api/strategies")
def strategies() -> JSONResponse:
    """Chunking strategies available to retrieve from. Both indices are built;
    the frontend uses this to offer a choice rather than hard-coding one."""
    return JSONResponse({"strategies": list(_STRATEGIES), "default": _DEFAULT_STRATEGY})


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
