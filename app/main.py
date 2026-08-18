"""FastAPI app entry point -- wires speech-to-text, guardrails, retrieval,
and generation into one endpoint (plan Task 8).

Run: uvicorn app.main:app --host 0.0.0.0 --port 8000
(see docs/plans/2026-08-18-voice-enabled-rag-pipeline.md -> Runtime Environment)
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.generation import generate_answer
from app.guardrails import check_groundedness, check_off_topic, check_unsafe_input
from app.retrieval import retrieve
from app.stt import transcribe

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
    except Exception:  # noqa: BLE001 -- intentional: any startup failure means not-ready, not a crash
        _ready = False
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


def _refusal(reason: str | None, latency_ms: float) -> dict[str, Any]:
    return {
        "transcript": None,
        "answer": None,
        "refusal_reason": reason,
        "latency_ms": latency_ms,
    }


@app.post("/api/ask")
async def ask(request: Request) -> JSONResponse:
    start = time.perf_counter()
    audio_bytes = await request.body()

    stt_result = transcribe(audio_chunks=[audio_bytes])
    if not stt_result.ok:
        return JSONResponse(_refusal("could not process audio", _elapsed_ms(start)))

    transcript = stt_result.value.transcript
    if not transcript.strip():
        return JSONResponse(_refusal("could not understand audio", _elapsed_ms(start)))

    unsafe = check_unsafe_input(transcript)
    if not unsafe.passed:
        return JSONResponse(_refusal(unsafe.reason, _elapsed_ms(start)))

    retrieval = retrieve(query=transcript, strategy=_DEFAULT_STRATEGY)
    top_score = retrieval.passages[0].score if retrieval.passages else 0.0
    off_topic = check_off_topic(top_similarity_score=top_score)
    if not off_topic.passed:
        return JSONResponse(_refusal(off_topic.reason, _elapsed_ms(start)))

    generation_result = generate_answer(query=transcript, retrieval=retrieval)
    if not generation_result.ok:
        return JSONResponse(
            _refusal("could not generate an answer", _elapsed_ms(start))
        )

    answer = generation_result.value.answer
    grounded = check_groundedness(answer=answer, retrieval=retrieval)
    if not grounded.passed:
        return JSONResponse(_refusal(grounded.reason, _elapsed_ms(start)))

    return JSONResponse(
        {
            "transcript": transcript,
            "answer": answer,
            "refusal_reason": None,
            "latency_ms": _elapsed_ms(start),
        }
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
