"""ElevenLabs Scribe v2 Realtime streaming speech-to-text, wrapped in the harness.

Streaming (not blocking REST) per the plan's Global Constraints -- a blocking
call would add the full recording length to measured latency (Engineering Risk 2).

SDK surface verified against the installed `elevenlabs` package (v2.64.0):
elevenlabs.realtime.scribe.ScribeRealtime / AudioFormat / CommitStrategy and
elevenlabs.realtime.connection.RealtimeEvents.
"""

import asyncio
import base64
import os
from typing import Any

from app.harness import StageResult, run_stage
from app.schemas import STTOutput

_MODEL_ID = "scribe_v2_realtime"
_SAMPLE_RATE = 16000
_COMMIT_TIMEOUT_SECONDS = 10


def extract_transcript(data: dict[str, Any] | None) -> str:
    """Pull the transcript out of a committed_transcript event payload.

    The key is `text`, NOT `transcript`. Observed payload, captured live from
    scribe_v2_realtime:

        {'message_type': 'committed_transcript',
         'text': 'What was the immediate impact of the Manhattan Project?'}

    Reading the wrong key returns '' for every query, which the pipeline then
    reports as "could not understand audio" -- a silent, total failure of the
    voice path. Kept as a separate pure function so it is testable without a
    live connection; the tests previously mocked the whole streaming call,
    which put this parsing below the mock boundary where no test could reach it.
    """
    if not data:
        return ""
    return data.get("text") or ""


async def _stream_transcript_async(audio_chunks: list[bytes]) -> str:
    from elevenlabs.realtime.connection import RealtimeEvents
    from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy, ScribeRealtime

    api_key = os.environ["ELEVENLABS_API_KEY"]
    client = ScribeRealtime(api_key=api_key)
    connection = await client.connect(
        {
            "model_id": _MODEL_ID,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": _SAMPLE_RATE,
            "commit_strategy": CommitStrategy.MANUAL,
            "language_code": "eng",
        }
    )

    committed: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    def on_committed(data: dict[str, Any]) -> None:
        if not committed.done():
            committed.set_result(extract_transcript(data))

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)

    try:
        for chunk in audio_chunks:
            await connection.send({"audio_base_64": base64.b64encode(chunk).decode()})
        await connection.commit()
        return await asyncio.wait_for(committed, timeout=_COMMIT_TIMEOUT_SECONDS)
    finally:
        await connection.close()


def _stream_transcript(audio_chunks: list[bytes]) -> str:
    """Blocking bridge to the async SDK. Call from a sync context only --
    e.g. a FastAPI sync `def` route, which Starlette runs in a worker thread,
    never from inside code already running on an asyncio event loop."""
    return asyncio.run(_stream_transcript_async(audio_chunks))


def transcribe(audio_chunks: list[bytes]) -> StageResult:
    result = run_stage(lambda: _stream_transcript(audio_chunks))
    if not result.ok:
        return result
    return StageResult(ok=True, value=STTOutput(transcript=result.value))
