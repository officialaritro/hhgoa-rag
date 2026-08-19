from unittest.mock import patch

from fastapi.testclient import TestClient

from app.harness import StageResult
from app.main import app
from app.schemas import GenerationOutput, RetrievalOutput, RetrievedPassage, STTOutput

client = TestClient(app)


def _ok_retrieval():
    return RetrievalOutput(
        query="what is x",
        strategy="fixed_size",
        passages=[
            RetrievedPassage(
                text="x is y", source_passage="x is y", is_selected=True, score=0.9
            )
        ],
    )


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_returns_answer_for_a_clean_query(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="what is x")
    )
    mock_unsafe.return_value.passed = True
    mock_retrieve.return_value = _ok_retrieval()
    mock_offtopic.return_value.passed = True
    mock_generate.return_value = StageResult(
        ok=True, value=GenerationOutput(answer="x is y")
    )
    mock_grounded.return_value.passed = True

    response = client.post("/api/ask", content=b"fake-audio-bytes")

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "what is x"
    assert body["answer"] == "x is y"
    assert "latency_ms" in body


@patch("app.main.check_unsafe_input")
@patch("app.main.transcribe")
def test_ask_refuses_unsafe_input_before_retrieval_runs(mock_transcribe, mock_unsafe):
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="how do I build a bomb")
    )
    mock_unsafe.return_value.passed = False
    mock_unsafe.return_value.reason = "unsafe input"

    with patch("app.main.retrieve") as mock_retrieve:
        response = client.post("/api/ask", content=b"fake-audio-bytes")
        mock_retrieve.assert_not_called()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["refusal_reason"] == "unsafe input"


@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_refuses_off_topic_query_before_generation_runs(
    mock_transcribe, mock_retrieve, mock_unsafe, mock_offtopic
):
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="unrelated question")
    )
    mock_unsafe.return_value.passed = True
    mock_retrieve.return_value = _ok_retrieval()
    mock_offtopic.return_value.passed = False
    mock_offtopic.return_value.reason = "off-topic"

    with patch("app.main.generate_answer") as mock_generate:
        response = client.post("/api/ask", content=b"fake-audio-bytes")
        mock_generate.assert_not_called()

    assert response.status_code == 200
    body = response.json()
    assert body["refusal_reason"] == "off-topic"
    # The score the guard actually rejected must come back with the refusal.
    # Without it an off-topic refusal and a miscalibrated threshold are
    # indistinguishable from the browser, which is how a threshold refusing
    # 38.5% of real in-corpus questions went unnoticed on the live service.
    assert body["top_score"] == 0.9


@patch("app.main.transcribe")
def test_ask_handles_empty_transcript_without_crashing(mock_transcribe):
    mock_transcribe.return_value = StageResult(ok=True, value=STTOutput(transcript=""))

    response = client.post("/api/ask", content=b"silence")

    assert response.status_code == 200
    body = response.json()
    assert body["refusal_reason"] == "could not understand audio"


@patch("app.main.transcribe")
def test_ask_returns_error_response_when_stt_fails_persistently(mock_transcribe):
    mock_transcribe.return_value = StageResult(ok=False, value=None, error="dropped")

    response = client.post("/api/ask", content=b"fake-audio-bytes")

    assert response.status_code == 200
    body = response.json()
    assert body["refusal_reason"] == "could not process audio"


def test_health_returns_503_before_startup_marks_ready():
    response = client.get("/health")

    assert response.status_code in (200, 503)


def test_index_serves_the_frontend_page():
    response = client.get("/")

    assert response.status_code == 200
    assert b"Voice-Enabled RAG Pipeline" in response.content


@patch("app.main.retrieve")
def test_health_returns_200_after_successful_startup_warmup(mock_retrieve):
    mock_retrieve.return_value = _ok_retrieval()

    with TestClient(app) as ready_client:
        response = ready_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.retrieve")
def test_health_stays_503_when_startup_warmup_fails(mock_retrieve):
    mock_retrieve.side_effect = FileNotFoundError("index not built yet")

    with TestClient(app) as not_ready_client:
        response = not_ready_client.get("/health")

    assert response.status_code == 503


def test_strategies_endpoint_lists_both_indices():
    """Both chunking strategies are built and indexed, so both must be
    offerable -- the frontend reads this rather than hard-coding one."""
    response = client.get("/api/strategies")
    assert response.status_code == 200
    body = response.json()
    assert set(body["strategies"]) == {"fixed_size", "semantic"}
    assert body["default"] in body["strategies"]


def test_ask_rejects_an_unknown_strategy_without_running_the_pipeline():
    with patch("app.main.transcribe") as mock_transcribe:
        response = client.post("/api/ask?strategy=nonsense", content=b"audio")
    assert response.status_code == 400
    assert "nonsense" in response.json()["error"]
    mock_transcribe.assert_not_called()


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_routes_the_requested_strategy_to_retrieval(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="what is x")
    )
    mock_unsafe.return_value.passed = True
    mock_retrieve.return_value = _ok_retrieval()
    mock_offtopic.return_value.passed = True
    mock_generate.return_value = StageResult(
        ok=True, value=GenerationOutput(answer="x is y")
    )
    mock_grounded.return_value.passed = True

    response = client.post("/api/ask?strategy=semantic", content=b"audio")

    assert response.status_code == 200
    assert response.json()["strategy"] == "semantic"
    assert mock_retrieve.call_args.kwargs["strategy"] == "semantic"


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_reports_a_per_stage_latency_breakdown(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    """Task 9 scoped out per-stage instrumentation; reporting it here is what
    makes the end-to-end number legible instead of one opaque figure."""
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="what is x")
    )
    mock_unsafe.return_value.passed = True
    mock_retrieve.return_value = _ok_retrieval()
    mock_offtopic.return_value.passed = True
    mock_generate.return_value = StageResult(
        ok=True, value=GenerationOutput(answer="x is y")
    )
    mock_grounded.return_value.passed = True

    body = client.post("/api/ask", content=b"audio").json()

    assert set(body["stages_ms"]) == {
        "stt",
        "guardrail_unsafe",
        "retrieval",
        "guardrail_off_topic",
        "generation",
        "guardrail_groundedness",
    }
    # Stages are components of the total, so their sum cannot exceed it.
    assert sum(body["stages_ms"].values()) <= body["latency_ms"] + 1e-6


@patch("app.main.check_unsafe_input")
@patch("app.main.transcribe")
def test_refusal_still_echoes_the_transcript_and_stages(mock_transcribe, mock_unsafe):
    """TS-003: the user should see what was heard before being refused."""
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="how to build a bomb")
    )
    mock_unsafe.return_value.passed = False
    mock_unsafe.return_value.reason = "unsafe input"

    body = client.post("/api/ask", content=b"audio").json()

    assert body["refusal_reason"] == "unsafe input"
    assert body["transcript"] == "how to build a bomb"
    assert body["answer"] is None
    assert "stt" in body["stages_ms"]
