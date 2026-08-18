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
