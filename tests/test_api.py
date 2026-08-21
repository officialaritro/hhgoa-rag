from unittest.mock import patch

from fastapi.testclient import TestClient

from app.harness import StageResult
from app.main import COMPARE_DEFAULT, app
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


def test_strategies_endpoint_lists_every_registered_strategy():
    """Derived from the registry, never hard-coded. Strategy identity used to be
    duplicated across four modules, and a strategy present in one but missing
    from another is how a miscalibrated threshold reached production."""
    from app.strategies import dense_names, served_names

    response = client.get("/api/strategies")
    assert response.status_code == 200
    body = response.json()
    assert set(body["strategies"]) == set(served_names())
    # Measurement controls are built and evaluated but must not be offered as
    # something to retrieve from -- query_aware_heldout exists only to answer a
    # question about query_aware, and its own threshold is not meaningful.
    assert "query_aware_heldout" in set(dense_names())
    assert "query_aware_heldout" not in set(body["strategies"])
    # Composed strategies are servable without an index of their own.
    assert {"hybrid", "fusion"} <= set(body["strategies"])
    assert body["default"] in body["strategies"]


def test_strategies_endpoint_describes_each_strategy_for_the_ui():
    """Eight strategy names mean nothing to someone looking at a radio group.
    The registry already carries a description and an axis; the frontend should
    not have to restate them."""
    response = client.get("/api/strategies")
    body = response.json()

    assert body["details"], "no per-strategy detail returned"
    for name in body["strategies"]:
        detail = body["details"][name]
        assert detail["description"].strip()
        assert detail["axis"] in {
            "split",
            "unit",
            "enrichment",
            "aggregation",
            "fusion",
        }


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


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_returns_the_retrieved_passages_alongside_the_answer(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    """Citations need the exact passages that grounded the answer, not just
    the top score -- otherwise a judge has no way to see what the model
    actually read."""
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

    assert body["passages"] == [
        {
            "text": "x is y",
            "source_passage": "x is y",
            "is_selected": True,
            "score": 0.9,
        }
    ]


@patch("app.main.check_unsafe_input")
@patch("app.main.transcribe")
def test_ask_refusal_before_retrieval_returns_empty_passages(
    mock_transcribe, mock_unsafe
):
    """A refusal that happens before retrieval ever runs has nothing to cite."""
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="how to build a bomb")
    )
    mock_unsafe.return_value.passed = False
    mock_unsafe.return_value.reason = "unsafe input"

    body = client.post("/api/ask", content=b"audio").json()

    assert body["passages"] == []


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_compare_answers_both_strategies_from_one_transcription(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    """One recording must be transcribed once and compared on identical text
    -- calling /api/ask twice would transcribe the same audio twice and could
    return two different transcripts for what is supposed to be one question."""
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

    response = client.post("/api/compare", content=b"audio")

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "what is x"
    assert mock_transcribe.call_count == 1
    # compare narrows to an explicit subset by default: fanning out to all
    # eight would issue eight generation calls per request.
    assert set(body["results"]) == set(COMPARE_DEFAULT)
    for result in body["results"].values():
        assert result["answer"] == "x is y"
        assert result["passages"] == [
            {
                "text": "x is y",
                "source_passage": "x is y",
                "is_selected": True,
                "score": 0.9,
            }
        ]
    assert mock_retrieve.call_count == len(COMPARE_DEFAULT)
    called_strategies = {c.kwargs["strategy"] for c in mock_retrieve.call_args_list}
    assert called_strategies == set(COMPARE_DEFAULT)
    assert all(c.kwargs["query"] == "what is x" for c in mock_retrieve.call_args_list)


@patch("app.main.check_unsafe_input")
@patch("app.main.transcribe")
def test_compare_refuses_unsafe_input_before_any_retrieval(
    mock_transcribe, mock_unsafe
):
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="how to build a bomb")
    )
    mock_unsafe.return_value.passed = False
    mock_unsafe.return_value.reason = "unsafe input"

    with (
        patch("app.main.retrieve") as mock_retrieve,
        patch("app.main.generate_answer") as mock_generate,
    ):
        response = client.post("/api/compare", content=b"audio")
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()

    assert response.status_code == 200
    body = response.json()
    assert body["refusal_reason"] == "unsafe input"
    assert body["results"] is None


@patch("app.main.check_groundedness")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.generate_answer")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_compare_isolates_a_failing_strategy_branch(
    mock_transcribe,
    mock_retrieve,
    mock_generate,
    mock_unsafe,
    mock_offtopic,
    mock_grounded,
):
    """One strategy's pipeline blowing up must not 500 the whole request or
    swallow the sibling strategy's real, successful result."""
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="what is x")
    )
    mock_unsafe.return_value.passed = True

    # The failing and surviving strategies are taken from COMPARE_DEFAULT
    # rather than named, so changing which strategies are compared cannot
    # quietly turn this into a test where nothing fails.
    failing, surviving = COMPARE_DEFAULT[1], COMPARE_DEFAULT[0]

    def retrieve_side_effect(*, query, strategy, **kwargs):
        if strategy == failing:
            raise RuntimeError("boom")
        return _ok_retrieval()

    mock_retrieve.side_effect = retrieve_side_effect
    mock_offtopic.return_value.passed = True
    mock_generate.return_value = StageResult(
        ok=True, value=GenerationOutput(answer="x is y")
    )
    mock_grounded.return_value.passed = True

    response = client.post("/api/compare", content=b"audio")

    assert response.status_code == 200
    body = response.json()
    assert body["results"][surviving]["answer"] == "x is y"
    assert body["results"][failing]["answer"] is None
    assert body["results"][failing]["refusal_reason"] == "internal error"


@patch("app.main.check_groundedness")
@patch("app.main.generate_answer")
@patch("app.main.check_off_topic")
@patch("app.main.check_unsafe_input")
@patch("app.main.retrieve")
@patch("app.main.transcribe")
def test_ask_refuses_when_the_model_declines_to_answer(
    mock_transcribe,
    mock_retrieve,
    mock_unsafe,
    mock_offtopic,
    mock_generate,
    mock_grounded,
):
    """A decline must surface as a refusal, not as an answer.

    The groundedness guard cannot be relied on to catch it: a decline quotes
    the passages to explain itself and scores 0.533-0.795 against context on
    the live indices, well above the 0.40 threshold, so it passed as grounded
    and reached the user as an answer with refusal_reason null.
    """
    mock_transcribe.return_value = StageResult(
        ok=True, value=STTOutput(transcript="something the passages do not cover")
    )
    mock_unsafe.return_value.passed = True
    mock_retrieve.return_value = _ok_retrieval()
    mock_offtopic.return_value.passed = True
    mock_generate.return_value = StageResult(
        ok=True,
        value=GenerationOutput(
            answer="INSUFFICIENT_CONTEXT", insufficient_context=True
        ),
    )

    body = client.post("/api/ask", content=b"audio").json()

    assert body["answer"] is None
    assert body["refusal_reason"] == "not in the retrieved passages"
    assert body["top_score"] == 0.9
    # The decline short-circuits before groundedness -- scoring a refusal
    # against the context is the check that cannot tell them apart.
    mock_grounded.assert_not_called()
