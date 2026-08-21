from unittest.mock import MagicMock, patch

import pytest

from scripts.benchmark_latency import compute_percentiles, run_batch, run_benchmark


def test_compute_percentiles_p50_p70_p100_on_known_values():
    # 10 values, 1..10 ms -- percentiles are easy to hand-verify.
    latencies_ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    result = compute_percentiles(latencies_ms)

    assert result["p50"] == 5.5
    assert result["p70"] == 7.3
    assert result["p100"] == 10


def test_compute_percentiles_flags_under_target_correctly():
    result = compute_percentiles([50, 60, 70], target_ms=100)

    assert result["p50_under_target"] is True
    assert result["p100_under_target"] is True


def test_compute_percentiles_flags_over_target_correctly():
    result = compute_percentiles([150, 250, 500], target_ms=200)

    assert result["p50_under_target"] is False
    assert result["p100_under_target"] is False


def test_run_batch_times_each_request_and_returns_one_latency_per_audio_path(tmp_path):
    audio_paths = []
    for i in range(3):
        path = tmp_path / f"clip_{i}.pcm"
        path.write_bytes(b"fake-pcm-audio")
        audio_paths.append(str(path))
    mock_client = MagicMock()

    latencies = run_batch(mock_client, audio_paths)

    assert len(latencies) == 3
    # run_batch returns (latency_ms, stages) per request now: the per-stage
    # breakdown is what lets boundary A be reported apart from the two
    # third-party calls that dominate the total.
    assert all(latency_ms >= 0 for latency_ms, _ in latencies)
    assert all(isinstance(stages, dict) for _, stages in latencies)
    assert mock_client.post.call_count == 3
    mock_client.post.assert_called_with("/api/ask", content=b"fake-pcm-audio")


@patch("scripts.benchmark_latency.run_batch")
def test_run_benchmark_discards_warmup_latencies_from_the_report(
    mock_run_batch, tmp_path
):
    audio_path = tmp_path / "clip.pcm"
    audio_path.write_bytes(b"audio")
    # First call = warmup batch (discarded), second call = measured batch.
    mock_run_batch.side_effect = [
        [(9999.0, {})],
        [(10, {"retrieval": 1.0}), (20, {"retrieval": 2.0}), (30, {"retrieval": 3.0})],
    ]

    result = run_benchmark([str(audio_path)], warmup=1, batch_size=3)

    assert result["p100"] == 30
    assert 9999.0 not in (result["p50"], result["p70"], result["p100"])
    assert result["warmup_queries_excluded"] == 1
    assert result["batch_size"] == 3


@patch("scripts.benchmark_latency.run_batch")
def test_run_benchmark_cycles_short_audio_list_to_reach_requested_total(
    mock_run_batch, tmp_path
):
    audio_path = tmp_path / "clip.pcm"
    audio_path.write_bytes(b"audio")
    mock_run_batch.return_value = [(10, {"retrieval": 1.0})] * 5

    run_benchmark([str(audio_path)], warmup=2, batch_size=3)

    warmup_call_paths, measured_call_paths = (
        call.args[1] for call in mock_run_batch.call_args_list
    )
    assert len(warmup_call_paths) == 2
    assert len(measured_call_paths) == 3


def test_boundary_latencies_splits_owned_cost_from_third_party():
    """The task asks for one number, "under 200ms". The pipeline contains two
    third-party calls whose latency we do not control, so a single figure hides
    where the time goes. Boundary A is what this system owns; B adds generation;
    C is what the user waits for."""
    from scripts.benchmark_latency import boundary_latencies

    records = [
        (
            3000.0,
            {
                "stt": 1200.0,
                "guardrail_unsafe": 0.0,
                "retrieval": 100.0,
                "guardrail_off_topic": 0.2,
                "generation": 1400.0,
                "guardrail_groundedness": 30.0,
            },
        )
    ]

    boundaries = boundary_latencies(records)

    assert boundaries["A"] == [pytest.approx(130.2)]
    assert boundaries["B"] == [pytest.approx(1530.2)]
    assert boundaries["C"] == [pytest.approx(3000.0)]


def test_boundary_a_counts_every_guardrail_stage():
    """A guardrail added later must not silently fall outside the boundary the
    200ms claim is made against."""
    from scripts.benchmark_latency import boundary_latencies

    records = [
        (
            100.0,
            {
                "retrieval": 10.0,
                "guardrail_unsafe": 1.0,
                "guardrail_off_topic": 2.0,
                "guardrail_groundedness": 3.0,
                "guardrail_something_new": 4.0,
                "generation": 50.0,
                "stt": 20.0,
            },
        )
    ]

    assert boundary_latencies(records)["A"] == [pytest.approx(20.0)]


def test_boundary_latencies_tolerates_a_refusal_with_missing_stages():
    """A refused request never reaches generation, so its stage dict is short.
    That must not raise or silently count as zero-latency success."""
    from scripts.benchmark_latency import boundary_latencies

    records = [(500.0, {"stt": 400.0, "guardrail_unsafe": 0.0, "retrieval": 20.0})]

    boundaries = boundary_latencies(records)

    assert boundaries["A"] == [pytest.approx(20.0)]
    assert boundaries["B"] == [pytest.approx(20.0)]


def test_boundary_a_includes_the_rerank_stage():
    """Reranking is inside the retrieval boundary the 200ms target covers. When
    it was reported as its own stage, a boundary rule matching only `retrieval`
    and `guardrail*` would have silently dropped ~53ms from every boundary A
    figure -- understating the number the submission is judged on."""
    from scripts.benchmark_latency import boundary_latencies

    records = [
        (
            500.0,
            {
                "stt": 300.0,
                "retrieval": 20.0,
                "rerank": 53.0,
                "guardrail_off_topic": 0.2,
                "guardrail_groundedness": 12.0,
                "generation": 100.0,
            },
        )
    ]

    assert boundary_latencies(records)["A"] == [pytest.approx(85.2)]
