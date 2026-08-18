from unittest.mock import MagicMock, patch

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
    assert all(latency_ms >= 0 for latency_ms in latencies)
    assert mock_client.post.call_count == 3
    mock_client.post.assert_called_with("/api/ask", content=b"fake-pcm-audio")


@patch("scripts.benchmark_latency.run_batch")
def test_run_benchmark_discards_warmup_latencies_from_the_report(
    mock_run_batch, tmp_path
):
    audio_path = tmp_path / "clip.pcm"
    audio_path.write_bytes(b"audio")
    # First call = warmup batch (discarded), second call = measured batch.
    mock_run_batch.side_effect = [[9999.0], [10, 20, 30]]

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
    mock_run_batch.return_value = [10] * 5

    run_benchmark([str(audio_path)], warmup=2, batch_size=3)

    warmup_call_paths, measured_call_paths = (
        call.args[1] for call in mock_run_batch.call_args_list
    )
    assert len(warmup_call_paths) == 2
    assert len(measured_call_paths) == 3
