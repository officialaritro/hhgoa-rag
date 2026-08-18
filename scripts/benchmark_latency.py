"""Latency benchmark harness (plan Task 9).

Measures the FULL pipeline (spoken audio in -> answer out, including
speech-to-text) by sending real synthesized audio through the live
/api/ask endpoint -- scripts/generate_test_audio.py produces that audio.
This benchmark cycles through a small fixed set of clips to reach the
requested batch size; it measures latency, not retrieval quality/diversity
(that is scripts/evaluate_retrieval.py's job, using real corpus text
queries directly).
"""

import itertools
import time
from typing import Any

import numpy as np

LATENCY_TARGET_MS = 200
DEFAULT_WARMUP_QUERIES = 3
DEFAULT_BATCH_SIZE = 30


def compute_percentiles(
    latencies_ms: list[float], target_ms: float = LATENCY_TARGET_MS
) -> dict[str, Any]:
    p50 = float(np.percentile(latencies_ms, 50))
    p70 = float(np.percentile(latencies_ms, 70))
    p100 = float(np.percentile(latencies_ms, 100))
    return {
        "p50": p50,
        "p70": p70,
        "p100": p100,
        "p50_under_target": p50 < target_ms,
        "p70_under_target": p70 < target_ms,
        "p100_under_target": p100 < target_ms,
    }


def run_batch(client: Any, audio_paths: list[str]) -> list[float]:
    """Sends each audio file's bytes to the live /api/ask endpoint, timing
    the full round trip (network + STT + retrieval + generation +
    guardrails)."""
    latencies_ms = []
    for path in audio_paths:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        start = time.perf_counter()
        client.post("/api/ask", content=audio_bytes)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return latencies_ms


def run_benchmark(
    audio_paths: list[str],
    warmup: int = DEFAULT_WARMUP_QUERIES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Cycles audio_paths (a small fixed clip set) to reach warmup+batch_size
    requests -- a latency benchmark needs enough real round trips for a
    stable percentile, not query diversity. base_url=None benchmarks the app
    in-process (local dev); a real URL benchmarks the deployed instance
    (Task 10), which is what the submission's numbers must reflect."""
    total_needed = warmup + batch_size
    cycled_paths = list(itertools.islice(itertools.cycle(audio_paths), total_needed))
    warmup_paths, measured_paths = cycled_paths[:warmup], cycled_paths[warmup:]

    if base_url:
        import httpx

        with httpx.Client(base_url=base_url) as client:
            run_batch(client, warmup_paths)
            latencies_ms = run_batch(client, measured_paths)
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            run_batch(client, warmup_paths)
            latencies_ms = run_batch(client, measured_paths)

    result = compute_percentiles(latencies_ms)
    result["warmup_queries_excluded"] = warmup
    result["batch_size"] = batch_size
    return result


if __name__ == "__main__":
    import argparse
    import json

    from scripts.generate_test_audio import synthesize_test_audio

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=None,
        help="e.g. http://<ec2-ip>:8000 for a live deployed run; omit for a local in-process run",
    )
    args = parser.parse_args()

    clip_paths = synthesize_test_audio()
    report = run_benchmark(clip_paths, base_url=args.base_url)
    print(json.dumps(report, indent=2))
