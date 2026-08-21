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
# httpx defaults to a 5s timeout. A full run is speech-to-text (~1.2s) plus
# generation (~2-3s), so the default silently times out most requests and the
# benchmark reports the timeout rather than the pipeline.
REQUEST_TIMEOUT_SECONDS = 120.0


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


def boundary_latencies(
    records: list[tuple[float, dict[str, float]]],
) -> dict[str, list[float]]:
    """Splits each request into the three boundaries the report is written around.

    The task asks for a single figure under 200ms, but the pipeline contains two
    third-party calls whose latency this system does not control -- speech-to-text
    and answer generation -- so one number would hide where the time actually
    goes and would be unactionable.

      A  retrieval and every guardrail: what this system owns and can optimise
      B  A plus answer generation
      C  the full round trip the user waits for, including speech-to-text

    Boundary A sums `retrieval`, `rerank`, and every stage beginning with
    `guardrail`, rather than an explicit list, so a guardrail added later cannot
    quietly fall outside the boundary the 200ms claim is made against. `rerank`
    is named explicitly because it is inside that boundary: when it was split
    out of `retrieval` into its own stage, a rule matching only `retrieval` and
    `guardrail*` would have silently dropped ~53ms from every figure.
    """
    owned: list[float] = []
    with_generation: list[float] = []
    end_to_end: list[float] = []
    for total_ms, stages in records:
        a = sum(
            value
            for name, value in stages.items()
            if name in {"retrieval", "rerank"} or name.startswith("guardrail")
        )
        owned.append(a)
        with_generation.append(a + stages.get("generation", 0.0))
        end_to_end.append(total_ms)
    return {"A": owned, "B": with_generation, "C": end_to_end}


def run_batch(
    client: Any, audio_paths: list[str]
) -> list[tuple[float, dict[str, float]]]:
    """Sends each audio file's bytes to the live /api/ask endpoint, timing the
    full round trip (network + STT + retrieval + generation + guardrails) and
    keeping the per-stage breakdown the response carries.

    The breakdown is not optional detail: without it the only number available
    is end-to-end, which is dominated by two third-party calls and so cannot
    show whether the part this system owns meets its target.
    """
    records: list[tuple[float, dict[str, float]]] = []
    for path in audio_paths:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        start = time.perf_counter()
        response = client.post("/api/ask", content=audio_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            raw = response.json().get("stages_ms")
        except Exception:  # noqa: BLE001 -- a malformed body must not lose the timing
            raw = None
        # Type-checked rather than trusted: a response shape change or an error
        # body would otherwise put a non-mapping into the aggregation and fail
        # much later, in percentile arithmetic, where the cause is invisible.
        stages = raw if isinstance(raw, dict) else {}
        records.append((elapsed_ms, stages))
    return records


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

        with httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            run_batch(client, warmup_paths)
            records = run_batch(client, measured_paths)
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            run_batch(client, warmup_paths)
            records = run_batch(client, measured_paths)

    boundaries = boundary_latencies(records)
    # The headline stays end-to-end, because that is what the user waits for and
    # what a single-number reading of the target would mean. Per-boundary
    # percentiles sit alongside it so the claim about the part this system owns
    # is separable from two third-party calls.
    result = compute_percentiles(boundaries["C"])
    result["boundaries"] = {
        name: compute_percentiles(values) for name, values in boundaries.items()
    }
    result["stage_p50_ms"] = {
        stage: float(np.percentile([s.get(stage, 0.0) for _, s in records], 50))
        for stage in sorted({k for _, s in records for k in s})
    }
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
