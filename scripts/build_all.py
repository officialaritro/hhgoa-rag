"""Unattended overnight index build for every registered chunking strategy.

Designed to be started and left alone: plug the machine in, run one command, go
to sleep. Everything below exists because something in that sentence can go
wrong at 3am with nobody watching.

**Two processes per strategy, never one.** On macOS arm64 faiss-cpu and torch
each bundle their own libomp. A process holding both aborts with `OMP: Error
#15` (exit 134), and the documented `KMP_DUPLICATE_LIB_OK=TRUE` escape hatch
only downgrades that to a segfault (exit 139) once MPS is in use. Both measured
on this machine. So embedding and indexing run as separate child processes --
this module re-invokes itself with `--phase` -- and the orchestrator itself
imports neither torch nor faiss.

**It resumes.** A run killed at strategy six does not redo the first five, and a
strategy whose vectors were written before the index step died is re-indexed
without re-embedding the expensive half.

**It stays awake.** The script re-executes itself under `caffeinate` so an idle
Mac does not suspend a 50-minute job ten minutes in.

Run:
    .venv/bin/python -m scripts.build_all
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.passages import build_passage_store, load_passage_store
from app.strategies import chunk_paths, get, names
from app.vectors import sidecar_path

CORPUS_PATH = "data/corpus.jsonl"
PASSAGE_STORE_PATH = "data/passages.pkl"
INSTANCE = "ubuntu@13.234.228.244"
PROGRESS_PREFIX = "@@PROGRESS "
DEFAULT_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 1024


class PhaseFailed(RuntimeError):
    """A phase exhausted its retries. Carries the last underlying cause so the
    morning's log says what actually went wrong, not just that something did."""


def vectors_path_for(strategy: str) -> str:
    """Transient float32 vectors, deleted once the index is built. The largest
    strategy is ~537 MB, so these are not kept around."""
    return f"data/vectors_{strategy}.f32"


def format_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def render_progress(
    label: str, done: int, total: int | None, elapsed: float
) -> str:
    """One status line. `total` is None for strategies that cannot know their
    chunk count without doing the chunking twice -- those degrade to a count
    rather than inventing a percentage."""
    rate = done / elapsed if elapsed > 0 else 0.0
    parts = [f"{label:<28}"]
    if total:
        fraction = min(1.0, done / total)
        filled = int(fraction * 24)
        parts.append(f"[{'#' * filled}{'.' * (24 - filled)}]")
        parts.append(f"{int(fraction * 100):3d}%")
        parts.append(f"{done:>9,}/{total:,}")
    else:
        parts.append(f"{done:>9,} chunks")
    parts.append(f"{rate:>7,.0f}/s")
    if total and rate > 0:
        parts.append(f"ETA {format_duration((total - done) / rate)}")
    return "  ".join(parts)


def work_remaining(
    index_path: str, force: bool = False, vectors_path: str | None = None
) -> tuple[str, ...]:
    """Which phases still need running for one strategy.

    An index with no manifest is treated as absent: that is the signature of a
    build killed between writing the two, and `load_index` refuses such an index
    anyway rather than serving from it.
    """
    if force:
        return ("embed", "index")
    manifest = Path(index_path).with_suffix(".manifest.json")
    if Path(index_path).exists() and manifest.exists():
        return ()
    if (
        vectors_path
        and Path(vectors_path).exists()
        and Path(sidecar_path(vectors_path)).exists()
    ):
        return ("index",)
    return ("embed", "index")


def run_with_retries(
    fn: Callable[[], Any],
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = 15.0,
    log: Callable[[str], None] = print,
) -> Any:
    """Retries a phase, then fails loudly.

    Raising rather than returning None matters: a None would let the caller
    record the strategy as built and move on, leaving a gap discovered only when
    the service tried to serve it.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 -- retry ANY failure, then re-raise wrapped
            if isinstance(exc, KeyboardInterrupt):
                raise
            last = exc
            log(f"    attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise PhaseFailed(f"failed after {attempts} attempts: {last}") from last


# ---------------------------------------------------------------- child phases


def _phase_embed(strategy_name: str, batch_size: int) -> int:
    """Child process: torch only. Never imports faiss."""
    from app.vectors import embed_chunks_to_disk

    passages = load_passage_store(PASSAGE_STORE_PATH)
    _, metadata_path = chunk_paths(strategy_name)
    strategy = get(strategy_name)
    assert strategy.chunker is not None

    last_emit = 0.0

    def emit(done: int) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if now - last_emit >= 0.25:
            print(f"{PROGRESS_PREFIX}{done}", flush=True)
            last_emit = now

    return embed_chunks_to_disk(
        chunks=strategy.chunker(passages),
        passages=passages,
        vectors_path=vectors_path_for(strategy_name),
        metadata_path=metadata_path,
        batch_size=batch_size,
        progress=emit,
    )


def _phase_index(strategy_name: str) -> int:
    """Child process: faiss only. Never imports torch."""
    from app.indexing import index_from_vectors

    index_path, metadata_path = chunk_paths(strategy_name)
    return index_from_vectors(
        vectors_path_for(strategy_name),
        index_path,
        metadata_path,
        progress=lambda done: print(f"{PROGRESS_PREFIX}{done}", flush=True),
    )


# --------------------------------------------------------------- orchestration


def _spawn_phase(
    phase: str,
    strategy: str,
    total: int | None,
    batch_size: int,
    log: Callable[[str], None],
) -> None:
    """Runs one phase as a child process, rendering its progress lines."""
    label = f"{strategy}/{phase}"
    started = time.monotonic()
    command = [
        sys.executable,
        "-m",
        "scripts.build_all",
        "--phase",
        phase,
        "--strategy",
        strategy,
        "--batch-size",
        str(batch_size),
    ]
    env = {**os.environ, "_BUILD_ALL_CAFFEINATED": "1"}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    interactive = sys.stdout.isatty()
    last_logged = 0.0
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        if line.startswith(PROGRESS_PREFIX):
            done = int(line[len(PROGRESS_PREFIX) :])
            rendered = render_progress(label, done, total, time.monotonic() - started)
            if interactive:
                print(f"\r  {rendered}", end="", flush=True)
            elif time.monotonic() - last_logged >= 30:
                # Not a terminal: carriage returns would produce one unreadable
                # mega-line in the log file, so emit a timestamped line instead.
                log(f"  {rendered}")
                last_logged = time.monotonic()
        elif line.strip():
            tail.append(line)
            del tail[:-40]
    process.wait()
    if interactive:
        print()
    if process.returncode != 0:
        raise RuntimeError(
            f"{label} exited {process.returncode}"
            + (f"; last output: {tail[-1]}" if tail else "")
        )


def _preflight(log: Callable[[str], None]) -> None:
    if not Path(CORPUS_PATH).exists():
        raise SystemExit(
            f"{CORPUS_PATH} not found.\n\n"
            f"Copy it down from the instance rather than re-running ingestion --\n"
            f"the corpus must be byte-identical to the one the measured findings\n"
            f"and chunk counts came from:\n\n"
            f"  scp -i ~/.ssh/hhgoa-rag-key.pem \\\n"
            f"      {INSTANCE}:/opt/hhgoa-rag/data/corpus.jsonl data/\n"
        )
    free_gb = shutil.disk_usage(".").free / 1e9
    log(f"  disk free: {free_gb:.1f} GB")
    if free_gb < 3:
        raise SystemExit(
            f"only {free_gb:.1f} GB free; the build needs room for ~450 MB of "
            f"indices plus one ~540 MB transient vector file at a time"
        )
    device = os.environ.get("EMBEDDING_DEVICE", "")
    if sys.platform == "darwin" and device != "mps":
        log(
            "  WARNING: EMBEDDING_DEVICE is not 'mps'. On Apple Silicon that is "
            "326 texts/sec instead of 506 -- a 78-minute build instead of 50."
        )
    else:
        log(f"  embedding device: {device or 'auto'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("embed", "index"))
    parser.add_argument("--strategy")
    parser.add_argument("--strategies", help="comma-separated subset to build")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument("--keep-vectors", action="store_true")
    parser.add_argument("--no-caffeinate", action="store_true")
    args = parser.parse_args(argv)

    # Child phase: do the one job and exit. Keeps torch and faiss apart.
    if args.phase:
        count = (
            _phase_embed(args.strategy, args.batch_size)
            if args.phase == "embed"
            else _phase_index(args.strategy)
        )
        print(f"{PROGRESS_PREFIX}{count}", flush=True)
        return 0

    # Parent: keep the machine awake for the duration.
    if (
        sys.platform == "darwin"
        and not args.no_caffeinate
        and not os.environ.get("_BUILD_ALL_CAFFEINATED")
        and shutil.which("caffeinate")
    ):
        os.environ["_BUILD_ALL_CAFFEINATED"] = "1"
        os.execvp(  # noqa: S606 -- fixed argv, no shell
            "caffeinate",
            ["caffeinate", "-ims", sys.executable, "-m", "scripts.build_all", *sys.argv[1:]],
        )

    Path("logs").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"build_{stamp}.log"
    log_file = log_path.open("w")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    started = time.monotonic()
    log(f"=== index build started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"log: {log_path}")
    _preflight(log)

    if args.force or not Path(PASSAGE_STORE_PATH).exists():
        log("building shared passage store ...")
        total_passages = build_passage_store(CORPUS_PATH, PASSAGE_STORE_PATH)
    else:
        total_passages = len(load_passage_store(PASSAGE_STORE_PATH))
    log(f"  passages: {total_passages:,}")

    selected = (
        tuple(s.strip() for s in args.strategies.split(",")) if args.strategies else names()
    )
    log(f"strategies: {', '.join(selected)}\n")

    results: list[dict[str, Any]] = []
    for strategy_name in selected:
        strategy = get(strategy_name)
        index_path, _ = chunk_paths(strategy_name)
        vectors = vectors_path_for(strategy_name)
        phases = work_remaining(index_path, args.force, vectors)
        if not phases:
            log(f"{strategy_name}: already built, skipping")
            results.append({"strategy": strategy_name, "status": "skipped"})
            continue

        log(f"{strategy_name}: {strategy.description}")
        entry: dict[str, Any] = {"strategy": strategy_name, "status": "built"}
        phase_started = time.monotonic()
        try:
            for phase in phases:
                # Only the embedding phase has a knowable total up front, and
                # only because whole_passage is one chunk per passage.
                total = (
                    total_passages
                    if phase == "embed" and strategy_name == "whole_passage"
                    else None
                )
                run_with_retries(
                    lambda p=phase, t=total: _spawn_phase(
                        p, strategy_name, t, args.batch_size, log
                    ),
                    attempts=args.attempts,
                    log=log,
                )
        except PhaseFailed as exc:
            log(f"  FAILED: {exc}")
            entry["status"] = "failed"
            entry["error"] = str(exc)
            results.append(entry)
            continue

        entry["seconds"] = time.monotonic() - phase_started
        manifest = Path(index_path).with_suffix(".manifest.json")
        if manifest.exists():
            entry["chunks"] = json.loads(manifest.read_text()).get("chunks")
        entry["index_mb"] = Path(index_path).stat().st_size / 1e6
        if not args.keep_vectors:
            Path(vectors).unlink(missing_ok=True)
            Path(sidecar_path(vectors)).unlink(missing_ok=True)
        log(
            f"  done: {entry.get('chunks'):,} chunks, "
            f"{entry['index_mb']:.1f} MB, {format_duration(entry['seconds'])}"
        )
        results.append(entry)

    log(f"\n=== summary ({format_duration(time.monotonic() - started)} total) ===")
    log(f"{'strategy':<20} {'status':<9} {'chunks':>10} {'index MB':>9} {'time':>8}")
    for entry in results:
        chunks = f"{entry['chunks']:,}" if entry.get("chunks") else "-"
        mb = f"{entry['index_mb']:.1f}" if entry.get("index_mb") else "-"
        seconds = format_duration(entry["seconds"]) if entry.get("seconds") else "-"
        log(f"{entry['strategy']:<20} {entry['status']:<9} {chunks:>10} {mb:>9} {seconds:>8}")

    failed = [entry for entry in results if entry["status"] == "failed"]
    if failed:
        log(f"\n{len(failed)} strategy/strategies FAILED -- see above. Re-run to resume.")
    else:
        log("\nall strategies built.")
    log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
