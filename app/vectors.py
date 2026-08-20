"""The embedding phase of an index build. Deliberately free of faiss.

On macOS arm64, faiss-cpu and torch each bundle their own copy of libomp.
Loading both into one process raises `OMP: Error #15` and aborts (exit 134);
setting the documented `KMP_DUPLICATE_LIB_OK=TRUE` escape hatch only converts
that into a segfault (exit 139) once MPS is actually used. Both were measured
on this machine. Since the overnight build embeds on MPS and writes a FAISS
index, the two halves have to run as separate processes that never co-load.

So this module embeds chunks and writes raw float32 vectors to disk; the
indexing process (app/indexing.py) memory-maps them and never imports torch.
Splitting the phases also means a failure while building the index does not
discard an hour of embedding, and peak memory drops to one batch rather than
the whole matrix.

The contract between the phases is deliberately dumb -- a flat float32 file
plus a JSON sidecar holding the shape and the model name -- so neither side has
to unpickle anything the other wrote.
"""

import json
import pickle
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from app.embeddings import active_model_name, embed_batch
from app.passages import PICKLE_PROTOCOL, Passage, resolve_text
from app.strategies import Chunk


def sidecar_path(vectors_path: str) -> str:
    return f"{vectors_path}.meta.json"


def read_vector_shape(vectors_path: str) -> tuple[int, int]:
    """(count, dimension) of the vectors written for this strategy."""
    meta = json.loads(Path(sidecar_path(vectors_path)).read_text())
    return int(meta["count"]), int(meta["dimension"])


def embed_chunks_to_disk(
    chunks: Iterator[Chunk],
    passages: list[Passage],
    vectors_path: str,
    metadata_path: str,
    *,
    batch_size: int = 1024,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Embeds every chunk in batches, appending float32 rows to `vectors_path`.

    Vector row order and metadata row order must stay identical: FAISS returns
    row ids, and the metadata row at that id is what resolves the parent
    passage. A drift between them makes every answer cite the wrong source
    while still looking plausible, so both are written from the same loop.
    """
    rows: list[Chunk] = []
    dimension = 0
    Path(vectors_path).parent.mkdir(parents=True, exist_ok=True)

    def flush(batch: list[Chunk], handle) -> None:
        nonlocal dimension
        texts = [resolve_text(row, passages) for row in batch]
        vectors = np.asarray(embed_batch(texts), dtype="float32")
        if dimension and vectors.shape[1] != dimension:
            raise ValueError(
                f"embedding dimension changed mid-build "
                f"({dimension} -> {vectors.shape[1]}); the model must not change "
                f"while an index is being built"
            )
        dimension = vectors.shape[1]
        handle.write(np.ascontiguousarray(vectors).tobytes())
        rows.extend(batch)
        if progress is not None:
            progress(len(rows))

    with open(vectors_path, "wb") as handle:
        batch: list[Chunk] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                flush(batch, handle)
                batch = []
        if batch:
            flush(batch, handle)

    if not rows:
        Path(vectors_path).unlink(missing_ok=True)
        raise ValueError(
            f"no chunks produced for {vectors_path}: the strategy's chunker "
            f"yielded nothing, which would leave an empty index that serves "
            f"zero results instead of failing"
        )

    with open(metadata_path, "wb") as f:
        pickle.dump(rows, f, protocol=PICKLE_PROTOCOL)
    Path(sidecar_path(vectors_path)).write_text(
        json.dumps(
            {
                "count": len(rows),
                "dimension": int(dimension),
                "embedding_model": active_model_name(),
            },
            indent=2,
        )
    )
    return len(rows)
