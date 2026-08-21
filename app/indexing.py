"""Builds and loads a FAISS index per chunking strategy.

The build half runs as its own process and never imports torch: on macOS arm64
faiss and torch each bundle their own libomp, and a process holding both
segfaults once MPS is used. app/vectors.py owns the embedding half. See
`index_from_vectors` below.
"""

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.embeddings import active_model_name
from app.vectors import read_vector_shape, sidecar_path


class IndexModelMismatch(RuntimeError):
    """An index was built with a different embedding model than is now active.

    This is the dangerous case, because nothing crashes on its own: both models
    emit 384-dim vectors, so FAISS searches happily and returns neighbours from
    an unrelated vector space. The answers look plausible and are meaningless.
    It happened in practice -- .env pointed at bge-small while the indices were
    built with all-MiniLM -- so the manifest below makes it loud instead.
    """


def _manifest_path(index_path: str) -> Path:
    return Path(index_path).with_suffix(".manifest.json")


def load_index(
    index_path: str, metadata_path: str
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Refuse to serve an index built by a different model rather than return
    # confident nonsense. Indices predating manifests load with a warning.
    manifest_file = _manifest_path(index_path)
    if manifest_file.exists():
        built_with = json.loads(manifest_file.read_text()).get("embedding_model")
        active = active_model_name()
        if built_with and built_with != active:
            raise IndexModelMismatch(
                f"{index_path} was built with {built_with!r} but "
                f"EMBEDDING_MODEL_NAME is {active!r}. Retrieval would search an "
                f"unrelated vector space and return plausible nonsense. Either set "
                f"EMBEDDING_MODEL_NAME={built_with!r} or rebuild the indices."
            )

    return index, metadata


def index_from_vectors(
    vectors_path: str,
    index_path: str,
    metadata_path: str,
    *,
    train_sample: int = 50_000,
    add_batch: int = 65_536,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Builds a FAISS index from the flat float32 file the embedding phase wrote.

    Runs as its own process, with torch never imported -- see app/vectors.py for
    the measured reason (faiss and torch each bundle libomp on macOS arm64;
    co-loading them aborts, and KMP_DUPLICATE_LIB_OK=TRUE turns the abort into
    a segfault once MPS is used).

    Vectors are memory-mapped and added in slices, so a 537 MB matrix never
    becomes a 537 MB allocation. The metadata row count is checked against the
    vector count first: if they disagree, FAISS row ids no longer address the
    right metadata row, and every answer would cite the wrong passage while
    still looking plausible.
    """
    count, dimension = read_vector_shape(vectors_path)
    sidecar = json.loads(Path(sidecar_path(vectors_path)).read_text())

    with open(metadata_path, "rb") as f:
        rows = pickle.load(f)
    if len(rows) != count:
        raise ValueError(
            f"metadata/vector mismatch for {index_path}: {len(rows)} metadata rows "
            f"against {count} vectors. FAISS returns row ids, so a mismatch makes "
            f"every result resolve the wrong parent passage. Rebuild both phases."
        )

    vectors = np.memmap(
        vectors_path, dtype="float32", mode="r", shape=(count, dimension)
    )
    index = faiss.IndexScalarQuantizer(
        dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    index.train(np.ascontiguousarray(vectors[: min(train_sample, count)]))

    for start in range(0, count, add_batch):
        index.add(np.ascontiguousarray(vectors[start : start + add_batch]))
        if progress is not None:
            progress(min(start + add_batch, count))

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    _manifest_path(index_path).write_text(
        json.dumps(
            {
                # From the sidecar, not active_model_name(): this process never
                # loads a model, so the embedding phase is the only truthful source.
                "embedding_model": sidecar["embedding_model"],
                "dimension": int(dimension),
                "chunks": int(count),
            },
            indent=2,
        )
    )
    return int(count)
