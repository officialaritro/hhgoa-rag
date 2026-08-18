"""Builds the FAISS index + metadata store for each chunking strategy
(plan Task 3). Invoked by deploy/setup.sh on first deploy.
"""

from app.indexing import build_index
from app.retrieval import INDEX_PATHS

_CHUNKS_PATHS = {
    "fixed_size": "data/chunks_fixed_size.jsonl",
    "semantic": "data/chunks_semantic.jsonl",
}

if __name__ == "__main__":
    for strategy, chunks_path in _CHUNKS_PATHS.items():
        index_path, metadata_path = INDEX_PATHS[strategy]
        written = build_index(chunks_path, index_path, metadata_path)
        print(f"{strategy}: indexed {written} chunks -> {index_path}")
