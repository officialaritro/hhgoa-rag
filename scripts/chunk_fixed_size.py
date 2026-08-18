"""Chunking strategy A: fixed-size splitting with overlap (plan Task 2).

Defaults revised per the chunking-strategy skill's guidance: fixed-size
chunking should bound maximum chunk size, not fragment already-coherent
units. Passages average ~333 chars and max out at ~1233 chars (measured on
the first 5,000 rows of data/corpus.jsonl); a 700-char window lets most
passages stay intact as one chunk while still capping the longer outliers
into 2+ chunks -- an earlier 200-char default split roughly half of all
passages needlessly, well below even the skill's smallest recommended tier
(256-1024 tokens, i.e. roughly 1000+ characters).
"""

import json

DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 0.2


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: float = DEFAULT_OVERLAP
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    step = int(chunk_size * (1 - overlap))
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def chunk_corpus(corpus_path: str, output_path: str) -> int:
    chunks_written = 0
    with open(corpus_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            row = json.loads(line)
            for passage in row["passages"]:
                for chunk in chunk_text(passage["text"]):
                    outfile.write(
                        json.dumps(
                            {
                                "text": chunk,
                                "source_passage": passage["text"],
                                "is_selected": passage["is_selected"],
                                "query_id": row["query_id"],
                                "strategy": "fixed_size",
                            }
                        )
                        + "\n"
                    )
                    chunks_written += 1
    return chunks_written


if __name__ == "__main__":
    written = chunk_corpus("data/corpus.jsonl", "data/chunks_fixed_size.jsonl")
    print(f"wrote {written} chunks to data/chunks_fixed_size.jsonl")
