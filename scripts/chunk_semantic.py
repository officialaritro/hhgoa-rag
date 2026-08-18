"""Chunking strategy B: semantic splitting on embedding-similarity breakpoints
between sentences (plan Task 3) -- differs in method from Task 2's fixed-size
character windows, not just in parameters.
"""

import json
import re

from app.embeddings import cosine_similarity, embed_batch

# Chunking-strategy skill guidance recommends 0.8 for embedding-based
# semantic boundary detection -- 0.5 under-splits, since general
# sentence-embedding similarity between unrelated sentences commonly sits
# in the 0.3-0.6 range (embedding-space anisotropy), so 0.5 merges much
# more than genuine thematic continuity would justify.
DEFAULT_SIMILARITY_THRESHOLD = 0.8

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]


def chunk_text_semantic(
    text: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences if sentences else [text]

    embeddings = embed_batch(sentences)
    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
        if similarity >= similarity_threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    chunks.append(" ".join(current))
    return chunks


def chunk_corpus(corpus_path: str, output_path: str) -> int:
    chunks_written = 0
    with open(corpus_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            row = json.loads(line)
            for passage in row["passages"]:
                for chunk in chunk_text_semantic(passage["text"]):
                    outfile.write(
                        json.dumps(
                            {
                                "text": chunk,
                                "source_passage": passage["text"],
                                "is_selected": passage["is_selected"],
                                "query_id": row["query_id"],
                                "strategy": "semantic",
                            }
                        )
                        + "\n"
                    )
                    chunks_written += 1
    return chunks_written


if __name__ == "__main__":
    written = chunk_corpus("data/corpus.jsonl", "data/chunks_semantic.jsonl")
    print(f"wrote {written} chunks to data/chunks_semantic.jsonl")
