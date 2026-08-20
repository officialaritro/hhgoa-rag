"""The eight chunking strategies, as pure functions over the passage store.

Each yields `Chunk` spans rather than text. Four axes are represented, and the
axis is the point -- a slate that varies only chunk size is a parameter sweep,
not a strategy slate:

  split       whole_passage, fixed_size, recursive, semantic
  unit        parent_child, sentence_window   (embed one thing, return another)
  enrichment  query_aware                    (embed more than the passage)
  aggregation query_group                     (chunks that cross passages)

Measured motivation for the slate, on this corpus (99,767 passages, mean 333
chars, p99 727, 349,983 sentences): `fixed_size`'s 700-char window exceeds only
1.4% of passages, so 98.6% of its output is the unmodified passage; and
`semantic` at threshold 0.8 merged only 4.6% of adjacent sentence pairs, so it
emitted 338,544 chunks against 349,983 sentences -- a sentence splitter with an
embedding bill.
The two shipped strategies were the two extremes with nothing in between.
"""

import json
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from app.embeddings import cosine_similarity, embed_batch
from app.passages import Chunk, Passage, per_passage

FIXED_SIZE_CHARS = 700
FIXED_SIZE_OVERLAP = 0.2
RECURSIVE_TARGET_CHARS = 400
PARENT_CHILD_CHARS = 200
QUERY_GROUP_TARGET_CHARS = 1000

# The measured median of the adjacent-sentence cosine distribution on this
# corpus (scripts/measure_semantic_threshold.py, 3,734 pairs over 1,500
# passages, 2026-08-21): p5 0.034, p25 0.218, p50 0.407, p75 0.587, p90 0.723.
#
# A threshold at the median merges 51.1% of adjacent pairs, i.e. ~2.04
# sentences per chunk, which puts this strategy in the gap between
# whole_passage and sentence_window that the slate exists to fill.
#
# The shipped 0.8 merged 4.6% -- above even the p90 -- so it emitted ~1.05
# sentences per chunk. That predicts ~333k chunks against the 338,544 actually
# observed, which is the direct confirmation that the strategy was a sentence
# splitter paying an embedding call per sentence.
#
# Note the shipped docstring's premise was not the error. It argued 0.8 from
# unrelated sentences sitting at 0.3-0.6, and the measured median of 0.407 says
# that range describes adjacent pairs here too. The error was only that 0.8 is
# far above the whole distribution, so nothing merged.
SEMANTIC_THRESHOLD = 0.40

# Sentences are embedded across passages, not per passage: at ~3.5 sentences
# each, a per-passage embed call would pay model overhead 99,767 times.
SEMANTIC_PASSAGE_BATCH = 256

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as (start, end) offsets into `text`.

    Offsets rather than strings because chunks are spans of their parent, so a
    chunker needs positions it can address. Separator whitespace is excluded
    from both sides, which is also what keeps recursive chunks from beginning
    mid-whitespace.
    """
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _whole_passage(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    if not passage["text"]:
        return
    yield {"parent_id": passage_id, "start": 0, "end": len(passage["text"])}


def _fixed_size(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """Character windows with overlap. Kept exactly as shipped, including the
    mid-word cuts -- `recursive` is the corrected version, and the comparison
    between them is one of the report's findings."""
    text = passage["text"]
    if not text:
        return
    if len(text) <= FIXED_SIZE_CHARS:
        yield {"parent_id": passage_id, "start": 0, "end": len(text)}
        return
    step = int(FIXED_SIZE_CHARS * (1 - FIXED_SIZE_OVERLAP))
    start = 0
    while start < len(text):
        end = min(start + FIXED_SIZE_CHARS, len(text))
        yield {"parent_id": passage_id, "start": start, "end": end}
        if start + FIXED_SIZE_CHARS >= len(text):
            break
        start += step


def _recursive(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """Sentence-aligned packing with one sentence of overlap.

    This is fixed-size chunking done correctly. The shipped `fixed_size` slices
    at a raw character offset, so it cuts words and sentences in half; here the
    atomic unit is a sentence, and a sentence longer than the target is hard
    split only as a last resort so chunk size stays bounded either way.
    """
    text = passage["text"]
    spans = sentence_spans(text)
    if not spans:
        return

    units: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start <= RECURSIVE_TARGET_CHARS:
            units.append((start, end))
        else:
            for offset in range(start, end, RECURSIVE_TARGET_CHARS):
                units.append((offset, min(offset + RECURSIVE_TARGET_CHARS, end)))

    index = 0
    while index < len(units):
        start = units[index][0]
        last = index
        while (
            last + 1 < len(units)
            and units[last + 1][1] - start <= RECURSIVE_TARGET_CHARS
        ):
            last += 1
        yield {"parent_id": passage_id, "start": start, "end": units[last][1]}
        if last + 1 >= len(units):
            break
        # Overlap by one whole sentence. `last > index` guards the case where a
        # single unit already filled the target, which would otherwise not advance.
        index = last if last > index else index + 1


def _parent_child(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """Embed a small child for precision; return the whole parent for context."""
    text = passage["text"]
    if not text:
        return
    for start in range(0, len(text), PARENT_CHILD_CHARS):
        yield {
            "parent_id": passage_id,
            "start": start,
            "end": min(start + PARENT_CHILD_CHARS, len(text)),
            "ret_start": 0,
            "ret_end": len(text),
        }


def _sentence_window(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """Embed one sentence; return it with a neighbour on each side.

    The most precise embedding unit available without losing the context the
    sentence depends on. Windows clamp at the passage edges rather than
    wrapping.
    """
    spans = sentence_spans(passage["text"])
    for index, (start, end) in enumerate(spans):
        yield {
            "parent_id": passage_id,
            "start": start,
            "end": end,
            "ret_start": spans[max(0, index - 1)][0],
            "ret_end": spans[min(len(spans) - 1, index + 1)][1],
        }


def _query_aware(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """One vector per passage, embedded with the passage's own gold query.

    Free document expansion: the dataset ships the natural-language question
    each passage answers, so the doc2query pattern needs no generated queries.
    Directly targets the query/passage vocabulary gap MS MARCO exists to test.
    """
    if not passage["text"]:
        return
    yield {
        "parent_id": passage_id,
        "start": 0,
        "end": len(passage["text"]),
        "embed_query": True,
    }


HELDOUT_QUERY_IDS_PATH = "data/heldout_query_ids.json"


@cache
def _load_heldout_query_ids() -> frozenset[int]:
    """Query ids whose passages must NOT be enriched with their own query.

    Evaluating query_aware with the same query it was built from measures
    self-reference: the passage's vector literally contains the query being
    searched for. But the relevance labels for a query live in that query's own
    corpus row, so excluding the row at search time makes recall structurally
    impossible -- it removes the only passages that could count as hits.

    The honest control is therefore an index, not a filter: build one where the
    evaluated rows' passages carry no query, leave every other row enriched, and
    search it with those same queries. That is exactly the production case of a
    question the index was not built around.
    """
    path = Path(HELDOUT_QUERY_IDS_PATH)
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text()))


def _query_aware_heldout(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """query_aware's control. Identical, except the evaluated rows go in bare."""
    if not passage["text"]:
        return
    heldout = _load_heldout_query_ids()
    chunk: Chunk = {
        "parent_id": passage_id,
        "start": 0,
        "end": len(passage["text"]),
    }
    if passage["query_id"] not in heldout:
        chunk["embed_query"] = True
    yield chunk


whole_passage_chunker = per_passage(_whole_passage)
query_aware_heldout_chunker = per_passage(_query_aware_heldout)
fixed_size_chunker = per_passage(_fixed_size)
recursive_chunker = per_passage(_recursive)
parent_child_chunker = per_passage(_parent_child)
sentence_window_chunker = per_passage(_sentence_window)
query_aware_chunker = per_passage(_query_aware)


def query_group_chunker(passages: list[Passage]) -> Iterator[Chunk]:
    """Aggregates upward: every passage sharing a query_id becomes one document,
    then splits. Tests whether the retrievable unit should be larger than a
    passage rather than smaller.

    These chunks store their own text, because a concatenation of two passages
    is not a substring of either and so cannot be span-addressed. `parent_id`
    points at the group's first passage, which is what carries the relevance
    label for evaluation.
    """
    groups: dict[int, list[int]] = {}
    for passage_id, passage in enumerate(passages):
        groups.setdefault(passage["query_id"], []).append(passage_id)

    for passage_ids in groups.values():
        joined = " ".join(passages[i]["text"] for i in passage_ids).strip()
        if not joined:
            continue
        for start in range(0, len(joined), QUERY_GROUP_TARGET_CHARS):
            piece = joined[start : start + QUERY_GROUP_TARGET_CHARS]
            if piece.strip():
                yield {
                    "parent_id": passage_ids[0],
                    "start": 0,
                    "end": 0,
                    "text": piece,
                }


def semantic_chunker(
    passages: list[Passage], threshold: float = SEMANTIC_THRESHOLD
) -> Iterator[Chunk]:
    """Splits where adjacent sentences stop being similar.

    Sentences are embedded in batches spanning many passages: at ~3.5 sentences
    per passage, embedding per passage would pay per-call model overhead 99,767
    times, which is what made the original semantic build the slowest step in
    the pipeline.
    """
    for batch_start in range(0, len(passages), SEMANTIC_PASSAGE_BATCH):
        batch = passages[batch_start : batch_start + SEMANTIC_PASSAGE_BATCH]
        spans_per_passage = [sentence_spans(p["text"]) for p in batch]
        texts: list[str] = []
        for passage, spans in zip(batch, spans_per_passage):
            texts.extend(passage["text"][s:e] for s, e in spans)
        if not texts:
            continue
        vectors = embed_batch(texts)

        cursor = 0
        for offset, (passage, spans) in enumerate(zip(batch, spans_per_passage)):
            passage_id = batch_start + offset
            if not spans:
                cursor += len(spans)
                continue
            group_start = spans[0][0]
            group_end = spans[0][1]
            for i in range(1, len(spans)):
                similar = (
                    cosine_similarity(vectors[cursor + i - 1], vectors[cursor + i])
                    >= threshold
                )
                if similar:
                    group_end = spans[i][1]
                else:
                    yield {
                        "parent_id": passage_id,
                        "start": group_start,
                        "end": group_end,
                    }
                    group_start, group_end = spans[i]
            yield {"parent_id": passage_id, "start": group_start, "end": group_end}
            cursor += len(spans)
