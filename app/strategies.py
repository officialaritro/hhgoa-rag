"""The one place a chunking strategy is defined.

Strategy identity used to be duplicated across four modules -- `_CHUNKS_PATHS`
in scripts/build_index.py, `INDEX_PATHS` in app/retrieval.py, `_STRATEGIES` in
app/main.py, and `_DEFAULT_OFFTOPIC_THRESHOLDS` in app/guardrails.py. At two
strategies that is eight entries to keep in sync; at ten it is forty, and a
missed entry is exactly the defect class that put a threshold calibrated
against one index into production as another index's default.

Everything downstream -- build, retrieval, `/api/strategies`, threshold
lookup, evaluation -- derives from this registry.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from app.passages import Passage


class Chunk(TypedDict, total=False):
    """A span of one parent passage, or (when `text` is present) a chunk that is
    not a contiguous substring of any single parent."""

    parent_id: int
    start: int
    end: int
    text: str


# A chunker sees the whole store, not one passage at a time, because not every
# strategy is per-passage: query_group deliberately aggregates across all the
# passages sharing a query_id. Per-passage strategies wrap themselves in
# `per_passage` rather than each re-implementing the loop.
Chunker = Callable[[list[Passage]], Iterator[Chunk]]


class UnknownStrategy(KeyError):
    """Raised instead of a bare KeyError so the message can list what is valid.

    A dict lookup failure gave no way to distinguish a typo from a strategy
    that exists in the slate but whose index was never built.
    """


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: Literal["dense", "hybrid", "fusion"]
    description: str
    chunker: Chunker | None = None
    members: tuple[str, ...] = field(default_factory=tuple)


def per_passage(
    fn: Callable[[Passage, int], Iterable[Chunk]],
) -> Chunker:
    """Adapts a per-passage chunker into a corpus-level one, passing each
    passage's id so the yielded spans can address their parent."""

    def chunker(passages: list[Passage]) -> Iterator[Chunk]:
        for passage_id, passage in enumerate(passages):
            yield from fn(passage, passage_id)

    return chunker


def _whole_passage(passage: Passage, passage_id: int) -> Iterator[Chunk]:
    """No split at all: one vector per passage.

    This is the control strategy. Measured on this corpus, `fixed_size`'s
    700-char window only exceeds 1.4% of passages (p99 is 727 chars), so it
    produces 101,131 chunks from 99,767 passages -- 98.6% of its output is the
    unmodified passage. This strategy makes that comparison explicit: if the
    two score the same, fixed-size chunking is doing nothing on this corpus.
    """
    yield {"parent_id": passage_id, "start": 0, "end": len(passage["text"])}


_REGISTRY: dict[str, Strategy] = {
    "whole_passage": Strategy(
        name="whole_passage",
        kind="dense",
        description="No split; one vector per passage. The control strategy.",
        chunker=per_passage(_whole_passage),
    ),
}


def get(name: str) -> Strategy:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownStrategy(
            f"unknown strategy {name!r}; registered: {', '.join(sorted(_REGISTRY))}"
        ) from None


def names() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def chunk_paths(name: str) -> tuple[str, str]:
    """Index and chunk-metadata paths for a strategy.

    Derived from the name rather than looked up, so a path can be named before
    its strategy is registered -- the build writes the artifacts, registration
    is what makes them servable.
    """
    return (f"data/index_{name}.faiss", f"data/chunks_{name}.pkl")
