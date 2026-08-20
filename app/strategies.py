"""The one place a chunking strategy is defined.

Strategy identity used to be duplicated across four modules -- `_CHUNKS_PATHS`
in scripts/build_index.py, `INDEX_PATHS` in app/retrieval.py, `_STRATEGIES` in
app/main.py, and `_DEFAULT_OFFTOPIC_THRESHOLDS` in app/guardrails.py. At two
strategies that is eight entries to keep in sync; at ten it is forty, and a
missed entry is exactly the defect class that put a threshold calibrated
against one index into production as another index's default.

Everything downstream -- build, retrieval, `/api/strategies`, threshold
lookup, evaluation -- derives from this registry. The chunking algorithms live
in app/chunkers.py; this module only says which ones exist and how they are
described.
"""

from dataclasses import dataclass, field
from typing import Literal

from app.chunkers import (
    fixed_size_chunker,
    parent_child_chunker,
    query_aware_chunker,
    query_aware_heldout_chunker,
    query_group_chunker,
    recursive_chunker,
    semantic_chunker,
    sentence_window_chunker,
    whole_passage_chunker,
)
from app.passages import Chunk, Chunker, per_passage

__all__ = [
    "Chunk",
    "Chunker",
    "Strategy",
    "UnknownStrategy",
    "chunk_paths",
    "dense_names",
    "get",
    "names",
    "per_passage",
    "served_names",
]


class UnknownStrategy(KeyError):
    """Raised instead of a bare KeyError so the message can list what is valid.

    A dict lookup failure gave no way to distinguish a typo from a strategy
    that exists in the slate but whose index was never built.
    """


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: Literal["dense", "hybrid", "fusion"]
    axis: Literal["split", "unit", "enrichment", "aggregation", "fusion"]
    description: str
    chunker: Chunker | None = None
    members: tuple[str, ...] = field(default_factory=tuple)
    # True when chunking itself calls the embedding model. Only semantic does,
    # and it is why semantic alone cannot report a chunk total before the build:
    # counting its chunks means running the boundary detection, which is the
    # expensive half of its work. Everything else chunks with string operations,
    # so the build counts first and shows a real percentage and ETA.
    chunking_embeds: bool = False
    # False for measurement controls: indices built to answer a question about
    # another strategy, which should not be offered as something to retrieve
    # from. They are still built, calibrated and evaluated.
    served: bool = True


_REGISTRY: dict[str, Strategy] = {
    "whole_passage": Strategy(
        name="whole_passage",
        kind="dense",
        axis="split",
        description=(
            "No split; one vector per passage. The control: fixed_size's window "
            "exceeds only 1.4% of passages, so if these two score alike, "
            "fixed-size chunking is doing nothing on this corpus."
        ),
        chunker=whole_passage_chunker,
    ),
    "fixed_size": Strategy(
        name="fixed_size",
        kind="dense",
        axis="split",
        description="700-character windows, 20% overlap, cutting mid-word.",
        chunker=fixed_size_chunker,
    ),
    "recursive": Strategy(
        name="recursive",
        kind="dense",
        axis="split",
        description=(
            "Sentence-aligned packing to ~400 characters with one sentence of "
            "overlap. Fixed-size chunking done without cutting words in half."
        ),
        chunker=recursive_chunker,
    ),
    "semantic": Strategy(
        name="semantic",
        kind="dense",
        axis="split",
        description=(
            "Splits where adjacent sentences stop being similar, at a retuned "
            "threshold. At the shipped 0.8 it merged almost nothing."
        ),
        chunker=semantic_chunker,
        chunking_embeds=True,
    ),
    "parent_child": Strategy(
        name="parent_child",
        kind="dense",
        axis="unit",
        description=(
            "Embeds a ~200-character child for precision, returns the whole "
            "parent passage for context."
        ),
        chunker=parent_child_chunker,
    ),
    "sentence_window": Strategy(
        name="sentence_window",
        kind="dense",
        axis="unit",
        description=(
            "Embeds a single sentence, returns it with one neighbour on each "
            "side. The most precise embedding unit that keeps its context."
        ),
        chunker=sentence_window_chunker,
    ),
    "query_aware": Strategy(
        name="query_aware",
        kind="dense",
        axis="enrichment",
        description=(
            "Embeds the passage with its own gold query prepended, returns the "
            "passage bare. Document expansion with no generated queries, since "
            "the dataset already ships the question each passage answers."
        ),
        chunker=query_aware_chunker,
    ),
    "query_aware_heldout": Strategy(
        name="query_aware_heldout",
        kind="dense",
        axis="enrichment",
        description=(
            "query_aware's control: identical, except the evaluated rows' "
            "passages are embedded without their own query. Measures whether "
            "query enrichment generalises to a question the index was not "
            "built around, which searching the enriched index cannot."
        ),
        chunker=query_aware_heldout_chunker,
        # A control, not a product. Its own off-topic threshold is not even
        # meaningful: calibration samples different rows than the evaluation
        # holds out, so most calibration queries still match their own enriched
        # passages. Only its recall against the held-out queries is valid.
        served=False,
    ),
    "query_group": Strategy(
        name="query_group",
        kind="dense",
        axis="aggregation",
        description=(
            "Concatenates every passage sharing a query_id into one document, "
            "then splits at ~1000 characters. Aggregates upward instead of down."
        ),
        chunker=query_group_chunker,
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


def dense_names() -> tuple[str, ...]:
    """Strategies backed by their own index, i.e. the ones a build produces.

    Includes measurement controls, which are built and evaluated but not served.
    Composed kinds (hybrid, fusion) are served by combining these at request
    time and have nothing of their own to build.
    """
    return tuple(n for n, s in _REGISTRY.items() if s.kind == "dense")


def served_names() -> tuple[str, ...]:
    """Strategies a request may retrieve from. Excludes measurement controls."""
    return tuple(n for n, s in _REGISTRY.items() if s.served)


def chunk_paths(name: str) -> tuple[str, str]:
    """Index and chunk-metadata paths for a strategy.

    Derived from the name rather than looked up, so a path can be named before
    its strategy is registered -- the build writes the artifacts, registration
    is what makes them servable.
    """
    return (f"data/index_{name}.faiss", f"data/chunks_{name}.pkl")
