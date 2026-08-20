"""Does each strategy actually do what its name claims?

This audit exists because the two shipped strategies did not. `fixed_size`'s
700-char window exceeded only 1.4% of passages, so it was a no-op wearing a
chunker's name; `semantic` at threshold 0.8 merged 4.6% of adjacent sentence
pairs, so it was `re.split` with an embedding bill. Both passed every test and
served traffic for days.

A strategy slate is only "vast" if the strategies are actually distinct. So each
one is checked against the specific claim its name makes, and a strategy that
degenerates into another is reported as such rather than counted.

No model is loaded: every check reads the built spans back off disk.

Run:
    .venv/bin/python -m scripts.audit_strategies
"""

import pickle
import statistics
from collections import Counter
from pathlib import Path

from app.chunkers import (
    FIXED_SIZE_CHARS,
    PARENT_CHILD_CHARS,
    QUERY_GROUP_TARGET_CHARS,
    RECURSIVE_TARGET_CHARS,
    sentence_spans,
)
from app.passages import load_passage_store, resolve_embed_text, resolve_text
from app.strategies import chunk_paths, dense_names


def load_rows(name: str) -> list[dict]:
    _, metadata_path = chunk_paths(name)
    return pickle.loads(Path(metadata_path).read_bytes())


def audit(name: str, rows: list[dict], passages: list[dict]) -> dict:
    per_parent = Counter(r["parent_id"] for r in rows)
    embed_lengths = []
    return_lengths = []
    differs = 0
    for row in rows:
        embed = resolve_embed_text(row, passages)
        returned = resolve_text(row, passages)
        embed_lengths.append(len(embed))
        return_lengths.append(len(returned))
        if embed != returned:
            differs += 1
    split_parents = sum(1 for c in per_parent.values() if c > 1)
    return {
        "chunks": len(rows),
        "parents_covered": len(per_parent),
        "chunks_per_parent": len(rows) / max(1, len(per_parent)),
        "parents_split_pct": 100 * split_parents / max(1, len(per_parent)),
        "embed_mean": statistics.mean(embed_lengths),
        "embed_max": max(embed_lengths),
        "return_mean": statistics.mean(return_lengths),
        "embed_differs_pct": 100 * differs / max(1, len(rows)),
    }


def verdicts(name: str, a: dict, rows: list[dict], passages: list[dict]) -> list[str]:
    """The claim each name makes, checked. Returns findings, worst first."""
    out = []
    if name == "whole_passage":
        out.append(
            f"CLAIM no split. {'HOLDS' if a['chunks_per_parent'] == 1.0 else 'FAILS'}: "
            f"{a['chunks_per_parent']:.3f} chunks/passage."
        )
    elif name == "fixed_size":
        out.append(
            f"CLAIM {FIXED_SIZE_CHARS}-char windows with overlap. DEGENERATE: only "
            f"{a['parents_split_pct']:.1f}% of passages are split at all. Kept "
            f"unchanged deliberately as the baseline being measured against."
        )
    elif name == "recursive":
        ends_on_terminator = 0
        checked = 0
        for row in rows[:20000]:
            text = resolve_embed_text(row, passages).rstrip()
            if not text:
                continue
            checked += 1
            parent_len = len(passages[row["parent_id"]]["text"])
            if text[-1] in ".!?" or row["end"] == parent_len:
                ends_on_terminator += 1
        out.append(
            f"CLAIM sentence-aligned to ~{RECURSIVE_TARGET_CHARS} chars. "
            f"{100 * ends_on_terminator / max(1, checked):.1f}% of chunks end on a "
            f"sentence terminator or the passage end; max chunk {a['embed_max']} chars."
        )
        out.append(
            f"  splits {a['parents_split_pct']:.1f}% of passages "
            f"({a['chunks_per_parent']:.2f} chunks/passage) -- vs fixed_size's "
            f"{100 * 1.4 / 100:.1f}%-ish, so it occupies the middle, but most "
            f"passages are still under the target and stay whole."
        )
    elif name == "semantic":
        by_parent: dict[int, list[dict]] = {}
        for row in rows:
            by_parent.setdefault(row["parent_id"], []).append(row)
        sentences_per_chunk = []
        for parent_id, parent_rows in list(by_parent.items())[:20000]:
            spans = sentence_spans(passages[parent_id]["text"])
            for row in parent_rows:
                covered = sum(
                    1 for s, e in spans if s >= row["start"] and e <= row["end"]
                )
                sentences_per_chunk.append(max(1, covered))
        counts = Counter(sentences_per_chunk)
        single = 100 * counts[1] / max(1, len(sentences_per_chunk))
        out.append(
            f"CLAIM merges similar adjacent sentences. "
            f"{statistics.mean(sentences_per_chunk):.2f} sentences/chunk; "
            f"{single:.1f}% of chunks are still a single sentence "
            f"(was ~95% at the shipped 0.8 threshold)."
        )
    elif name == "parent_child":
        out.append(
            f"CLAIM embed ~{PARENT_CHILD_CHARS}-char child, return whole parent. "
            f"{'HOLDS' if a['embed_max'] <= PARENT_CHILD_CHARS else 'FAILS'}: "
            f"embed mean {a['embed_mean']:.0f} max {a['embed_max']}, "
            f"return mean {a['return_mean']:.0f}. "
            f"embed differs from return on {a['embed_differs_pct']:.1f}% of chunks."
        )
    elif name == "sentence_window":
        out.append(
            f"CLAIM embed one sentence, return it +/-1. embed mean "
            f"{a['embed_mean']:.0f} chars, return mean {a['return_mean']:.0f} "
            f"({a['return_mean'] / max(1, a['embed_mean']):.2f}x wider). "
            f"Window is genuinely wider on {a['embed_differs_pct']:.1f}% of chunks; "
            f"the rest are single-sentence passages where it collapses to "
            f"whole_passage."
        )
    elif name == "query_aware":
        empty = sum(
            1 for r in rows[:50000] if not passages[r["parent_id"]]["query"].strip()
        )
        out.append(
            f"CLAIM embed query+passage, return passage bare. query present on "
            f"{100 * (1 - empty / max(1, min(len(rows), 50000))):.1f}% of chunks; "
            f"embed differs from return on {a['embed_differs_pct']:.1f}%. "
            f"embed mean {a['embed_mean']:.0f} vs return {a['return_mean']:.0f}."
        )
    elif name == "query_group":
        multi = 0
        for row in rows[:20000]:
            parent_text = passages[row["parent_id"]]["text"]
            if row.get("text") and row["text"] not in parent_text:
                multi += 1
        out.append(
            f"CLAIM concatenate passages sharing a query_id, split ~"
            f"{QUERY_GROUP_TARGET_CHARS}. {100 * multi / max(1, min(len(rows), 20000)):.1f}% "
            f"of chunks are NOT a substring of their nominal parent, i.e. they "
            f"genuinely cross passage boundaries. mean {a['return_mean']:.0f} chars, "
            f"max {a['embed_max']}."
        )
    return out


if __name__ == "__main__":
    passages = load_passage_store("data/passages.pkl")
    print(f"passages: {len(passages):,}\n")
    print(
        f"{'strategy':<17}{'chunks':>10}{'/passage':>10}{'split%':>8}"
        f"{'embed':>8}{'return':>8}{'e!=r%':>7}"
    )
    print("-" * 68)
    audits = {}
    for name in dense_names():
        rows = load_rows(name)
        a = audit(name, rows, passages)
        audits[name] = (a, rows)
        print(
            f"{name:<17}{a['chunks']:>10,}{a['chunks_per_parent']:>10.2f}"
            f"{a['parents_split_pct']:>8.1f}{a['embed_mean']:>8.0f}"
            f"{a['return_mean']:>8.0f}{a['embed_differs_pct']:>7.1f}"
        )
    print("\n" + "=" * 68)
    print("DOES EACH NAME EARN ITSELF?")
    print("=" * 68)
    for name in dense_names():
        a, rows = audits[name]
        print(f"\n{name}")
        for line in verdicts(name, a, rows, passages):
            print(f"  {line}")
