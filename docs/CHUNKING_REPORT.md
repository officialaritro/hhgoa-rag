# Chunking Strategy Comparison

Measured 2026-08-21 over **500 corpus queries** that carry at
least one `is_selected` relevance label, against **9 indices** built from the
same 99,767-passage corpus and the same embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, int8 scalar-quantized FAISS).

Retrieval over-fetches 4x and collapses candidates by parent passage before
truncating, which is exactly what `app/retrieval.py` serves, so these numbers
describe the shipped pipeline rather than a variant of it.

## What the numbers say

**Chunking does nothing on this corpus.** `whole_passage`, `fixed_size` and
`recursive` score an identical 0.848 recall@5, and their MRR@10 differs only in the
third decimal (0.591 / 0.592 / 0.593). Those are three genuinely different
splitters -- no split, 700-char windows cutting mid-word, and sentence-aligned
400-char packing -- and they are indistinguishable. The corpus explains it: mean
passage length is 333 characters and p99 is 727, so a passage is already the
atomic retrievable unit and there is nothing useful to cut.

**Splitting finer makes it worse, monotonically.** No split 0.848, 200-char
children 0.822, ~1.6-sentence semantic groups 0.818, single sentences 0.790. Every
increase in granularity costs recall.

**But the driver is topical purity, not chunk size.** `query_group` has the
*largest* chunks (876 characters mean) and by far the worst recall, 0.422. It
concatenates every passage answering the same query, so each vector is a blur of
ten loosely related topics. One coherent passage is the optimum: sub-splitting
loses context, super-merging loses focus.

**Query enrichment shows no reliable gain, and its apparent gain was an artifact.**
See the note below the table -- this was the single most misleading result in the
set, and it was wrong in both directions before being pinned down.

**Recommended default: `whole_passage` or `fixed_size`.** Tied-best recall,
essentially tied-best MRR, the smallest index of the competitive strategies, fast
search, and the fewest off-topic leaks (1 of 8). They are interchangeable, which is
itself the first finding restated.

## Retrieval quality

| strategy | axis | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| `query_aware_heldout` \* | enrichment | 0.416 | **0.850** | 0.962 | 0.599 | 0.685 |
| `whole_passage` | split | 0.410 | **0.848** | 0.960 | 0.591 | 0.679 |
| `fixed_size` | split | 0.410 | **0.848** | 0.958 | 0.592 | 0.679 |
| `recursive` | split | 0.414 | **0.848** | 0.950 | 0.593 | 0.679 |
| `parent_child` | unit | 0.412 | **0.822** | 0.942 | 0.580 | 0.667 |
| `semantic` | split | 0.348 | **0.818** | 0.918 | 0.535 | 0.628 |
| `sentence_window` | unit | 0.340 | **0.790** | 0.924 | 0.526 | 0.622 |
| `query_aware` | enrichment | 0.312 | **0.790** | 1.000 | 0.528 | 0.639 |
| `query_group` | aggregation | 0.400 | **0.422** | 0.426 | 0.411 | 0.415 |

### The `query_aware` asterisk, and why neither of its numbers is clean

`query_aware` embeds each passage with the gold query that passage answers -- free
document expansion, since the dataset ships the query. It was the strategy this
work expected most from. It does not deliver, and establishing that took three
corrections.

The tell is in the table: **`query_aware` scores recall@10 of exactly 1.000.**
Every one of the 500 queries finds its own enriched passage within ten results,
because that passage's vector literally contains the query being searched for.
That is self-reference, not retrieval. Its recall@5 of 0.790 is *below*
`whole_passage`, because enriching all 99,767 passages makes them uniformly
query-shaped and therefore harder to tell apart -- the noise costs more than the
self-match gains.

`query_aware_heldout` is a control index: identical, except the 500 evaluated
rows' passages are embedded bare while every other row stays enriched. It scores
0.850, marginally the best in the table. **That number is also an artifact**, in
the opposite direction: those rows' passages are clean while all their competitors
carry query noise, which is an advantage production would never grant.

So the honest reading is a bracket, 0.790 to 0.850, with `whole_passage`'s 0.848
sitting inside it. **Query enrichment shows no measurable gain on this corpus.**
A clean measurement would need queries that never entered the index at all, which
this dataset does not provide.

The same self-reference inflated its guardrail calibration: measured naively the
off-topic threshold came out at 0.722 with zero leaks of eight probes, apparently
the only strategy whose in-corpus and off-topic score distributions separate
cleanly. Held out it is 0.400 with five leaks, the worst of the slate. Shipping the
naive threshold would have refused most real traffic.

## Cost and calibration

| strategy | chunks | index MB | metadata MB | off-topic threshold | false refusal | leaks (of 8) |
|---|---|---|---|---|---|---|
| `query_aware_heldout` | 99,767 | 38.3 | 2.1 | 0.702 | 5.0% | 0 |
| `whole_passage` | 99,767 | 38.3 | 1.8 | 0.559 | 5.0% | 1 |
| `fixed_size` | 101,131 | 38.8 | 1.9 | 0.558 | 5.0% | 1 |
| `recursive` | 130,868 | 50.3 | 2.4 | 0.563 | 5.0% | 2 |
| `parent_child` | 219,334 | 84.2 | 5.9 | 0.554 | 5.0% | 3 |
| `semantic` | 219,100 | 84.1 | 4.0 | 0.557 | 5.0% | 4 |
| `sentence_window` | 349,983 | 134.4 | 9.4 | 0.574 | 5.0% | 3 |
| `query_aware` | 99,767 | 38.3 | 2.1 | 0.400 | 5.0% | 5 |
| `query_group` | 37,984 | 14.6 | 34.3 | 0.499 | 5.0% | 2 |

**1,357,701 vectors across 9 indices, 585 MB total.**
Chunk metadata is span-addressed -- `(parent_id, start, end)` into one shared
passage store -- rather than each chunk carrying its own copy of its parent text.
That is 18.4 bytes per chunk against 763.9 measured on the previous scheme, a 41x
reduction, and it is what makes nine indices cost roughly what two did.

## Latency

Decomposed rather than summed, because only one half scales with index size, and
because embedding and FAISS cannot share a process on this build machine (both
link their own OpenMP runtime; co-loading them segfaults once Metal is in use).

Query embedding is strategy-independent: **P50 9.9 ms, P100 26.5 ms**.

| strategy | search P50 | search P100 | embed + search P50 |
|---|---|---|---|
| `query_group` | 2.62 ms | 2.93 ms | 12.6 ms |
| `whole_passage` | 6.75 ms | 10.27 ms | 16.7 ms |
| `query_aware_heldout` | 6.75 ms | 7.19 ms | 16.7 ms |
| `query_aware` | 6.75 ms | 7.15 ms | 16.7 ms |
| `fixed_size` | 6.84 ms | 7.26 ms | 16.8 ms |
| `recursive` | 8.81 ms | 9.26 ms | 18.7 ms |
| `semantic` | 14.68 ms | 15.27 ms | 24.6 ms |
| `parent_child` | 14.70 ms | 15.17 ms | 24.6 ms |
| `sentence_window` | 23.38 ms | 24.55 ms | 33.3 ms |
