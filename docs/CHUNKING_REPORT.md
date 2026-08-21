# Chunking Strategy Comparison

Measured 2026-08-21 over **500 corpus queries** that carry at
least one `is_selected` relevance label, against **11 indices** built from the
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

**Fusing granularities is the only thing that beat a plain passage.** `fusion`
merges whole passages, 200-character children and single sentences by reciprocal
rank and reaches 0.854 recall@5, the best number here. The margin over
`whole_passage` is 0.6 points, which on 500 queries is three queries -- real but
small, and it costs 45 ms of search against 6.7 ms. Members were chosen for
diversity of failure mode rather than individual score: fusing the three
strategies that already tie each other would have added nothing.

**Lexical fusion is a negative result.** `hybrid` scores 0.740, ten points *below*
dense alone. BM25 by itself reaches only 0.558, and a weight sweep shows fusion
stops hurting only once the lexical ranking is weighted almost to zero:

| lexical weight | recall@5 |
|---|---|
| 1.0 (standard RRF) | 0.740 |
| 0.5 | 0.790 |
| 0.2 | 0.820 |
| 0.1 | 0.840 |
| 0.05 | 0.850 |
| *dense alone, no fusion* | *0.848* |

0.850 against 0.848 is one query inside the noise. `hybrid` therefore ships at
equal weight, because that is what hybrid retrieval means and the honest answer is
that it does not help here; tuning to 0.05 would produce a number that looks like
a tie while having switched lexical retrieval off.

The cause is the corpus rather than the implementation. These are
natural-language questions against short web passages, and an answer rarely
repeats its question's words -- exactly the vocabulary gap dense retrieval exists
to close. BM25's reputation on MS MARCO comes from official document ranking with
tuned stemming and stopword handling; neither is present here, and at a 0.558
starting point preprocessing would not close a 29-point gap.

**Recommended default: `whole_passage` or `fixed_size`.** Tied-best recall among
the single strategies, essentially tied-best MRR, the smallest competitive index,
6.7 ms search, and the fewest off-topic leaks (1 of 8). They are interchangeable,
which is the first finding restated. `fusion` is the quality ceiling if 45 ms of
search is acceptable, but 0.854 against 0.848 does not justify making it the
default for a voice demo where the dominant cost is elsewhere entirely.

## Retrieval quality

| strategy | axis | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| `fusion` | fusion | 0.414 | **0.854** | 0.958 | 0.596 | 0.683 |
| `query_aware_heldout` \* | enrichment | 0.416 | **0.850** | 0.962 | 0.599 | 0.685 |
| `whole_passage` | split | 0.410 | **0.848** | 0.960 | 0.591 | 0.679 |
| `fixed_size` | split | 0.410 | **0.848** | 0.958 | 0.592 | 0.679 |
| `recursive` | split | 0.414 | **0.848** | 0.950 | 0.593 | 0.679 |
| `parent_child` | unit | 0.412 | **0.822** | 0.942 | 0.580 | 0.667 |
| `semantic` | split | 0.348 | **0.818** | 0.918 | 0.535 | 0.628 |
| `sentence_window` | unit | 0.340 | **0.790** | 0.924 | 0.526 | 0.622 |
| `query_aware` | enrichment | 0.312 | **0.790** | 1.000 | 0.528 | 0.639 |
| `hybrid` | fusion | 0.340 | **0.740** | 0.898 | 0.510 | 0.602 |
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
| `fusion` | 669,084 (shared) | - | - | inherited | - | - |
| `query_aware_heldout` | 99,767 | 38.3 | 2.1 | 0.702 | 5.0% | 0 |
| `whole_passage` | 99,767 | 38.3 | 1.8 | 0.559 | 5.0% | 1 |
| `fixed_size` | 101,131 | 38.8 | 1.9 | 0.558 | 5.0% | 1 |
| `recursive` | 130,868 | 50.3 | 2.4 | 0.563 | 5.0% | 2 |
| `parent_child` | 219,334 | 84.2 | 5.9 | 0.554 | 5.0% | 3 |
| `semantic` | 219,100 | 84.1 | 4.0 | 0.557 | 5.0% | 4 |
| `sentence_window` | 349,983 | 134.4 | 9.4 | 0.574 | 5.0% | 3 |
| `query_aware` | 99,767 | 38.3 | 2.1 | 0.400 | 5.0% | 5 |
| `hybrid` | 99,767 (shared) | - | - | inherited | - | - |
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

Query embedding is strategy-independent: **P50 9.7 ms, P100 26.1 ms**.

| strategy | search P50 | search P100 | embed + search P50 |
|---|---|---|---|
| `query_group` | 2.62 ms | 2.87 ms | 12.3 ms |
| `whole_passage` | 6.73 ms | 10.43 ms | 16.4 ms |
| `query_aware_heldout` | 6.74 ms | 7.04 ms | 16.4 ms |
| `query_aware` | 6.74 ms | 7.05 ms | 16.4 ms |
| `fixed_size` | 6.83 ms | 7.22 ms | 16.5 ms |
| `recursive` | 8.81 ms | 9.26 ms | 18.5 ms |
| `hybrid` | 10.01 ms | 25.15 ms | 19.7 ms |
| `semantic` | 14.66 ms | 26.22 ms | 24.3 ms |
| `parent_child` | 14.68 ms | 15.02 ms | 24.4 ms |
| `sentence_window` | 23.34 ms | 33.47 ms | 33.0 ms |
| `fusion` | 45.01 ms | 59.94 ms | 54.7 ms |

## Reranking: the largest gain here, and not from chunking

The table above raises a question it cannot answer. `whole_passage` reaches
recall@10 of 0.960 but recall@1 of only 0.410 -- the relevant passage is
almost always retrieved and simply not ranked first. That is an **ordering**
problem, and no chunking strategy addresses it. A bi-encoder embeds query and
passage separately and never compares them directly; a cross-encoder reads
them together.

Measured over the same 500 queries, reranking `whole_passage`
candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`:

| candidate depth | recall@1 | recall@5 | MRR@10 | rerank P50 |
|---|---|---|---|---|
| *none (dense order)* | *0.410* | *0.848* | *0.591* | *0 ms* |
| 5 | 0.486 | **0.848** | 0.634 | 17.7 ms |
| 10 | 0.504 | **0.916** | 0.671 | 37.6 ms |
| 20 | 0.500 | **0.916** | 0.670 | 77.5 ms |
| 50 | 0.500 | **0.916** | 0.670 | 179.3 ms |

**recall@5 goes 0.848 to 0.916.** That is +6.8 points, well
outside the ~1.6pp standard error at this sample size -- and far larger than
anything the chunking slate achieved. `fusion`, the best chunking-side result,
reached 0.854, which is *inside* that error bar against plain dense retrieval.

**Depth 10 is chosen on both axes at once.** Quality peaks there
(0.504 recall@1 against 0.500 at both 20 and 50), so a deeper candidate pool
gives the model more chances to promote something wrong rather than more
chances to find the answer. And it is the cheapest: measured on CPU, which is
what the instance runs, 36 ms at depth 10 against 66 ms at 20 and 164 ms at
50, on top of ~70 ms already owned. Depth 50 alone would breach the 200 ms
target.

The honest reading of this report as a whole: the dataset does not reward
chunking, and the eight strategies establish that with evidence. What it
rewards is reranking, which the chunking numbers pointed at all along.
