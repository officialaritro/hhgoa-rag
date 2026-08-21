"""Claim-level groundedness scoring.

Replaces a whole-answer cosine against the concatenated context. That had two
defects, one of them a correctness bug:

**Truncation.** `all-MiniLM-L6-v2` caps at 256 tokens. The median k=5 context
measures 334 tokens and 94% of real contexts exceed the cap, so the guard was
comparing the answer against a silently shortened context on almost every
request -- up to two thirds discarded in the worst case measured. Passages are
now embedded individually; one passage averages 333 characters, about 80 tokens,
so nothing is cut.

**Averaging.** One fabricated claim inside four sound sentences barely moves a
whole-answer cosine. Support is now computed per answer sentence.

A literal number check runs alongside, because the failure that matters most is
invisible to an embedding: "founded in 1987" against a context saying 1897 is
one digit apart and semantically identical.
"""

import re

import numpy as np

from app.embeddings import embed_batch
from app.schemas import RetrievalOutput

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Digits with optional thousands separators and decimals. Bare small integers
# are excluded below rather than here, so the pattern stays readable.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Integers at or below this are not treated as factual claims. Models write
# "3 signs" and "1." as prose and enumeration, not as figures lifted from a
# passage, and flagging those would refuse most list-shaped answers. Anything
# with a decimal point or a thousands separator is always checked.
_TRIVIAL_INTEGER = 20


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def _normalise_number(raw: str) -> str:
    return raw.replace(",", "")


def unsupported_numbers(answer: str, context: str) -> list[str]:
    """Numbers asserted in the answer that do not appear in the context.

    Compared after stripping thousands separators, so "1,234" in an answer
    matches "1234" in a passage -- treating those as different would refuse
    correct answers.
    """
    context_numbers = {_normalise_number(m) for m in _NUMBER.findall(context)}
    missing: list[str] = []
    for raw in _NUMBER.findall(answer):
        normalised = _normalise_number(raw)
        if normalised in context_numbers:
            continue
        try:
            value = float(normalised)
        except ValueError:
            continue
        if "." not in normalised and "," not in raw and value <= _TRIVIAL_INTEGER:
            continue
        missing.append(raw)
    return missing


def sentence_support(answer: str, retrieval: RetrievalOutput) -> list[float]:
    """Best per-passage cosine for each sentence of the answer.

    Two batched embed calls total -- one for the answer's sentences, one for the
    passages -- because this runs on every request and per-item calls would pay
    model overhead once per sentence and once per passage.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return []
    if not retrieval.passages:
        # Nothing to be grounded in. A refusal, not a crash.
        return [0.0] * len(sentences)

    # Reuse any vector retrieval already computed. For a chunk whose embedded
    # span is its returned span, the FAISS index holds exactly the vector of the
    # text being returned, so re-embedding it is pure waste -- and this guard was
    # the largest cost the pipeline owns.
    supplied: dict[int, list[float]] = {}
    to_embed: list[str] = []
    for position, passage in enumerate(retrieval.passages):
        if passage.vector:
            supplied[position] = passage.vector
        else:
            to_embed.append(passage.text)

    embedded = np.asarray(embed_batch(sentences + to_embed), dtype="float32")
    sentence_vectors = embedded[: len(sentences)]
    fresh = list(embedded[len(sentences) :])

    # A vector of the wrong width means a stale artifact -- a different embedding
    # model, most likely -- and trusting it would corrupt every decision
    # silently. Embed that passage instead.
    width = sentence_vectors.shape[1]
    mismatched = [i for i, v in supplied.items() if len(v) != width]
    if mismatched:
        extra = np.asarray(
            embed_batch([retrieval.passages[i].text for i in mismatched]),
            dtype="float32",
        )
        for i, vector in zip(mismatched, extra):
            supplied[i] = list(vector)

    rows_out: list[np.ndarray] = []
    fresh_iter = iter(fresh)
    for position in range(len(retrieval.passages)):
        if position in supplied:
            rows_out.append(np.asarray(supplied[position], dtype="float32"))
        else:
            rows_out.append(next(fresh_iter))
    passage_vectors = np.vstack(rows_out)

    # Both sides are L2-normalised by embed_batch, so the dot product is cosine.
    # Max over passages: a claim backed by one passage is grounded even when the
    # other four are about something else, which averaging would punish.
    similarities = sentence_vectors @ passage_vectors.T
    return [float(row.max()) for row in similarities]


# Measured on 40 real generated answers (scripts/calibrate_groundedness.py,
# 2026-08-21), aggregating per-sentence support by mean:
#
#   grounded (own context)     min 0.524  median 0.678  max 0.942
#   ungrounded (other context) min -0.060 median 0.051  max 0.271
#
# The distributions separate cleanly, gap (0.271, 0.524), midpoint 0.397. At a
# threshold giving zero false refusals, mean catches 100% of ungrounded answers;
# min catches 89.5% and p25 86.8%, because both punish the framing and
# transitional sentences real prose contains. `max` also reaches 100% but is
# rejected on principle: it passes an answer with one supported sentence and
# four fabrications, which is the failure this guard exists for.
#
# The literal number check covers what mean cannot -- a single fabricated figure
# barely moves an average. It flagged 0 of 40 genuinely grounded answers.
#
# Decline sentinels are excluded from this calibration because they are handled
# earlier in the pipeline; a bare INSUFFICIENT_CONTEXT scores near zero against
# any context and would drag the grounded distribution down artificially.
MEASURED_THRESHOLD = 0.40

# CORRECTION, measured 2026-08-21. An earlier commit message claimed this guard
# took 110.7ms to 53.3ms, "a 2.1x reduction". That compared the old guard on the
# EC2 instance against this one on an M1 with MPS -- different hardware, so not a
# comparison at all.
#
# On matched hardware over 25 real answers, this guard is SLOWER:
#
#   old (2 single embeds, context concatenated)  P50 31.2 ms
#   new (1 batched call, passages separate)      P50 38.9 ms   1.25x slower
#
# Which is the expected direction once stated plainly: the old guard truncated
# at 256 tokens and so fed the model less work. Processing the whole context
# costs more than processing three quarters of it.
#
# So this change buys correctness, not speed, and the claim that it "funds the
# reranker" was wrong. What it buys is real and measured -- clean separation
# where the distributions previously overlapped, 100% of ungrounded answers
# caught at zero false refusals, and fabricated figures detected at all -- but
# the latency budget has to absorb it rather than being helped by it.
GUARD_LATENCY_IS_A_COST_NOT_A_SAVING = True
