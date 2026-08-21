"""Anthropic Messages API call for grounded answer generation, wrapped in
the harness.

Model: claude-haiku-4-5-20251001, overridable via CLAUDE_MODEL_ID (see
.env.example / docs/ENVIRONMENT_VARIABLES.md) -- a small/fast tier, a
starting point for the latency target.
"""

import os

from app.harness import StageResult, run_stage
from app.schemas import GenerationOutput, RetrievalOutput

_MAX_TOKENS = 512

# Sentinel the model returns instead of prose when it cannot answer.
#
# Asking it to "say so explicitly" was not enough. The guardrail after
# generation scores the answer against the retrieved context, and a spoken
# decline *quotes the passages* to explain itself, so it scores as well as a
# real answer: measured on the live indices, declines scored 0.533 (fixed_size)
# and 0.795 (semantic) against a 0.40 threshold. Every one passed as grounded
# and reached the user as an answer carrying no refusal reason.
#
# A fixed token is checkable without a second LLM call (plan Global
# Constraints) and cannot be mistaken for an answer the way prose can.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# Brevity is not a style preference here, it is two measured wins.
#
# Latency: generation is the largest cost in the pipeline (2.1s to a complete
# answer against 650ms to first token), and output tokens dominate that. Measured
# before this instruction, answers ran to a median of 79 words -- roughly 32
# seconds of synthesised speech for a question a person asked out loud.
#
# Groundedness: 35% of measured answers opened with a framing phrase ("Based on
# the passages provided...", "According to..."). Those sentences correctly match
# no passage, so they drag down per-sentence support in the groundedness guard
# and cost real answers a refusal they do not deserve.
_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the passages provided below. "
    "Do not use outside knowledge. If the passages do not contain enough "
    f"information to answer, reply with exactly {INSUFFICIENT_CONTEXT} and "
    "nothing else -- no explanation, and do not quote the passages. "
    "Otherwise this is a voice assistant, so the answer is spoken aloud: "
    "answer in at most three short sentences, and stop. "
    "State the answer directly -- never preface it with 'based on the "
    "passages', 'according to', or any mention of the passages at all. "
    "Plain spoken-style prose only: no markdown, no headings, no bullet or "
    "numbered lists, no asterisks."
)


def _build_prompt(query: str, retrieval: RetrievalOutput) -> str:
    passages_block = "\n\n".join(
        f"Passage {i + 1}: {p.text}" for i, p in enumerate(retrieval.passages)
    )
    return f"Question: {query}\n\nPassages:\n{passages_block}"


def _call_model(prompt: str) -> str:
    from anthropic import Anthropic
    from anthropic.types import TextBlock

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_id = os.environ.get("CLAUDE_MODEL_ID", "claude-haiku-4-5-20251001")
    if model_id == "claude-haiku-4-5":
        model_id = "claude-haiku-4-5-20251001"

    message = client.messages.create(
        model=model_id,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    # No tools/thinking on this request, so content[0] should always be
    # TextBlock -- but the SDK's return type is a broader union (tool use,
    # thinking, etc.), so basedpyright correctly flags an unchecked .text
    # access. Guard it explicitly rather than let a mismatched block type
    # raise an opaque AttributeError inside run_stage's except-Exception.
    first_block = message.content[0]
    if not isinstance(first_block, TextBlock):
        raise TypeError(
            f"expected a text response block, got {type(first_block).__name__}"
        )
    return first_block.text


def _is_decline(answer: str) -> bool:
    """Whether the model returned the decline sentinel rather than an answer.

    Matches only at the start of the reply, after stripping whitespace and
    trailing punctuation. Substring matching anywhere would let a real answer
    that merely mentions the sentinel refuse a question -- a false positive
    here refuses a real question, the failure mode the off-topic threshold
    already shipped once.
    """
    return answer.strip().rstrip(".!").upper().startswith(INSUFFICIENT_CONTEXT)


def generate_answer(query: str, retrieval: RetrievalOutput) -> StageResult:
    prompt = _build_prompt(query, retrieval)
    result = run_stage(lambda: _call_model(prompt))
    if not result.ok:
        return result
    answer = result.value
    return StageResult(
        ok=True,
        value=GenerationOutput(answer=answer, insufficient_context=_is_decline(answer)),
    )
