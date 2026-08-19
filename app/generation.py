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

_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the passages provided below. "
    "Do not use outside knowledge. If the passages do not contain enough "
    "information to answer, say so explicitly instead of guessing. "
    "This is a voice assistant: respond in plain, natural spoken-style "
    "prose -- no markdown, no headings, no bullet or numbered lists, no "
    "asterisks for emphasis."
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


def generate_answer(query: str, retrieval: RetrievalOutput) -> StageResult:
    prompt = _build_prompt(query, retrieval)
    result = run_stage(lambda: _call_model(prompt))
    if not result.ok:
        return result
    return StageResult(ok=True, value=GenerationOutput(answer=result.value))
