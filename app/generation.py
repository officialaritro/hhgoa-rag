"""Anthropic Messages API call for grounded answer generation, wrapped in
the harness.

Model: claude-3-5-haiku-20241022 (see .env.example / docs/ENVIRONMENT_VARIABLES.md)
-- a small/fast tier, a starting point for the latency target.
"""

import os

from app.harness import StageResult, run_stage
from app.schemas import GenerationOutput, RetrievalOutput

_DEFAULT_MODEL_ID = "claude-3-5-haiku-20241022"
_MAX_TOKENS = 512

_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the passages provided below. "
    "Do not use outside knowledge. If the passages do not contain enough "
    "information to answer, say so explicitly instead of guessing."
)


def _build_prompt(query: str, retrieval: RetrievalOutput) -> str:
    passages_block = "\n\n".join(
        f"Passage {i + 1}: {p.text}" for i, p in enumerate(retrieval.passages)
    )
    return f"Question: {query}\n\nPassages:\n{passages_block}"


def _call_model(prompt: str) -> str:
    from anthropic import Anthropic

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
    return message.content[0].text


def generate_answer(query: str, retrieval: RetrievalOutput) -> StageResult:
    prompt = _build_prompt(query, retrieval)
    result = run_stage(lambda: _call_model(prompt))
    if not result.ok:
        return result
    return StageResult(ok=True, value=GenerationOutput(answer=result.value))
