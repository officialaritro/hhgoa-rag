"""Anthropic Claude call for grounded answer generation, wrapped in the harness.

Model: claude-haiku-4-5 (see .env.example / docs/ENVIRONMENT_VARIABLES.md) --
the fastest/smallest current Claude tier, a starting point for the latency
target. Confirm this is still current before deploying (Task 6 Key Decisions).
"""

import os

from app.harness import StageResult, run_stage
from app.schemas import GenerationOutput, RetrievalOutput

_DEFAULT_MODEL_ID = "claude-haiku-4-5"
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
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_id = os.environ.get("CLAUDE_MODEL_ID", _DEFAULT_MODEL_ID)
    response = client.messages.create(
        model=model_id,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_answer(query: str, retrieval: RetrievalOutput) -> StageResult:
    prompt = _build_prompt(query, retrieval)
    result = run_stage(lambda: _call_model(prompt))
    if not result.ok:
        return result
    return StageResult(ok=True, value=GenerationOutput(answer=result.value))
