"""OpenAI Chat Completions call for grounded answer generation, wrapped in
the harness.

Model: gpt-5.4-mini-2026-03-17 (see .env.example / docs/ENVIRONMENT_VARIABLES.md)
-- a small/fast current OpenAI tier, a starting point for the latency
target. Confirm this is still current before deploying (Task 6 Key Decisions).
"""

import os

from app.harness import StageResult, run_stage
from app.schemas import GenerationOutput, RetrievalOutput

_DEFAULT_MODEL_ID = "gpt-5.4-mini-2026-03-17"
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
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_id = os.environ.get("OPENAI_MODEL_ID", _DEFAULT_MODEL_ID)
    completion = client.chat.completions.create(
        model=model_id,
        max_completion_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


def generate_answer(query: str, retrieval: RetrievalOutput) -> StageResult:
    prompt = _build_prompt(query, retrieval)
    result = run_stage(lambda: _call_model(prompt))
    if not result.ok:
        return result
    return StageResult(ok=True, value=GenerationOutput(answer=result.value))
