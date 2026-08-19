"""Shared call wrapper used by every external call (speech-to-text, generation):
retry once on failure, then return a typed error result instead of raising.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class StageResult(BaseModel):
    ok: bool
    value: Any = None
    error: str | None = None


def run_stage[T](fn: Callable[[], T], retries: int = 1) -> StageResult:
    last_error: str | None = None
    for _ in range(retries + 1):
        try:
            return StageResult(ok=True, value=fn())
        except Exception as exc:  # noqa: BLE001 -- intentional: convert ANY external-call failure into a StageResult, never raise
            import traceback
            traceback.print_exc()
            last_error = str(exc)
    return StageResult(ok=False, value=None, error=last_error)
