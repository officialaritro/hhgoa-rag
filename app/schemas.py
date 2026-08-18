"""Pydantic input/output models for every pipeline stage boundary.

Later tasks import these shared shapes rather than redefining them --
see plan Task 1's Key Decisions in docs/plans/2026-08-18-voice-enabled-rag-pipeline.md.
"""

from pydantic import BaseModel


class STTInput(BaseModel):
    audio_bytes: bytes


class STTOutput(BaseModel):
    transcript: str


class RetrievedPassage(BaseModel):
    text: str
    source_passage: str
    is_selected: bool
    score: float


class RetrievalOutput(BaseModel):
    query: str
    strategy: str
    passages: list[RetrievedPassage]


class GenerationInput(BaseModel):
    query: str
    retrieval: RetrievalOutput


class GenerationOutput(BaseModel):
    answer: str
