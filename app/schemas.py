"""Pydantic input/output models for every pipeline stage boundary.

Every stage imports these shared shapes rather than redefining them, so a
change to what retrieval returns is a change every consumer sees.
"""

from pydantic import BaseModel, Field


class STTInput(BaseModel):
    audio_bytes: bytes


class STTOutput(BaseModel):
    transcript: str


class RetrievedPassage(BaseModel):
    text: str
    source_passage: str
    is_selected: bool
    score: float
    # The embedding of `text`, when retrieval already had it. Present only for
    # chunks whose embedded span IS their returned span, where the vector in the
    # FAISS index is exactly the vector of the text being returned. The
    # groundedness guard reuses it instead of re-embedding what was just
    # computed -- that guard measured 170ms on the instance and pushed boundary A
    # past its 200ms target. Excluded from the API response, which has no use
    # for 384 floats per passage.
    vector: list[float] | None = Field(default=None, exclude=True)


class RetrievalOutput(BaseModel):
    query: str
    strategy: str
    passages: list[RetrievedPassage]


class GenerationInput(BaseModel):
    query: str
    retrieval: RetrievalOutput


class GenerationOutput(BaseModel):
    answer: str
    # True when the model reported it could not answer from the passages. This
    # is carried as a flag rather than inferred from the answer text later,
    # because a decline is indistinguishable from an answer by similarity: the
    # model quotes the passages to explain itself, so its refusal scores as
    # grounded (measured 0.533-0.795 against context, threshold 0.40).
    insufficient_context: bool = False
