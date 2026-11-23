"""Pydantic schemas."""

from pydantic import BaseModel, Field


class CorrectnessVerdict(BaseModel):
    """LLM-as-judge output for answers without sources.

    `reason` comes first so the model explains before it scores.
    """

    reason: str = Field(description="One or two sentences justifying the scores.")
    correctness: int = Field(ge=1, le=5, description="Agreement with the reference answer.")


class RagVerdict(CorrectnessVerdict):
    """LLM-as-judge output for answers generated from retrieved sources."""

    groundedness: int = Field(
        ge=1, le=5, description="How well the answer's claims are supported by the sources."
    )


class TopicLabel(BaseModel):
    """LLM-generated name for a topic cluster."""

    label: str = Field(min_length=3, max_length=60, description="2-5 word topic name.")
    description: str = Field(min_length=10, description="One sentence describing the topic.")
