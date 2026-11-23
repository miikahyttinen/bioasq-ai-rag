"""Pydantic schemas: LLM structured outputs and API requests/responses."""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core.retrieve import Strategy


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


# --- API --------------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)
    strategy: Strategy = "hybrid"
    k: int = Field(default=settings.default_k, ge=1, le=settings.max_chat_k)
    model: str | None = None
    think: bool | None = None


class Source(BaseModel):
    n: int = Field(description="Citation number used in answers, starting at 1.")
    chunk_id: str
    pmid: int
    text: str
    score: float
    url: str
    cluster_id: int | None = None
    cluster_label: str | None = None


class SearchResponse(BaseModel):
    query: str
    strategy: Strategy
    k: int
    seconds: float
    results: list[Source]


class Passage(BaseModel):
    pmid: int
    text: str
    url: str
    cluster_id: int | None = None
    cluster_label: str | None = None
    nearest_cluster_id: int | None = None
    nearest_cluster_label: str | None = None


class QuestionSummary(BaseModel):
    qid: int
    question: str
    n_gold: int
    has_usable: bool


class QuestionPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[QuestionSummary]


class QuestionDetail(QuestionSummary):
    answer: str
    relevant_ids: list[int]
    usable_relevant_ids: list[int]
    retrieval: dict | None = Field(
        default=None, description="Stored top-20 PMIDs and metrics per strategy."
    )
    evaluation: dict | None = Field(
        default=None, description="Stored answer-evaluation record, if this was sampled."
    )


class RunSource(Source):
    in_gold: bool


class QuestionRun(BaseModel):
    qid: int
    question: str
    gold_answer: str
    gold_ids: list[int]
    strategy: Strategy
    k: int
    sources: list[RunSource]
    hits: int
    recall: float | None
    answer: str
    model: str
    seconds: float


class Health(BaseModel):
    status: Literal["ok", "degraded", "down"]
    ollama: bool
    chat_model: str
    models: dict[str, bool]
    collection_count: int | None
    bm25_index: bool
    topics: bool
    eval_results: bool
    problems: list[str]
