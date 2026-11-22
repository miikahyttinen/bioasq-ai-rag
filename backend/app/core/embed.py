"""Embedding through Ollama's /api/embed, with batching."""

from collections.abc import Iterator, Sequence
from functools import lru_cache

import ollama

from app.config import settings
from app.core.prompts import embedding_query


@lru_cache
def get_client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def batched[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_texts(texts: Sequence[str], keep_alive: str | int | None = None) -> list[list[float]]:
    """Embed texts as-is (documents). Vectors come back L2-normalized."""
    if not texts:
        return []
    response = get_client().embed(
        model=settings.embed_model, input=list(texts), keep_alive=keep_alive
    )
    return [list(v) for v in response.embeddings]


def embed_queries(questions: Sequence[str]) -> list[list[float]]:
    """Embed questions with the retrieval instruction prefix."""
    return embed_texts([embedding_query(q) for q in questions])


def embed_query(question: str) -> list[float]:
    return embed_queries([question])[0]


def unload_embed_model() -> None:
    get_client().embed(model=settings.embed_model, input=[""], keep_alive=0)
