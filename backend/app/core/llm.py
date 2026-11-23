"""Ollama chat wrapper."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.core.embed import get_client


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    format: dict[str, Any] | str | None = None,
    think: bool | None = None,
    keep_alive: str | int | None = None,
) -> ChatResult:
    start = time.perf_counter()
    response = get_client().chat(
        model=model or settings.chat_model,
        messages=messages,
        format=format,
        think=settings.chat_think if think is None else think,
        options={"num_ctx": settings.num_ctx, "temperature": temperature},
        keep_alive=keep_alive,
    )
    return ChatResult(
        text=response.message.content or "",
        prompt_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        seconds=time.perf_counter() - start,
    )


@dataclass
class StreamStats:
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    think: bool | None = None,
    stats: StreamStats | None = None,
) -> Iterator[str]:
    """Yield answer text pieces as they are generated; fills `stats` when done."""
    start = time.perf_counter()
    stream = get_client().chat(
        model=model or settings.chat_model,
        messages=messages,
        think=settings.chat_think if think is None else think,
        options={"num_ctx": settings.num_ctx, "temperature": temperature},
        stream=True,
    )
    for part in stream:
        if part.message.content:
            yield part.message.content
        if part.done and stats is not None:
            stats.prompt_tokens = part.prompt_eval_count or 0
            stats.output_tokens = part.eval_count or 0
    if stats is not None:
        stats.seconds = time.perf_counter() - start


def chat_json[M: BaseModel](
    messages: list[dict[str, str]],
    schema: type[M],
    chat_fn: Callable[..., ChatResult] = chat,
    attempts: int = 2,
) -> tuple[M | None, int, str | None]:
    """Request JSON constrained to `schema`; validate, retrying on invalid output.

    Returns (parsed or None, attempts used, last validation error).
    """
    error = None
    for attempt in range(1, attempts + 1):
        result = chat_fn(messages, format=schema.model_json_schema(), temperature=0.0)
        try:
            return schema.model_validate_json(result.text), attempt, None
        except ValidationError as exc:
            error = str(exc).splitlines()[0]
    return None, attempts, error


def unload(model: str | None = None) -> None:
    get_client().chat(model=model or settings.chat_model, messages=[], keep_alive=0)
