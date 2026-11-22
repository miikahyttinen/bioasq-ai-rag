"""Ollama chat wrapper."""

import time
from dataclasses import dataclass
from typing import Any

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


def unload(model: str | None = None) -> None:
    get_client().chat(model=model or settings.chat_model, messages=[], keep_alive=0)
