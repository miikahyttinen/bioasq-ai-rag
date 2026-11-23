"""POST /api/chat: retrieve sources, then stream the answer over Server-Sent Events.

Events:
  sources  {"sources": [...]}              before the answer, so citations can link early
  token    {"text": "..."}                 answer text pieces
  done     {"sources": [...], "model", "prompt_tokens", "output_tokens", "seconds"}
  error    {"message": "..."}              generation failed mid-stream
"""

import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.common import resolve_model, to_sources
from app.core import llm, prompts
from app.core.retrieve import search
from app.models import ChatRequest

router = APIRouter()


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    model = resolve_model(request.model)
    # Retrieval runs before streaming starts, so its errors become normal HTTP errors.
    hits = search(request.message, request.strategy, request.k)
    sources = [s.model_dump() for s in to_sources(hits)]
    messages = prompts.chat_messages(
        request.message, hits, [m.model_dump() for m in request.history]
    )

    def events() -> Iterator[str]:
        yield sse("sources", {"sources": sources})
        stats = llm.StreamStats()
        try:
            for piece in llm.stream_chat(messages, model=model, think=request.think, stats=stats):
                yield sse("token", {"text": piece})
        except Exception as exc:  # stream already started; report in-band
            yield sse("error", {"message": f"Generation failed: {exc}"})
            return
        yield sse(
            "done",
            {
                "sources": sources,
                "model": model,
                "prompt_tokens": stats.prompt_tokens,
                "output_tokens": stats.output_tokens,
                "seconds": round(stats.seconds, 2),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
