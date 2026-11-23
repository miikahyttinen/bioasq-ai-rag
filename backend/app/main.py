"""FastAPI app.

cd backend && uv run uvicorn app.main:app --reload --port 8000
"""

import httpx
import ollama
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, eval, health, passages, questions, search, topics
from app.core.data import MissingDataError

app = FastAPI(
    title="BioASQ RAG Explorer",
    description="Local RAG over the rag-mini-bioasq dataset. A study tool, not medical advice.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Compresses large JSON (the topic map is ~2.4 MB); Starlette skips text/event-stream.
app.add_middleware(GZipMiddleware, minimum_size=1000)

for module in (chat, search, passages, questions, topics, eval, health):
    app.include_router(module.router, prefix="/api")


@app.exception_handler(MissingDataError)
def missing_data(_: Request, exc: MissingDataError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(httpx.ConnectError)
@app.exception_handler(ollama.ResponseError)
def ollama_unavailable(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"Ollama error: {exc}"})
