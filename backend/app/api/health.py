"""GET /api/health: is Ollama up, are the models installed, are indexes built."""

import ollama
from fastapi import APIRouter

from app.config import settings
from app.core.store import get_collection
from app.models import Health

router = APIRouter()


def installed_models() -> set[str] | None:
    """Installed Ollama model tags, or None if Ollama is unreachable."""
    try:
        response = ollama.Client(host=settings.ollama_host, timeout=3).list()
    except Exception:
        return None
    return {m.model for m in response.models}


@router.get("/health")
def health() -> Health:
    problems: list[str] = []

    installed = installed_models()
    reachable = installed is not None
    if not reachable:
        problems.append(f"Ollama is not reachable at {settings.ollama_host}")
    wanted = {settings.chat_model, settings.fallback_chat_model, settings.embed_model}
    models = {name: reachable and name in installed for name in sorted(wanted)}
    if reachable:
        problems += [f"Model not installed: {name}" for name, ok in models.items() if not ok]

    try:
        count = get_collection().count() if settings.chroma_dir.exists() else None
    except Exception:
        count = None
    if not count:
        problems.append("Vector collection is empty or missing (run scripts.03_embed)")

    bm25 = (settings.bm25_dir / "chunk_ids.json").exists()
    if not bm25:
        problems.append("BM25 index missing (run scripts.04_bm25)")
    topics = (settings.topics_dir / "map.json").exists()
    if not topics:
        problems.append("Topic map missing (run scripts.05_topics)")
    eval_results = (settings.eval_dir / "retrieval_summary.json").exists()
    if not eval_results:
        problems.append("Evaluation results missing (run scripts.06_evaluate)")

    chat_ready = reachable and models[settings.chat_model] and models[settings.embed_model]
    if not chat_ready or not count or not bm25:
        status = "down"
    elif problems:
        status = "degraded"
    else:
        status = "ok"

    return Health(
        status=status,
        ollama=reachable,
        chat_model=settings.chat_model,
        models=models,
        collection_count=count,
        bm25_index=bm25,
        topics=topics,
        eval_results=eval_results,
        problems=problems,
    )
