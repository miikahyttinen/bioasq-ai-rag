"""Access to the persistent Chroma collection of chunk embeddings."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import settings


@lru_cache
def get_client(path: Path | None = None) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(path or settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(path: Path | None = None) -> Collection:
    # embedding_function=None: vectors always come from Ollama, never Chroma's
    # built-in model (which would otherwise be downloaded on first use).
    return get_client(path).get_or_create_collection(
        settings.collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def existing_ids(collection: Collection, page_size: int = 5000) -> set[str]:
    ids: set[str] = set()
    offset = 0
    while True:
        page = collection.get(include=[], limit=page_size, offset=offset)["ids"]
        ids.update(page)
        if len(page) < page_size:
            return ids
        offset += page_size
