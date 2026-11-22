"""Embed all chunks into the persistent Chroma collection.

Resumable: chunk IDs already in the collection are skipped, so rerunning after a
crash continues where it stopped. At the end all vectors are exported to
storage/embeddings.npy (+ embedding_ids.json, same order) for the topic map.
"""

import json
import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from chromadb.api.models.Collection import Collection
from tqdm import tqdm

from app.config import settings
from app.core.embed import batched, embed_query, embed_texts, unload_embed_model
from app.core.store import existing_ids, get_collection

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def embed_with_retry(embed_fn: EmbedFn, texts: Sequence[str], attempts: int = 3) -> list:
    for attempt in range(1, attempts + 1):
        try:
            return embed_fn(texts)
        except Exception as exc:  # Ollama hiccups (timeouts, model reload) are transient
            if attempt == attempts:
                raise
            print(f"\nEmbedding failed ({exc}); retry {attempt}/{attempts - 1}")
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def embed_into_collection(
    chunks: pd.DataFrame, collection: Collection, embed_fn: EmbedFn, batch_size: int
) -> int:
    """Embed chunks whose IDs are not yet in the collection. Returns how many were added."""
    done = existing_ids(collection)
    todo = chunks[~chunks["chunk_id"].isin(done)]
    print(f"Chunks: {len(chunks):,} total, {len(done):,} already embedded, {len(todo):,} to do")
    if todo.empty:
        return 0

    records = todo.to_dict("records")
    with tqdm(total=len(records), unit="chunk") as progress:
        for batch in batched(records, batch_size):
            vectors = embed_with_retry(embed_fn, [r["text"] for r in batch])
            collection.add(
                ids=[r["chunk_id"] for r in batch],
                embeddings=vectors,
                documents=[r["text"] for r in batch],
                metadatas=[
                    {"pmid": int(r["pmid"]), "chunk_index": int(r["chunk_index"])} for r in batch
                ],
            )
            progress.update(len(batch))
    return len(records)


def export_embeddings(collection: Collection, chunk_ids: list[str], page_size: int = 2000) -> None:
    """Save all vectors as float32 [n_chunks, dim] in chunks.parquet order."""
    by_id: dict[str, np.ndarray] = {}
    for page in batched(chunk_ids, page_size):
        got = collection.get(ids=list(page), include=["embeddings"])
        by_id.update(zip(got["ids"], got["embeddings"], strict=True))

    missing = [cid for cid in chunk_ids if cid not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} chunks missing from collection, e.g. {missing[:3]}")

    matrix = np.asarray([by_id[cid] for cid in chunk_ids], dtype=np.float32)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.storage_dir / "embeddings.npy", matrix)
    (settings.storage_dir / "embedding_ids.json").write_text(json.dumps(chunk_ids))
    print(f"Exported embeddings.npy {matrix.shape} ({matrix.nbytes / 1e6:.0f} MB)")


def smoke_test(collection: Collection, question: str = "What causes Pompe disease?") -> None:
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    print(f"\nSmoke test: {question!r}")
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
    ):
        print(f"  {1 - dist:.3f}  pmid {meta['pmid']:>9}  {doc[:90]}...")


def main() -> None:
    start = time.perf_counter()
    chunks = pd.read_parquet(settings.processed_dir / "chunks.parquet")
    collection = get_collection()

    added = embed_into_collection(chunks, collection, embed_texts, settings.embed_batch_size)
    embed_seconds = time.perf_counter() - start
    if added:
        print(
            f"Embedded {added:,} chunks in {embed_seconds / 60:.1f} min "
            f"({added / embed_seconds:.1f} chunks/s)"
        )

    count = collection.count()
    print(f"Collection '{settings.collection_name}': {count:,} vectors (chunks: {len(chunks):,})")
    if count != len(chunks):
        raise RuntimeError("Collection count does not match chunk count")

    export_embeddings(collection, chunks["chunk_id"].tolist())
    smoke_test(collection)
    unload_embed_model()
    print(f"\nDone in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
