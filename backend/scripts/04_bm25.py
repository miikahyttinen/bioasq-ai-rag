"""Build the BM25 keyword index over chunk texts -> storage/bm25/."""

import time

import pandas as pd

from app.config import settings
from app.core.bm25 import build_index, get_index


def main() -> None:
    start = time.perf_counter()
    chunks = pd.read_parquet(settings.processed_dir / "chunks.parquet")
    build_index(chunks["chunk_id"].tolist(), chunks["text"].tolist(), settings.bm25_dir)
    print(f"Indexed {len(chunks):,} chunks in {time.perf_counter() - start:.1f}s")

    index = get_index()
    print(f"Vocabulary: {len(index.tokenizer.get_vocab_dict()):,} terms -> {settings.bm25_dir}")

    question = "What causes Pompe disease?"
    print(f"\nSmoke test: {question!r}")
    for row, score in index.search_many([question], n=5)[0]:
        print(f"  {score:6.2f}  pmid {index.pmids[row]:>9}  {index.texts[row][:90]}...")


if __name__ == "__main__":
    main()
