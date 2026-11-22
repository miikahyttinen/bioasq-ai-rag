"""BM25 keyword index over chunk texts (bm25s)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import bm25s
import pandas as pd
import Stemmer
from bm25s.tokenization import Tokenizer

from app.config import settings

# Default bm25s splitter drops 1-char tokens; biomedical text needs them
# ("IL-6", "vitamin D", "hepatitis B", "type 2 diabetes").
TOKEN_PATTERN = r"(?u)\b\w+\b"
CHUNK_IDS_FILE = "chunk_ids.json"


def make_tokenizer() -> Tokenizer:
    return Tokenizer(
        lower=True,
        splitter=TOKEN_PATTERN,
        stopwords="en",
        stemmer=Stemmer.Stemmer("english"),
    )


def build_index(chunk_ids: list[str], texts: list[str], out_dir: Path) -> None:
    tokenizer = make_tokenizer()
    tokens = tokenizer.tokenize(texts, return_as="tuple", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(out_dir)
    tokenizer.save_vocab(out_dir)
    tokenizer.save_stopwords(out_dir)
    (out_dir / CHUNK_IDS_FILE).write_text(json.dumps(chunk_ids))


@dataclass(frozen=True)
class BM25Index:
    retriever: bm25s.BM25
    tokenizer: Tokenizer
    chunk_ids: list[str]
    pmids: list[int]
    texts: list[str]

    def search_many(self, queries: list[str], n: int) -> list[list[tuple[int, float]]]:
        """Top-n (row, score) per query. Rows with score 0 (no shared terms) are dropped."""
        if not queries:
            return []
        token_ids = self.tokenizer.tokenize(
            queries, update_vocab=False, return_as="ids", show_progress=False
        )
        rows, scores = self.retriever.retrieve(
            token_ids, k=min(n, len(self.chunk_ids)), show_progress=False
        )
        return [
            [(int(r), float(s)) for r, s in zip(row, score, strict=True) if s > 0]
            for row, score in zip(rows, scores, strict=True)
        ]


def load_index(index_dir: Path, chunks: pd.DataFrame) -> BM25Index:
    tokenizer = make_tokenizer()
    tokenizer.load_vocab(index_dir)
    tokenizer.load_stopwords(index_dir)
    chunk_ids = json.loads((index_dir / CHUNK_IDS_FILE).read_text())
    rows = chunks.set_index("chunk_id").loc[chunk_ids]
    return BM25Index(
        retriever=bm25s.BM25.load(index_dir, show_progress=False),
        tokenizer=tokenizer,
        chunk_ids=chunk_ids,
        pmids=rows["pmid"].astype(int).tolist(),
        texts=rows["text"].tolist(),
    )


@lru_cache
def get_index() -> BM25Index:
    chunks = pd.read_parquet(settings.processed_dir / "chunks.parquet")
    return load_index(settings.bm25_dir, chunks)
