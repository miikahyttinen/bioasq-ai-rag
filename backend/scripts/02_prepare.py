"""Clean and chunk passages; parse questions and their usable gold passage IDs.

Outputs (data/processed/):
  passages.parquet   pmid, text                     one row per usable abstract
  chunks.parquet     chunk_id, pmid, chunk_index, text, char_len
  questions.parquet  qid, question, answer, relevant_ids, usable_relevant_ids, has_usable
"""

import json
import time

import pandas as pd

from app.config import settings
from app.core.text import chunk_text, is_usable, normalize_whitespace


def prepare_passages() -> pd.DataFrame:
    raw = pd.read_parquet(settings.raw_dir / "passages.parquet").reset_index()
    raw = raw.rename(columns={"id": "pmid", "passage": "text"})
    total = len(raw)

    usable = raw["text"].map(lambda t: is_usable(t, settings.min_passage_chars))
    missing = (raw["text"].str.strip().str.lower() == "nan").sum()
    passages = raw[usable].copy()
    passages["text"] = passages["text"].map(normalize_whitespace)

    print(f"Passages: {total:,} total")
    print(f"  dropped {missing:,} missing abstracts ('nan')")
    print(f"  dropped {total - missing - len(passages):,} stubs, empty or citation-only")
    print(f"  kept    {len(passages):,}")
    return passages[["pmid", "text"]].reset_index(drop=True)


def make_chunks(passages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pmid, text in zip(passages["pmid"], passages["text"], strict=True):
        pieces = chunk_text(
            text,
            max_whole_chars=settings.max_whole_passage_chars,
            chunk_chars=settings.chunk_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        for i, piece in enumerate(pieces):
            rows.append(
                {
                    "chunk_id": f"{pmid}-{i}",
                    "pmid": int(pmid),
                    "chunk_index": i,
                    "text": piece,
                    "char_len": len(piece),
                }
            )
    chunks = pd.DataFrame(rows)

    per_passage = chunks.groupby("pmid").size()
    print(f"Chunks: {len(chunks):,}")
    print(f"  passages kept whole: {(per_passage == 1).sum():,}")
    print(f"  passages split:      {(per_passage > 1).sum():,} (max {per_passage.max()} chunks)")
    print(
        f"  chunk length: median {chunks['char_len'].median():.0f}, "
        f"min {chunks['char_len'].min()}, max {chunks['char_len'].max()}"
    )
    return chunks


def prepare_questions(usable_pmids: set[int]) -> pd.DataFrame:
    raw = pd.read_parquet(settings.raw_dir / "questions.parquet").reset_index()
    questions = raw.rename(columns={"id": "qid"})
    # Plain object columns: pandas 3 infers an Arrow list dtype here that it writes
    # to parquet but cannot read back.
    relevant = [json.loads(ids) for ids in questions["relevant_passage_ids"]]
    questions["relevant_ids"] = pd.Series(relevant, dtype=object)
    questions["usable_relevant_ids"] = pd.Series(
        [[i for i in ids if i in usable_pmids] for ids in relevant], dtype=object
    )
    questions["has_usable"] = questions["usable_relevant_ids"].map(len) > 0
    questions = questions.drop(columns=["relevant_passage_ids"])

    print(f"Questions: {len(questions):,} total")
    print(f"  usable (>= 1 non-empty gold passage): {questions['has_usable'].sum():,}")
    print(f"  without usable gold passages:         {(~questions['has_usable']).sum():,}")
    return questions


def main() -> None:
    start = time.perf_counter()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    passages = prepare_passages()
    chunks = make_chunks(passages)
    questions = prepare_questions(set(passages["pmid"]))

    passages.to_parquet(settings.processed_dir / "passages.parquet", index=False)
    chunks.to_parquet(settings.processed_dir / "chunks.parquet", index=False)
    questions.to_parquet(settings.processed_dir / "questions.parquet", index=False)

    print(f"Saved to {settings.processed_dir} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
