"""Cached, read-only access to pipeline outputs (processed data, topics, eval results).

Each loader reads its files once per process. `clear_caches()` resets them (tests, or
after re-running a pipeline step).
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import settings


class MissingDataError(RuntimeError):
    """A pipeline output the request needs has not been built yet."""

    def __init__(self, path: Path, step: str):
        super().__init__(f"{path.name} not found; run `{step}` first")


def _require(path: Path, step: str) -> Path:
    if not path.exists():
        raise MissingDataError(path, step)
    return path


def _read_json(path: Path, step: str):
    return json.loads(_require(path, step).read_text())


@lru_cache
def passages() -> dict[int, str]:
    df = pd.read_parquet(
        _require(settings.processed_dir / "passages.parquet", "scripts.02_prepare")
    )
    return dict(zip(df["pmid"].astype(int), df["text"], strict=True))


@lru_cache
def questions() -> pd.DataFrame:
    df = pd.read_parquet(
        _require(settings.processed_dir / "questions.parquet", "scripts.02_prepare")
    )
    df["relevant_ids"] = df["relevant_ids"].map(lambda ids: [int(i) for i in ids])
    df["usable_relevant_ids"] = df["usable_relevant_ids"].map(lambda ids: [int(i) for i in ids])
    return df.set_index("qid", drop=False)


@lru_cache
def topic_points() -> list[dict]:
    return _read_json(settings.topics_dir / "map.json", "scripts.05_topics")


@lru_cache
def topic_clusters() -> list[dict]:
    return _read_json(settings.topics_dir / "clusters.json", "scripts.05_topics")


@lru_cache
def clusters_by_id() -> dict[int, dict]:
    return {c["cluster_id"]: c for c in topic_clusters()}


@lru_cache
def cluster_of_pmid() -> dict[int, tuple[int, int]]:
    """pmid -> (cluster_id, nearest_cluster)."""
    return {p["pmid"]: (p["cluster_id"], p["nearest_cluster"]) for p in topic_points()}


@lru_cache
def retrieval_summary() -> dict:
    return _read_json(settings.eval_dir / "retrieval_summary.json", "scripts.06_evaluate")


@lru_cache
def retrieval_questions() -> dict[int, dict]:
    rows = _read_json(settings.eval_dir / "retrieval_questions.json", "scripts.06_evaluate")
    return {r["qid"]: r for r in rows}


@lru_cache
def answers_summary() -> dict:
    return _read_json(settings.eval_dir / "answers_summary.json", "scripts.06_evaluate")


@lru_cache
def answer_records() -> dict[int, dict]:
    path = _require(settings.eval_dir / "answers.jsonl", "scripts.06_evaluate")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["qid"]: r for r in records}


def clear_caches() -> None:
    for loader in (
        passages,
        questions,
        topic_points,
        topic_clusters,
        clusters_by_id,
        cluster_of_pmid,
        retrieval_summary,
        retrieval_questions,
        answers_summary,
        answer_records,
    ):
        loader.cache_clear()


def pubmed_url(pmid: int) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
