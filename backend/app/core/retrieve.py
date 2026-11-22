"""Dense, BM25 and hybrid (Reciprocal Rank Fusion) retrieval.

Every strategy returns hits deduplicated by PubMed ID (best chunk per article),
best first. The *_many functions batch many queries for evaluation.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from app.config import settings
from app.core.bm25 import get_index
from app.core.embed import batched, embed_queries
from app.core.store import get_collection

Strategy = Literal["dense", "bm25", "hybrid"]
STRATEGIES: tuple[Strategy, ...] = ("dense", "bm25", "hybrid")

# Articles average ~1.3 chunks, so fetching 2x chunks leaves enough after dedup.
CHUNKS_PER_ARTICLE = 2
QUERY_BATCH = 64


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    pmid: int
    text: str
    score: float


def dedupe_by_pmid(hits: Sequence[Hit]) -> list[Hit]:
    """Keep the first (best-ranked) chunk of each article, preserving order."""
    seen: set[int] = set()
    unique = []
    for hit in hits:
        if hit.pmid not in seen:
            seen.add(hit.pmid)
            unique.append(hit)
    return unique


def rrf_fuse(rankings: Sequence[Sequence[Hit]], rrf_k: int) -> list[Hit]:
    """Reciprocal Rank Fusion at article level: score = sum of 1 / (rrf_k + rank).

    Rankings must already be deduplicated by pmid; ranks start at 1. Each fused hit
    keeps the chunk from the ranking where the article ranked best (earlier
    rankings win ties), with its score replaced by the RRF score.
    """
    scores: dict[int, float] = {}
    best: dict[int, tuple[int, Hit]] = {}  # pmid -> (best rank, hit)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.pmid] = scores.get(hit.pmid, 0.0) + 1.0 / (rrf_k + rank)
            if hit.pmid not in best or rank < best[hit.pmid][0]:
                best[hit.pmid] = (rank, hit)

    order = sorted(scores, key=lambda pmid: (-scores[pmid], best[pmid][0], pmid))
    return [replace(best[pmid][1], score=scores[pmid]) for pmid in order]


def dense_many(queries: Sequence[str], k: int) -> list[list[Hit]]:
    collection = get_collection()
    results: list[list[Hit]] = []
    for batch in batched(list(queries), QUERY_BATCH):
        response = collection.query(
            query_embeddings=embed_queries(batch),
            n_results=k * CHUNKS_PER_ARTICLE,
            include=["documents", "metadatas", "distances"],
        )
        for ids, docs, metas, dists in zip(
            response["ids"],
            response["documents"],
            response["metadatas"],
            response["distances"],
            strict=True,
        ):
            hits = [
                Hit(chunk_id=i, pmid=int(m["pmid"]), text=d, score=1.0 - dist)
                for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)
            ]
            results.append(dedupe_by_pmid(hits)[:k])
    return results


def bm25_many(queries: Sequence[str], k: int) -> list[list[Hit]]:
    index = get_index()
    results = []
    for rows in index.search_many(list(queries), n=k * CHUNKS_PER_ARTICLE):
        hits = [
            Hit(
                chunk_id=index.chunk_ids[row],
                pmid=index.pmids[row],
                text=index.texts[row],
                score=score,
            )
            for row, score in rows
        ]
        results.append(dedupe_by_pmid(hits)[:k])
    return results


def hybrid_many(queries: Sequence[str], k: int) -> list[list[Hit]]:
    n = max(k, settings.fusion_candidates)
    dense_results = dense_many(queries, n)
    bm25_results = bm25_many(queries, n)
    return [
        rrf_fuse([d, b], settings.rrf_k)[:k]
        for d, b in zip(dense_results, bm25_results, strict=True)
    ]


def search_many(queries: Sequence[str], strategy: Strategy, k: int) -> list[list[Hit]]:
    match strategy:
        case "dense":
            return dense_many(queries, k)
        case "bm25":
            return bm25_many(queries, k)
        case "hybrid":
            return hybrid_many(queries, k)
    raise ValueError(f"Unknown strategy: {strategy}")


def search(query: str, strategy: Strategy = "hybrid", k: int | None = None) -> list[Hit]:
    return search_many([query], strategy, k or settings.default_k)[0]


def dense(query: str, k: int) -> list[Hit]:
    return search(query, "dense", k)


def bm25(query: str, k: int) -> list[Hit]:
    return search(query, "bm25", k)


def hybrid(query: str, k: int) -> list[Hit]:
    return search(query, "hybrid", k)
