"""GET /api/search: retrieval only, no LLM."""

import time

from fastapi import APIRouter, Query

from app.api.common import to_sources
from app.core.retrieve import Strategy, search
from app.models import SearchResponse

router = APIRouter()


@router.get("/search")
def search_passages(
    q: str = Query(min_length=1, max_length=1000),
    strategy: Strategy = "hybrid",
    k: int = Query(10, ge=1, le=50),
) -> SearchResponse:
    start = time.perf_counter()
    hits = search(q, strategy, k)
    return SearchResponse(
        query=q,
        strategy=strategy,
        k=k,
        seconds=round(time.perf_counter() - start, 3),
        results=to_sources(hits),
    )
