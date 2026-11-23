"""Helpers shared by the API routers."""

from collections.abc import Sequence

from fastapi import HTTPException

from app.config import settings
from app.core import data
from app.core.retrieve import Hit
from app.models import Source


def cluster_info(pmid: int) -> tuple[int | None, str | None, int | None, str | None]:
    """(cluster_id, label, nearest_cluster_id, nearest_label); Nones if no topic map yet."""
    try:
        cluster, nearest = data.cluster_of_pmid().get(pmid, (None, None))
        clusters = data.clusters_by_id()
    except data.MissingDataError:
        return None, None, None, None

    def label(cid: int | None) -> str | None:
        return clusters[cid]["label"] if cid is not None and cid in clusters else None

    return cluster, label(cluster), nearest, label(nearest)


def to_sources(hits: Sequence[Hit]) -> list[Source]:
    sources = []
    for n, hit in enumerate(hits, start=1):
        cluster_id, cluster_label, _, _ = cluster_info(hit.pmid)
        sources.append(
            Source(
                n=n,
                chunk_id=hit.chunk_id,
                pmid=hit.pmid,
                text=hit.text,
                score=round(hit.score, 4),
                url=data.pubmed_url(hit.pmid),
                cluster_id=cluster_id,
                cluster_label=cluster_label,
            )
        )
    return sources


def resolve_model(requested: str | None) -> str:
    allowed = (settings.chat_model, settings.fallback_chat_model)
    if requested is None:
        return settings.chat_model
    if requested not in allowed:
        raise HTTPException(400, f"model must be one of {list(allowed)}")
    return requested
