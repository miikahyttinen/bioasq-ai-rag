"""GET /api/topics/map and /api/topics/clusters."""

from fastapi import APIRouter

from app.core import data

router = APIRouter()


@router.get("/topics/map")
def topic_map() -> list[dict]:
    """One point per article: {pmid, x, y, cluster_id, nearest_cluster}."""
    return data.topic_points()


@router.get("/topics/clusters")
def topic_clusters() -> list[dict]:
    """Clusters largest first, then the "Unclustered" group (cluster_id -1)."""
    return data.topic_clusters()
