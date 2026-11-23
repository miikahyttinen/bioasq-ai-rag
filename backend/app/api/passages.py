"""GET /api/passages/{pmid}: full abstract text and its topic."""

from fastapi import APIRouter, HTTPException

from app.api.common import cluster_info
from app.core import data
from app.models import Passage

router = APIRouter()


@router.get("/passages/{pmid}")
def get_passage(pmid: int) -> Passage:
    text = data.passages().get(pmid)
    if text is None:
        raise HTTPException(404, f"No usable passage with PMID {pmid}")
    cluster_id, cluster_label, nearest_id, nearest_label = cluster_info(pmid)
    return Passage(
        pmid=pmid,
        text=text,
        url=data.pubmed_url(pmid),
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        nearest_cluster_id=nearest_id,
        nearest_cluster_label=nearest_label,
    )
