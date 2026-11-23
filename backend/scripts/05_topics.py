"""Build the topic map: UMAP to 2D, HDBSCAN clusters, LLM topic labels.

  uv run python -m scripts.05_topics            # reuses cached UMAP projections
  uv run python -m scripts.05_topics --fresh    # recompute projections

Outputs in storage/topics/:
  map.json        [{pmid, x, y, cluster_id, nearest_cluster}]  one point per article
  clusters.json   [{cluster_id, label, description, size, x, y, representative_pmids}]
  projection.npz  cached UMAP results (2D and clustering dims)
"""

import argparse
import json
import time
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from app.config import settings
from app.core import prompts
from app.core.llm import ChatResult, chat, chat_json, unload
from app.core.topics import article_vectors, centroids, nearest_cluster, representatives
from app.models import TopicLabel

NOISE_LABEL = {
    "label": "Unclustered",
    "description": "Abstracts that do not fall clearly into any dense topic cluster.",
}


def timed(message: str):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"{message}...", end="", flush=True)

        def __exit__(self, *exc):
            print(f" {time.perf_counter() - self.start:.1f}s")

    return Timer()


def project(vectors: np.ndarray, n_components: int, min_dist: float) -> np.ndarray:
    import umap  # slow import (numba); only needed when projecting

    with warnings.catch_warnings():
        # random_state disables UMAP's parallelism and warns about it; we want reproducibility.
        warnings.filterwarnings("ignore", message=".*random_state.*")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=settings.topics_n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=settings.topics_seed,
            compatibility_layout=True,  # pin the current UMAP algorithm (reproducible maps)
        )
        return reducer.fit_transform(vectors).astype(np.float32)


def load_projections(vectors: np.ndarray, pmids: np.ndarray, fresh: bool):
    cache = settings.topics_dir / "projection.npz"
    if not fresh and cache.exists():
        cached = np.load(cache)
        if np.array_equal(cached["pmids"], pmids):
            print(f"Using cached projections ({cache.name}; --fresh to recompute)")
            return cached["xy"], cached["xc"]

    with timed(f"UMAP {vectors.shape[1]}D -> 2D (map)"):
        xy = project(vectors, 2, min_dist=0.1)
    # min_dist=0 packs clusters tightly, which suits density-based clustering.
    with timed(f"UMAP {vectors.shape[1]}D -> {settings.topics_cluster_dims}D (clustering)"):
        xc = project(vectors, settings.topics_cluster_dims, min_dist=0.0)
    np.savez(cache, pmids=pmids, xy=xy, xc=xc)
    return xy, xc


def label_clusters(
    labels: np.ndarray,
    vectors: np.ndarray,
    pmids: np.ndarray,
    texts: dict[int, str],
    chat_fn: Callable[..., ChatResult] = chat,
) -> dict[int, dict]:
    """Ask the LLM for a label per cluster, largest cluster first."""
    clusters = [c for c in np.unique(labels) if c >= 0]
    clusters.sort(key=lambda c: -(labels == c).sum())
    named: dict[int, dict] = {}
    used: list[str] = []
    for cluster in tqdm(clusters, unit="cluster"):
        rows = representatives(vectors, labels, cluster, settings.topics_representatives)
        rep_pmids = [int(pmids[r]) for r in rows]
        excerpts = [texts[p][: settings.topics_representative_chars] for p in rep_pmids]
        size = int((labels == cluster).sum())
        topic, attempts, error = chat_json(
            prompts.topic_label_messages(excerpts, size, used), TopicLabel, chat_fn
        )
        if topic is None:
            print(f"\nCluster {cluster}: no valid label after {attempts} attempts ({error})")
            topic = TopicLabel(label=f"Topic {cluster}", description="Label generation failed.")
        used.append(topic.label)
        named[int(cluster)] = {**topic.model_dump(), "representative_pmids": rep_pmids}
    return named


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="recompute UMAP projections")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="skip LLM labelling (overwrites labels with placeholders)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    settings.topics_dir.mkdir(parents=True, exist_ok=True)

    chunk_vectors = np.load(settings.storage_dir / "embeddings.npy")
    chunk_ids = json.loads((settings.storage_dir / "embedding_ids.json").read_text())
    pmids, vectors = article_vectors(chunk_vectors, chunk_ids)
    print(f"Articles: {len(pmids):,} (from {len(chunk_ids):,} chunk vectors)")

    xy, xc = load_projections(vectors, pmids, args.fresh)

    with timed("HDBSCAN"):
        labels = HDBSCAN(
            min_cluster_size=settings.topics_min_cluster_size,
            min_samples=settings.topics_min_samples,
            copy=True,
        ).fit_predict(xc)
    n_clusters = int(labels.max()) + 1
    print(f"Clusters: {n_clusters}, noise: {np.mean(labels < 0):.0%} of articles")
    nearest = nearest_cluster(xc, labels)

    passages = pd.read_parquet(settings.processed_dir / "passages.parquet")
    texts = dict(zip(passages["pmid"].astype(int), passages["text"], strict=True))

    if args.no_labels:
        named = {
            c: {"label": f"Topic {c}", "description": "", "representative_pmids": []}
            for c in range(n_clusters)
        }
    else:
        print(f"Labelling {n_clusters} clusters with {settings.chat_model}")
        try:
            named = label_clusters(labels, vectors, pmids, texts)
        finally:
            unload()

    centers = centroids(xy, labels)
    clusters = [
        {
            "cluster_id": c,
            **named[c],
            "size": int((labels == c).sum()),
            "x": round(float(centers[c][0]), 4),
            "y": round(float(centers[c][1]), 4),
        }
        for c in sorted(named, key=lambda c: -(labels == c).sum())
    ]
    clusters.append(
        {
            "cluster_id": -1,
            **NOISE_LABEL,
            "representative_pmids": [],
            "size": int((labels < 0).sum()),
            "x": None,
            "y": None,
        }
    )
    points = [
        {
            "pmid": int(p),
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "cluster_id": int(c),
            "nearest_cluster": int(n),
        }
        for p, (x, y), c, n in zip(pmids, xy, labels, nearest, strict=True)
    ]

    (settings.topics_dir / "map.json").write_text(json.dumps(points))
    (settings.topics_dir / "clusters.json").write_text(json.dumps(clusters, indent=2))
    print(f"\nSaved {len(points):,} points and {n_clusters} clusters to {settings.topics_dir}")

    print("\nLargest topics:")
    for c in clusters[:15]:
        print(f"  {c['size']:>5}  {c['label']}")
    print(f"\nDone in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
