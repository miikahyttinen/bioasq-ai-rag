"""Vector helpers for the topic map: per-article vectors, cluster centroids, representatives."""

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def article_vectors(
    chunk_vectors: np.ndarray, chunk_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Average each article's chunk vectors (then re-normalize).

    Chunk IDs are "{pmid}-{index}". Returns (sorted unique pmids, vectors in that order).
    """
    pmids = np.array([int(cid.rsplit("-", 1)[0]) for cid in chunk_ids])
    unique, inverse = np.unique(pmids, return_inverse=True)
    summed = np.zeros((len(unique), chunk_vectors.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, chunk_vectors)
    return unique, normalize(summed).astype(np.float32)


def centroids(points: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean point per cluster label, ignoring noise (-1)."""
    return {int(c): points[labels == c].mean(axis=0) for c in np.unique(labels) if c >= 0}


def nearest_cluster(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cluster of each point; noise points get the cluster with the nearest centroid."""
    cents = centroids(points, labels)
    if not cents:
        return labels.copy()
    ids = np.array(list(cents))
    matrix = np.stack([cents[c] for c in ids])
    distances = np.linalg.norm(points[:, None, :] - matrix[None, :, :], axis=2)
    result = labels.copy()
    noise = labels < 0
    result[noise] = ids[distances[noise].argmin(axis=1)]
    return result


def representatives(vectors: np.ndarray, labels: np.ndarray, cluster: int, n: int) -> np.ndarray:
    """Row indices of the n members most similar (cosine) to the cluster's mean direction."""
    members = np.flatnonzero(labels == cluster)
    direction = normalize(vectors[members].mean(axis=0, keepdims=True))[0]
    similarity = normalize(vectors[members]) @ direction
    return members[np.argsort(-similarity, kind="stable")[:n]]
