import importlib
import json

import numpy as np
import pytest

from app.core import prompts
from app.core.llm import ChatResult
from app.core.topics import (
    article_vectors,
    centroids,
    nearest_cluster,
    normalize,
    representatives,
)

topics_script = importlib.import_module("scripts.05_topics")


def test_article_vectors_average_chunks_and_normalize():
    chunk_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    pmids, vectors = article_vectors(chunk_vectors, ["7-0", "7-1", "5-0"])

    assert pmids.tolist() == [5, 7]
    np.testing.assert_allclose(vectors[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(vectors[1], [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_normalize_leaves_zero_vectors_alone():
    out = normalize(np.array([[0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0], [1.0, 0.0]])


def test_centroids_ignore_noise():
    points = np.array([[0, 0], [2, 0], [10, 10], [99, 99]], dtype=float)
    labels = np.array([0, 0, 1, -1])
    cents = centroids(points, labels)
    assert set(cents) == {0, 1}
    np.testing.assert_array_equal(cents[0], [1, 0])


def test_nearest_cluster_assigns_noise_to_closest_centroid():
    points = np.array([[0, 0], [1, 0], [10, 0], [11, 0], [2, 0], [9, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1, -1])
    assert nearest_cluster(points, labels).tolist() == [0, 0, 1, 1, 0, 1]


def test_nearest_cluster_with_only_noise():
    labels = np.array([-1, -1])
    assert nearest_cluster(np.zeros((2, 2)), labels).tolist() == [-1, -1]


def test_representatives_are_closest_to_cluster_direction():
    vectors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0], [0.95, 0.05]], dtype=np.float32
    )
    labels = np.array([0, 0, 0, 1, 0])
    reps = representatives(vectors, labels, cluster=0, n=2)
    assert set(reps.tolist()) <= {0, 1, 2, 4}
    assert 2 not in reps.tolist()  # the outlier-ish member is least typical
    assert len(reps) == 2


def test_topic_prompt_lists_excerpts_and_used_labels():
    messages = prompts.topic_label_messages(["Alpha text", "Beta text"], 120, ["Malaria"])
    user = messages[1]["content"]
    assert "[1] Alpha text" in user and "[2] Beta text" in user
    assert "cluster of 120 abstracts" in user
    assert "Malaria" in user
    assert "already used" not in prompts.topic_label_messages(["x"], 5)[1]["content"]


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, **kwargs):
        self.prompts.append(messages[1]["content"])
        return ChatResult(self.replies.pop(0), 10, 10, 0.1)


@pytest.fixture
def two_clusters():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 1])
    pmids = np.array([10, 11, 20, 21, 22])
    texts = {p: f"abstract {p}" for p in pmids.tolist()}
    return labels, vectors, pmids, texts


def label(text: str) -> str:
    return json.dumps({"label": text, "description": f"About {text.lower()} in detail."})


def test_label_clusters_largest_first_and_avoids_used_labels(two_clusters, monkeypatch):
    monkeypatch.setattr(topics_script.settings, "topics_representatives", 2)
    fake = FakeChat([label("Gene Therapy"), label("Malaria Drugs")])

    named = topics_script.label_clusters(*two_clusters, chat_fn=fake)

    assert named[1]["label"] == "Gene Therapy"  # cluster 1 is larger -> labelled first
    assert named[0]["label"] == "Malaria Drugs"
    assert "Gene Therapy" in fake.prompts[1]
    assert len(named[1]["representative_pmids"]) == 2


def test_label_clusters_falls_back_after_invalid_output(two_clusters):
    fake = FakeChat(["nope", "{}", label("Gene Therapy"), label("Malaria Drugs")])
    named = topics_script.label_clusters(*two_clusters, chat_fn=fake)
    assert named[1]["label"] == "Topic 1"
    assert named[0]["label"] == "Gene Therapy"
