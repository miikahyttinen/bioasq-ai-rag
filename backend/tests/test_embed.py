import importlib

import pandas as pd
import pytest

from app.core.embed import batched
from app.core.prompts import embedding_query
from app.core.store import existing_ids, get_collection

embed_script = importlib.import_module("scripts.03_embed")


def fake_chunks(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": [f"{1000 + i // 2}-{i % 2}" for i in range(n)],
            "pmid": [1000 + i // 2 for i in range(n)],
            "chunk_index": [i % 2 for i in range(n)],
            "text": [f"chunk text {i}" for i in range(n)],
        }
    )


class FakeEmbedder:
    """Deterministic 4-dim vectors; can be told to crash after N calls."""

    def __init__(self, fail_after_calls: int | None = None):
        self.calls = 0
        self.texts_seen: list[str] = []
        self.fail_after_calls = fail_after_calls

    def __call__(self, texts):
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise KeyboardInterrupt("simulated crash")
        self.calls += 1
        self.texts_seen.extend(texts)
        return [[float(len(t)), 1.0, 0.0, float(i)] for i, t in enumerate(texts)]


@pytest.fixture
def collection(tmp_path):
    return get_collection(tmp_path / "chroma")


def test_batched_splits_with_remainder():
    assert [list(b) for b in batched(list(range(7)), 3)] == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batched([], 3)) == []


def test_query_gets_instruction_prefix():
    q = embedding_query("What causes Pompe disease?")
    assert q.startswith("Instruct: ")
    assert q.endswith("\nQuery: What causes Pompe disease?")


def test_collection_uses_cosine(collection):
    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_embeds_all_chunks_with_metadata(collection):
    chunks = fake_chunks(10)
    embedder = FakeEmbedder()

    added = embed_script.embed_into_collection(chunks, collection, embedder, batch_size=4)

    assert added == 10
    assert embedder.calls == 3  # 4 + 4 + 2
    assert collection.count() == 10
    got = collection.get(ids=["1002-1"], include=["metadatas", "documents"])
    assert got["metadatas"][0] == {"pmid": 1002, "chunk_index": 1}
    assert got["documents"][0] == "chunk text 5"


def test_resume_skips_already_embedded_chunks(collection):
    chunks = fake_chunks(10)

    with pytest.raises(KeyboardInterrupt):
        embed_script.embed_into_collection(
            chunks, collection, FakeEmbedder(fail_after_calls=2), batch_size=3
        )
    assert collection.count() == 6  # two batches landed before the crash

    resumed = FakeEmbedder()
    added = embed_script.embed_into_collection(chunks, collection, resumed, batch_size=3)

    assert added == 4
    assert resumed.texts_seen == [f"chunk text {i}" for i in range(6, 10)]
    assert existing_ids(collection) == set(chunks["chunk_id"])


def test_rerun_when_complete_does_nothing(collection):
    chunks = fake_chunks(5)
    embed_script.embed_into_collection(chunks, collection, FakeEmbedder(), batch_size=2)

    embedder = FakeEmbedder()
    assert embed_script.embed_into_collection(chunks, collection, embedder, batch_size=2) == 0
    assert embedder.calls == 0


def test_existing_ids_pages_through_large_collections(collection):
    embed_script.embed_into_collection(fake_chunks(25), collection, FakeEmbedder(), batch_size=10)
    assert len(existing_ids(collection, page_size=7)) == 25


def test_retry_recovers_from_transient_error(monkeypatch):
    monkeypatch.setattr(embed_script.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("ollama restarting")
        return [[1.0] for _ in texts]

    assert embed_script.embed_with_retry(flaky, ["a"]) == [[1.0]]
    assert calls["n"] == 2
