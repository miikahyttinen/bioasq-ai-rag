import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.api import health as health_api
from app.api import questions as questions_api
from app.api import search as search_api
from app.config import settings
from app.core import data, llm
from app.core.retrieve import Hit
from app.main import app

PASSAGES = {
    101: "Pompe disease is caused by acid alpha-glucosidase deficiency.",
    102: "GAA mutations lead to glycogen accumulation in muscle.",
    103: "BRCA1 mutations raise breast cancer risk.",
}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Small fake pipeline outputs in tmp dirs, with loader caches reset."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    for d in (settings.processed_dir, settings.topics_dir, settings.eval_dir):
        d.mkdir(parents=True)

    pd.DataFrame({"pmid": list(PASSAGES), "text": list(PASSAGES.values())}).to_parquet(
        settings.processed_dir / "passages.parquet", index=False
    )
    pd.DataFrame(
        {
            "qid": [1, 2, 3],
            "question": ["What causes Pompe disease?", "Is BRCA1 linked to cancer?", "Q3?"],
            "answer": ["GAA deficiency.", "Yes.", "Unknown."],
            "relevant_ids": pd.Series([[101, 102, 999], [103], [999]], dtype=object),
            "usable_relevant_ids": pd.Series([[101, 102], [103], []], dtype=object),
            "has_usable": [True, True, False],
        }
    ).to_parquet(settings.processed_dir / "questions.parquet", index=False)

    (settings.topics_dir / "map.json").write_text(
        json.dumps(
            [
                {"pmid": 101, "x": 0.1, "y": 0.2, "cluster_id": 0, "nearest_cluster": 0},
                {"pmid": 102, "x": 0.3, "y": 0.1, "cluster_id": -1, "nearest_cluster": 0},
                {"pmid": 103, "x": 5.0, "y": 5.0, "cluster_id": 1, "nearest_cluster": 1},
            ]
        )
    )
    (settings.topics_dir / "clusters.json").write_text(
        json.dumps(
            [
                {"cluster_id": 0, "label": "Lysosomal Storage Diseases", "size": 1},
                {"cluster_id": 1, "label": "Breast Cancer Genetics", "size": 1},
                {"cluster_id": -1, "label": "Unclustered", "size": 1},
            ]
        )
    )

    data.clear_caches()
    yield tmp_path
    data.clear_caches()


def fake_hits(query, strategy="hybrid", k=8):
    pmids = [103] if "BRCA1" in query else [101, 102, 103]
    return [Hit(f"{p}-0", p, PASSAGES[p], 1.0 - i / 10) for i, p in enumerate(pmids)][:k]


@pytest.fixture
def client(storage, monkeypatch):
    for module in (chat_api, search_api, questions_api):
        monkeypatch.setattr(module, "search", fake_hits)
    return TestClient(app)


# --- search & passages -------------------------------------------------------------------


def test_search_returns_numbered_sources_with_topics(client):
    r = client.get("/api/search", params={"q": "pompe", "strategy": "bm25", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "bm25" and body["k"] == 2
    first = body["results"][0]
    assert first["n"] == 1 and first["pmid"] == 101
    assert first["url"] == "https://pubmed.ncbi.nlm.nih.gov/101/"
    assert first["cluster_label"] == "Lysosomal Storage Diseases"


def test_search_validates_parameters(client):
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "strategy": "magic"}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422


def test_passage_with_noise_cluster_reports_nearest_topic(client):
    body = client.get("/api/passages/102").json()
    assert body["text"].startswith("GAA mutations")
    assert body["cluster_id"] == -1 and body["cluster_label"] == "Unclustered"
    assert body["nearest_cluster_label"] == "Lysosomal Storage Diseases"


def test_unknown_passage_is_404(client):
    assert client.get("/api/passages/424242").status_code == 404


def test_missing_pipeline_output_is_503(client):
    (settings.topics_dir / "map.json").unlink()
    data.clear_caches()
    r = client.get("/api/topics/map")
    assert r.status_code == 503
    assert "scripts.05_topics" in r.json()["detail"]


# --- questions ------------------------------------------------------------------------------


def test_list_questions_filters_and_paginates(client):
    body = client.get("/api/questions", params={"query": "brca1"}).json()
    assert body["total"] == 1 and body["items"][0]["qid"] == 2

    body = client.get("/api/questions", params={"usable_only": True}).json()
    assert body["total"] == 2

    body = client.get("/api/questions", params={"page": 2, "page_size": 2}).json()
    assert body["total"] == 3 and [q["qid"] for q in body["items"]] == [3]


def test_question_detail_without_eval_files(client):
    body = client.get("/api/questions/1").json()
    assert body["answer"] == "GAA deficiency."
    assert body["usable_relevant_ids"] == [101, 102] and body["n_gold"] == 2
    assert body["retrieval"] is None and body["evaluation"] is None
    assert client.get("/api/questions/77").status_code == 404


def test_run_question_marks_gold_sources(client, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda messages, **kw: llm.ChatResult("GAA [1].", 50, 5, 0.1))
    body = client.get("/api/questions/1/run", params={"k": 3}).json()
    assert body["answer"] == "GAA [1]."
    assert [s["in_gold"] for s in body["sources"]] == [True, True, False]
    assert body["hits"] == 2 and body["recall"] == 1.0
    assert body["gold_answer"] == "GAA deficiency."


def test_run_rejects_unknown_model(client):
    r = client.get("/api/questions/1/run", params={"model": "gpt-99"})
    assert r.status_code == 400


# --- chat (SSE) ---------------------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_streams_sources_tokens_and_done(client, monkeypatch):
    seen = {}

    def fake_stream(messages, *, model=None, think=None, stats=None, **kw):
        seen["messages"], seen["model"] = messages, model
        yield "Pompe is caused by "
        yield "GAA deficiency [1]."
        stats.prompt_tokens, stats.output_tokens, stats.seconds = 300, 12, 0.5

    monkeypatch.setattr(llm, "stream_chat", fake_stream)
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    with client.stream(
        "POST", "/api/chat", json={"message": "What causes Pompe disease?", "history": history}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode())

    names = [e for e, _ in events]
    assert names == ["sources", "token", "token", "done"]
    assert events[0][1]["sources"][0]["pmid"] == 101
    assert "".join(p["text"] for e, p in events if e == "token").endswith("GAA deficiency [1].")
    assert events[-1][1]["output_tokens"] == 12 and events[-1][1]["sources"]
    # system + 2 history messages + new user turn containing numbered sources
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]
    assert "[1] (PMID 101)" in seen["messages"][-1]["content"]
    assert seen["model"] == settings.chat_model


def test_chat_reports_generation_error_in_stream(client, monkeypatch):
    def broken(messages, **kw):
        yield "partial"
        raise RuntimeError("model crashed")

    monkeypatch.setattr(llm, "stream_chat", broken)
    with client.stream("POST", "/api/chat", json={"message": "x"}) as r:
        events = parse_sse(r.read().decode())
    assert [e for e, _ in events] == ["sources", "token", "error"]
    assert "model crashed" in events[-1][1]["message"]


def test_chat_validates_request(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x", "k": 99}).status_code == 422
    assert client.post("/api/chat", json={"message": "x", "model": "other"}).status_code == 400


def test_cors_allows_vite_dev_server(client):
    r = client.options(
        "/api/search",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


# --- topics, eval, health -------------------------------------------------------------------------


def test_topics_endpoints(client):
    assert len(client.get("/api/topics/map").json()) == 3
    assert client.get("/api/topics/clusters").json()[0]["label"] == "Lysosomal Storage Diseases"


def test_eval_results_empty_then_populated(client):
    body = client.get("/api/eval/results").json()
    assert body["retrieval"] is None and body["answer_records"] == []

    metrics = {"recall@10": 0.5, "mrr": 0.5, "hit@10": 1.0}
    (settings.eval_dir / "retrieval_summary.json").write_text(
        json.dumps({"best_strategy": "hybrid", "strategies": {"hybrid": metrics}})
    )
    (settings.eval_dir / "retrieval_questions.json").write_text(
        json.dumps(
            [
                {
                    "qid": q,
                    "question": f"Q{q}",
                    "n_gold": 2,
                    "strategies": {"hybrid": {"pmids": [1], "recall@10": r, "mrr": r}},
                }
                for q, r in [(1, 1.0), (2, 0.0)]
            ]
        )
    )
    judge = {"ok": True, "reason": "r", "correctness": 4, "groundedness": 5}
    record = {
        "qid": 1,
        "question": "Q1",
        "gold_answer": "G",
        "hits": 1,
        "retrieved": [{}],
        "rag": {"answer": "A [1]", "seconds": 1.0, "judge": judge},
        "no_rag": {"answer": "B", "seconds": 0.5, "judge": {**judge, "groundedness": None}},
    }
    (settings.eval_dir / "answers.jsonl").write_text(json.dumps(record) + "\n")
    data.clear_caches()

    body = client.get("/api/eval/results").json()
    assert body["worst_retrieval"]["strategy"] == "hybrid"
    assert [q["qid"] for q in body["worst_retrieval"]["questions"]] == [2, 1]
    assert "pmids" not in body["worst_retrieval"]["questions"][0]
    assert body["answer_records"][0]["rag"]["correctness"] == 4


def test_health_reports_down_when_ollama_unreachable(client, monkeypatch):
    monkeypatch.setattr(health_api, "installed_models", lambda: None)
    body = client.get("/api/health").json()
    assert body["status"] == "down" and body["ollama"] is False
    assert any("not reachable" in p for p in body["problems"])


def test_health_ok_when_everything_present(client, monkeypatch):
    monkeypatch.setattr(
        health_api,
        "installed_models",
        lambda: {settings.chat_model, settings.fallback_chat_model, settings.embed_model},
    )

    class FakeCollection:
        def count(self):
            return 3

    monkeypatch.setattr(health_api, "get_collection", lambda: FakeCollection())
    settings.chroma_dir.mkdir(parents=True)
    settings.bm25_dir.mkdir(parents=True)
    (settings.bm25_dir / "chunk_ids.json").write_text("[]")
    (settings.eval_dir / "retrieval_summary.json").write_text("{}")

    body = client.get("/api/health").json()
    assert body["status"] == "ok", body["problems"]
    assert body["collection_count"] == 3 and all(body["models"].values())
