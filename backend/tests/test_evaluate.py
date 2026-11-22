import importlib
import json

import pandas as pd
import pytest

from app.core import prompts
from app.core.llm import ChatResult
from app.core.metrics import (
    hit_at_k,
    mean_metrics,
    question_metrics,
    recall_at_k,
    reciprocal_rank,
)
from app.core.retrieve import Hit
from app.models import CorrectnessVerdict, RagVerdict

evaluate = importlib.import_module("scripts.06_evaluate")

# --- metrics -------------------------------------------------------------------------------


def test_recall_at_k():
    retrieved = [5, 1, 9, 2, 7]
    gold = {1, 2, 3, 4}
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 0.25
    assert recall_at_k(retrieved, gold, 5) == 0.5
    assert recall_at_k(retrieved, gold, 100) == 0.5


def test_recall_rejects_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 1)


def test_hit_at_k():
    assert hit_at_k([5, 1], {1}, 1) == 0.0
    assert hit_at_k([5, 1], {1}, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0
    assert reciprocal_rank([7, 8, 3], {3, 4}) == pytest.approx(1 / 3)
    assert reciprocal_rank([7, 8], {3}) == 0.0
    assert reciprocal_rank([], {3}) == 0.0


def test_question_and_mean_metrics():
    m1 = question_metrics([1, 2], [1])
    m2 = question_metrics([9, 1], [1, 2])
    assert m1["recall@1"] == 1.0 and m1["mrr"] == 1.0 and m1["hit@10"] == 1.0
    assert m2["recall@1"] == 0.0 and m2["recall@5"] == 0.5 and m2["mrr"] == 0.5
    avg = mean_metrics([m1, m2])
    assert avg["mrr"] == 0.75
    assert avg["recall@5"] == 0.75


# --- prompts -------------------------------------------------------------------------------


def hit(pmid: int, text: str = "text") -> Hit:
    return Hit(chunk_id=f"{pmid}-0", pmid=pmid, text=text, score=1.0)


def test_sources_are_numbered_with_pmids():
    text = prompts.format_sources([hit(111, "Alpha."), hit(222, "Beta.")])
    assert text == "[1] (PMID 111)\nAlpha.\n\n[2] (PMID 222)\nBeta."


def test_rag_prompt_contains_rules_sources_and_question():
    messages = prompts.rag_messages("Q?", [hit(1, "S1")], hint="Short.")
    assert messages[0]["role"] == "system"
    assert "ONLY" in messages[0]["content"] and "[1]" in messages[0]["content"]
    assert messages[1]["content"].endswith("Question: Q?\n\nShort.")
    assert "[1] (PMID 1)\nS1" in messages[1]["content"]


def test_judge_prompts():
    rag = prompts.judge_rag_messages("Q?", "REF", "ANS", [hit(1, "SRC")])[1]["content"]
    assert all(s in rag for s in ("REF", "ANS", "SRC", "groundedness"))
    no_rag = prompts.judge_no_rag_messages("Q?", "REF", "ANS")[1]["content"]
    assert "groundedness" not in no_rag and "REF" in no_rag


# --- judge parsing and retry --------------------------------------------------------------------


class FakeChat:
    """Returns queued replies; records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = self.replies.pop(0) if self.replies else "default answer"
        return ChatResult(text=text, prompt_tokens=100, output_tokens=20, seconds=0.1)


VALID_RAG = json.dumps({"reason": "ok", "correctness": 4, "groundedness": 5})


def test_judge_parses_valid_json_and_sends_schema():
    fake = FakeChat([VALID_RAG])
    verdict = evaluate.judge([], RagVerdict, fake)
    assert verdict == {
        "ok": True,
        "attempts": 1,
        "reason": "ok",
        "correctness": 4,
        "groundedness": 5,
    }
    assert fake.calls[0][1]["format"] == RagVerdict.model_json_schema()


def test_judge_retries_once_on_invalid_json():
    fake = FakeChat(["not json", VALID_RAG])
    verdict = evaluate.judge([], RagVerdict, fake)
    assert verdict["ok"] and verdict["attempts"] == 2
    assert len(fake.calls) == 2


def test_judge_rejects_out_of_range_scores_then_gives_up():
    bad = json.dumps({"reason": "x", "correctness": 9, "groundedness": 5})
    fake = FakeChat([bad, "{}"])
    verdict = evaluate.judge([], RagVerdict, fake)
    assert verdict["ok"] is False and verdict["attempts"] == 2 and verdict["error"]


def test_correctness_verdict_needs_no_groundedness():
    fake = FakeChat([json.dumps({"reason": "r", "correctness": 2})])
    assert evaluate.judge([], CorrectnessVerdict, fake)["correctness"] == 2


# --- answer evaluation loop ------------------------------------------------------------------


@pytest.fixture
def questions():
    return pd.DataFrame(
        {
            "qid": [10, 11, 12],
            "question": ["Q10?", "Q11?", "Q12?"],
            "answer": ["A10", "A11", "A12"],
            "gold": [[1], [2], [3]],
        }
    )


@pytest.fixture
def fake_search(monkeypatch):
    def search_many(queries, strategy, k):
        # Q10 retrieves its gold article 1; others retrieve unrelated article 99.
        return [[hit(1 if q == "Q10?" else 99)] for q in queries]

    monkeypatch.setattr(evaluate, "search_many", search_many)


def scripted_chat():
    """answer, judge, answer, judge ... for every question."""
    replies = []
    for _ in range(3):
        replies += ["rag answer [1]", VALID_RAG, "plain answer", '{"reason":"r","correctness":2}']
    return FakeChat(replies)


def test_evaluate_answers_writes_records(questions, fake_search, tmp_path):
    out = tmp_path / "answers.jsonl"
    records = evaluate.evaluate_answers(questions, 3, "hybrid", 1, scripted_chat(), out)

    assert len(records) == 3
    assert len(out.read_text().splitlines()) == 3
    r = next(r for r in records if r["qid"] == 10)
    assert r["hits"] == 1 and r["retrieved"][0]["in_gold"] is True
    assert r["rag"]["answer"] == "rag answer [1]"
    assert r["rag"]["judge"]["groundedness"] == 5
    assert r["no_rag"]["judge"]["correctness"] == 2
    assert r["config"]["strategy"] == "hybrid"


def test_evaluate_answers_resumes(questions, fake_search, tmp_path):
    out = tmp_path / "answers.jsonl"
    evaluate.evaluate_answers(questions, 2, "hybrid", 1, scripted_chat(), out)

    fake = scripted_chat()
    records = evaluate.evaluate_answers(questions, 3, "hybrid", 1, fake, out)

    assert len(records) == 3
    assert len(fake.calls) == 4  # only the one new question: 2 answers + 2 judges


def test_changed_config_is_not_reused(questions, fake_search, tmp_path):
    out = tmp_path / "answers.jsonl"
    evaluate.evaluate_answers(questions, 3, "hybrid", 1, scripted_chat(), out)
    fake = scripted_chat()
    evaluate.evaluate_answers(questions, 3, "bm25", 1, fake, out)
    assert len(fake.calls) == 12


def test_summary(questions, fake_search, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.settings, "storage_dir", tmp_path)
    records = evaluate.evaluate_answers(
        questions, 3, "hybrid", 1, scripted_chat(), tmp_path / "a.jsonl"
    )
    s = evaluate.summarize_answers(records, seconds=1.0)
    assert s["rag"]["correctness"] == 4 and s["rag"]["groundedness"] == 5
    assert s["no_rag"]["correctness"] == 2
    assert s["rag_correctness_by_retrieval"]["with_gold_source"]["n"] == 1
    assert (tmp_path / "eval" / "answers_summary.json").exists()
