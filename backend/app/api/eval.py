"""GET /api/eval/results: stored evaluation results for the dashboard."""

from fastapi import APIRouter, Query

from app.core import data

router = APIRouter()


def _optional(loader):
    try:
        return loader()
    except data.MissingDataError:
        return None


def _compact_answer(record: dict) -> dict:
    def side(name: str) -> dict:
        judge = record[name]["judge"]
        return {
            "answer": record[name]["answer"],
            "correctness": judge.get("correctness"),
            "groundedness": judge.get("groundedness"),
            "reason": judge.get("reason"),
            "seconds": record[name]["seconds"],
        }

    return {
        "qid": record["qid"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "hits": record["hits"],
        "n_sources": len(record["retrieved"]),
        "rag": side("rag"),
        "no_rag": side("no_rag"),
    }


def _worst_retrieval(strategy: str, limit: int) -> list[dict]:
    rows = _optional(data.retrieval_questions)
    if not rows:
        return []
    scored = [
        {
            "qid": r["qid"],
            "question": r["question"],
            "n_gold": r["n_gold"],
            **{k: v for k, v in r["strategies"][strategy].items() if k != "pmids"},
        }
        for r in rows.values()
    ]
    scored.sort(key=lambda r: (r["recall@10"], r["mrr"], -r["n_gold"]))
    return scored[:limit]


@router.get("/eval/results")
def eval_results(worst_limit: int = Query(50, ge=1, le=500)) -> dict:
    retrieval = _optional(data.retrieval_summary)
    strategy = retrieval["best_strategy"] if retrieval else None
    answers = _optional(data.answer_records)
    return {
        "retrieval": retrieval,
        "answers": _optional(data.answers_summary),
        "answer_records": [_compact_answer(r) for r in answers.values()] if answers else [],
        "worst_retrieval": {
            "strategy": strategy,
            "questions": _worst_retrieval(strategy, worst_limit) if strategy else [],
        },
    }
