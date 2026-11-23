"""Browse BioASQ questions and run one through RAG.

GET /api/questions?query=&page=&page_size=&usable_only=
GET /api/questions/{qid}
GET /api/questions/{qid}/run?strategy=&k=
"""

import time

from fastapi import APIRouter, HTTPException, Query

from app.api.common import resolve_model, to_sources
from app.config import settings
from app.core import data, llm, prompts
from app.core.metrics import recall_at_k
from app.core.retrieve import Strategy, search
from app.models import QuestionDetail, QuestionPage, QuestionRun, QuestionSummary, RunSource

router = APIRouter()


def _question(qid: int):
    df = data.questions()
    if qid not in df.index:
        raise HTTPException(404, f"No question with qid {qid}")
    return df.loc[qid]


def _summary(row) -> QuestionSummary:
    return QuestionSummary(
        qid=int(row.qid),
        question=row.question,
        n_gold=len(row.usable_relevant_ids),
        has_usable=bool(row.has_usable),
    )


@router.get("/questions")
def list_questions(
    query: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    usable_only: bool = False,
) -> QuestionPage:
    df = data.questions()
    if usable_only:
        df = df[df["has_usable"]]
    if query.strip():
        df = df[df["question"].str.contains(query.strip(), case=False, regex=False)]
    start = (page - 1) * page_size
    return QuestionPage(
        total=len(df),
        page=page,
        page_size=page_size,
        items=[_summary(row) for row in df.iloc[start : start + page_size].itertuples()],
    )


@router.get("/questions/{qid}")
def get_question(qid: int) -> QuestionDetail:
    row = _question(qid)
    try:
        retrieval = data.retrieval_questions().get(qid, {}).get("strategies")
    except data.MissingDataError:
        retrieval = None
    try:
        evaluation = data.answer_records().get(qid)
    except data.MissingDataError:
        evaluation = None
    return QuestionDetail(
        **_summary(row).model_dump(),
        answer=row.answer,
        relevant_ids=row.relevant_ids,
        usable_relevant_ids=row.usable_relevant_ids,
        retrieval=retrieval,
        evaluation=evaluation,
    )


@router.get("/questions/{qid}/run")
def run_question(
    qid: int,
    strategy: Strategy = "hybrid",
    k: int = Query(settings.default_k, ge=1, le=settings.max_chat_k),
    model: str | None = None,
) -> QuestionRun:
    row = _question(qid)
    model_name = resolve_model(model)
    gold = set(row.usable_relevant_ids)

    start = time.perf_counter()
    hits = search(row.question, strategy, k)
    result = llm.chat(prompts.rag_messages(row.question, hits), model=model_name)

    sources = [RunSource(**s.model_dump(), in_gold=s.pmid in gold) for s in to_sources(hits)]
    return QuestionRun(
        qid=qid,
        question=row.question,
        gold_answer=row.answer,
        gold_ids=sorted(gold),
        strategy=strategy,
        k=k,
        sources=sources,
        hits=sum(s.in_gold for s in sources),
        recall=recall_at_k([h.pmid for h in hits], gold, k) if gold else None,
        answer=result.text.strip(),
        model=model_name,
        seconds=round(time.perf_counter() - start, 2),
    )
