"""Evaluate retrieval (all usable questions) and answers (a fixed random sample).

  uv run python -m scripts.06_evaluate                     # both parts
  uv run python -m scripts.06_evaluate --part retrieval    # fast, no LLM
  uv run python -m scripts.06_evaluate --part answers --sample 20

Outputs in storage/eval/:
  retrieval_summary.json    mean metrics per strategy
  retrieval_questions.json  per-question top-20 PMIDs and metrics per strategy
  answers.jsonl             one record per sampled question (resumable)
  answers_summary.json      RAG vs no-RAG judge scores
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from app.config import settings
from app.core import prompts
from app.core.embed import unload_embed_model
from app.core.llm import ChatResult, chat, chat_json, unload
from app.core.metrics import mean_metrics, question_metrics
from app.core.retrieve import STRATEGIES, Hit, search_many
from app.models import CorrectnessVerdict, RagVerdict

RETRIEVAL_DEPTH = 20
ChatFn = Callable[..., ChatResult]


def load_usable_questions() -> pd.DataFrame:
    questions = pd.read_parquet(settings.processed_dir / "questions.parquet")
    usable = questions[questions["has_usable"]].reset_index(drop=True)
    usable["gold"] = usable["usable_relevant_ids"].map(lambda ids: [int(i) for i in ids])
    return usable


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# --- retrieval -------------------------------------------------------------------------


def evaluate_retrieval(questions: pd.DataFrame) -> dict:
    texts = questions["question"].tolist()
    per_question = [
        {"qid": int(q.qid), "question": q.question, "n_gold": len(q.gold), "strategies": {}}
        for q in questions.itertuples()
    ]
    summary: dict[str, dict] = {}

    for strategy in STRATEGIES:
        start = time.perf_counter()
        results = search_many(texts, strategy, RETRIEVAL_DEPTH)
        seconds = time.perf_counter() - start

        metrics = []
        for record, gold, hits in zip(per_question, questions["gold"], results, strict=True):
            pmids = [h.pmid for h in hits]
            m = question_metrics(pmids, gold)
            metrics.append(m)
            record["strategies"][strategy] = {"pmids": pmids, **m}
        summary[strategy] = {**mean_metrics(metrics), "seconds": round(seconds, 1)}

    best = max(STRATEGIES, key=lambda s: (summary[s]["recall@10"], summary[s]["mrr"]))
    result = {
        "n_questions": len(questions),
        "depth": RETRIEVAL_DEPTH,
        "embed_model": settings.embed_model,
        "best_strategy": best,
        "strategies": summary,
    }
    write_json(settings.eval_dir / "retrieval_summary.json", result)
    write_json(settings.eval_dir / "retrieval_questions.json", per_question)
    return result


def print_retrieval_table(result: dict) -> None:
    cols = ["recall@1", "recall@5", "recall@10", "recall@20", "mrr", "hit@10"]
    print(f"\nRetrieval on {result['n_questions']:,} questions (PubMed-ID level)\n")
    print(f"{'strategy':<9}" + "".join(f"{c:>11}" for c in cols) + f"{'time':>9}")
    for strategy, m in result["strategies"].items():
        row = "".join(f"{m[c]:>11.3f}" for c in cols)
        mark = "  <- best" if strategy == result["best_strategy"] else ""
        print(f"{strategy:<9}{row}{m['seconds']:>8.1f}s{mark}")


# --- answers ---------------------------------------------------------------------------


def judge(messages: list[dict], schema: type[BaseModel], chat_fn: ChatFn) -> dict:
    """Ask for JSON matching `schema`; retry once if it is invalid."""
    verdict, attempts, error = chat_json(messages, schema, chat_fn)
    if verdict is None:
        return {"ok": False, "attempts": attempts, "error": error}
    return {"ok": True, "attempts": attempts, **verdict.model_dump()}


def generate(messages: list[dict], chat_fn: ChatFn) -> dict:
    result = chat_fn(messages, temperature=0.0)
    if result.prompt_tokens >= settings.num_ctx - 64:
        print(f"\nWarning: prompt used {result.prompt_tokens} tokens; context may be truncated")
    return {
        "answer": result.text.strip(),
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(result.seconds, 2),
    }


def answer_question(q, sources: list[Hit], chat_fn: ChatFn) -> dict:
    gold = set(q.gold)
    hint = prompts.EVAL_LENGTH_HINT

    rag = generate(prompts.rag_messages(q.question, sources, hint), chat_fn)
    rag["judge"] = judge(
        prompts.judge_rag_messages(q.question, q.answer, rag["answer"], sources),
        RagVerdict,
        chat_fn,
    )
    no_rag = generate(prompts.no_rag_messages(q.question, hint), chat_fn)
    no_rag["judge"] = judge(
        prompts.judge_no_rag_messages(q.question, q.answer, no_rag["answer"]),
        CorrectnessVerdict,
        chat_fn,
    )
    return {
        "qid": int(q.qid),
        "question": q.question,
        "gold_answer": q.answer,
        "gold_ids": sorted(gold),
        "retrieved": [
            {"pmid": h.pmid, "chunk_id": h.chunk_id, "score": h.score, "in_gold": h.pmid in gold}
            for h in sources
        ],
        "hits": sum(h.pmid in gold for h in sources),
        "rag": rag,
        "no_rag": no_rag,
    }


def run_config(strategy: str, k: int) -> dict:
    return {"model": settings.chat_model, "strategy": strategy, "k": k, "seed": settings.eval_seed}


def load_done(path: Path, config: dict) -> dict[int, dict]:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("config") == config:
            done[record["qid"]] = record
    return done


def evaluate_answers(
    questions: pd.DataFrame,
    sample_size: int,
    strategy: str,
    k: int,
    chat_fn: ChatFn = chat,
    out_path: Path | None = None,
) -> list[dict]:
    out_path = out_path or settings.eval_dir / "answers.jsonl"
    config = run_config(strategy, k)
    sample = questions.sample(min(sample_size, len(questions)), random_state=settings.eval_seed)

    done = load_done(out_path, config)
    todo = sample[~sample["qid"].isin(done)]
    print(
        f"\nAnswer evaluation: {len(sample)} questions ({len(done)} already done), "
        f"model {config['model']}, {strategy} retrieval, k={k}"
    )

    records = [done[qid] for qid in sample["qid"] if qid in done]
    if todo.empty:
        return records

    # Retrieve everything first, then free the embedding model before the LLM loads.
    all_sources = search_many(todo["question"].tolist(), strategy, k)
    if chat_fn is chat:
        unload_embed_model()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as out, tqdm(total=len(todo), unit="q") as progress:
        for q, sources in zip(todo.itertuples(), all_sources, strict=True):
            record = {**answer_question(q, sources, chat_fn), "config": config}
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
            progress.update(1)
    return records


def _avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize_answers(records: list[dict], seconds: float) -> dict:
    def scores(side: str, field: str) -> list:
        return [r[side]["judge"].get(field) for r in records]

    rag_correct = scores("rag", "correctness")
    with_hit = [c for r, c in zip(records, rag_correct, strict=True) if r["hits"]]
    without_hit = [c for r, c in zip(records, rag_correct, strict=True) if not r["hits"]]

    summary = {
        "n_questions": len(records),
        "config": records[0]["config"] if records else None,
        "rag": {
            "correctness": _avg(rag_correct),
            "groundedness": _avg(scores("rag", "groundedness")),
            "correctness_distribution": dict(sorted(Counter(rag_correct).items(), key=str)),
            "share_correct_4plus": _avg([c is not None and c >= 4 for c in rag_correct]),
            "judge_failures": sum(not r["rag"]["judge"]["ok"] for r in records),
            "mean_answer_seconds": _avg([r["rag"]["seconds"] for r in records]),
        },
        "no_rag": {
            "correctness": _avg(scores("no_rag", "correctness")),
            "correctness_distribution": dict(
                sorted(Counter(scores("no_rag", "correctness")).items(), key=str)
            ),
            "share_correct_4plus": _avg(
                [c is not None and c >= 4 for c in scores("no_rag", "correctness")]
            ),
            "judge_failures": sum(not r["no_rag"]["judge"]["ok"] for r in records),
            "mean_answer_seconds": _avg([r["no_rag"]["seconds"] for r in records]),
        },
        "rag_correctness_by_retrieval": {
            "with_gold_source": {"n": len(with_hit), "correctness": _avg(with_hit)},
            "without_gold_source": {"n": len(without_hit), "correctness": _avg(without_hit)},
        },
        "seconds": round(seconds, 1),
        "note": "Judged by the same local model that generated the answers (self-grading bias).",
    }
    write_json(settings.eval_dir / "answers_summary.json", summary)
    return summary


def print_answer_summary(s: dict) -> None:
    rag, no_rag, by = s["rag"], s["no_rag"], s["rag_correctness_by_retrieval"]
    print(f"\nAnswer quality on {s['n_questions']} questions (judge scores 1-5)\n")
    print(f"{'':<10}{'correct':>9}{'grounded':>10}{'>=4':>7}{'judge fail':>12}{'s/answer':>10}")
    print(
        f"{'RAG':<10}{rag['correctness']:>9}{rag['groundedness']:>10}"
        f"{rag['share_correct_4plus']:>7}{rag['judge_failures']:>12}{rag['mean_answer_seconds']:>10}"
    )
    print(
        f"{'no RAG':<10}{no_rag['correctness']:>9}{'-':>10}"
        f"{no_rag['share_correct_4plus']:>7}{no_rag['judge_failures']:>12}"
        f"{no_rag['mean_answer_seconds']:>10}"
    )
    w, wo = by["with_gold_source"], by["without_gold_source"]
    print(f"\nRAG correctness with a gold source retrieved: {w['correctness']} (n={w['n']})")
    print(f"RAG correctness without a gold source:       {wo['correctness']} (n={wo['n']})")


# --- main ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["retrieval", "answers", "all"], default="all")
    parser.add_argument("--sample", type=int, default=settings.eval_sample_size)
    parser.add_argument("--strategy", choices=STRATEGIES, help="default: best from retrieval")
    parser.add_argument("--k", type=int, default=settings.default_k)
    args = parser.parse_args()

    questions = load_usable_questions()

    if args.part in ("retrieval", "all"):
        result = evaluate_retrieval(questions)
        print_retrieval_table(result)

    if args.part in ("answers", "all"):
        strategy = args.strategy
        if strategy is None:
            summary_path = settings.eval_dir / "retrieval_summary.json"
            strategy = json.loads(summary_path.read_text())["best_strategy"]
        start = time.perf_counter()
        try:
            records = evaluate_answers(questions, args.sample, strategy, args.k)
        finally:
            unload()
        print_answer_summary(summarize_answers(records, time.perf_counter() - start))


if __name__ == "__main__":
    main()
