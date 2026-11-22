"""Retrieval metrics at the PubMed-ID level."""

from collections.abc import Collection, Sequence

KS = (1, 5, 10, 20)


def recall_at_k(retrieved: Sequence[int], gold: Collection[int], k: int) -> float:
    """Share of gold articles found in the top k."""
    if not gold:
        raise ValueError("gold set is empty")
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def hit_at_k(retrieved: Sequence[int], gold: Collection[int], k: int) -> float:
    """1.0 if at least one gold article is in the top k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: Sequence[int], gold: Collection[int]) -> float:
    """1 / rank of the first gold article (ranks start at 1); 0 if none retrieved."""
    gold_set = set(gold)
    for rank, pmid in enumerate(retrieved, start=1):
        if pmid in gold_set:
            return 1.0 / rank
    return 0.0


def question_metrics(retrieved: Sequence[int], gold: Collection[int]) -> dict[str, float]:
    """All metrics for one question. MRR is computed over the full retrieved list."""
    metrics = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in KS}
    metrics["hit@10"] = hit_at_k(retrieved, gold, 10)
    metrics["mrr"] = reciprocal_rank(retrieved, gold)
    return metrics


def mean_metrics(per_question: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {key: sum(m[key] for m in per_question) / len(per_question) for key in keys}
