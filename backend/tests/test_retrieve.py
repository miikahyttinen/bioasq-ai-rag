import pandas as pd
import pytest

from app.config import settings
from app.core.bm25 import build_index, load_index
from app.core.retrieve import STRATEGIES, Hit, dedupe_by_pmid, rrf_fuse, search


def hit(pmid: int, chunk: int = 0, score: float = 1.0) -> Hit:
    return Hit(chunk_id=f"{pmid}-{chunk}", pmid=pmid, text=f"text {pmid}-{chunk}", score=score)


# --- deduplication --------------------------------------------------------------


def test_dedupe_keeps_best_chunk_per_article_in_order():
    hits = [hit(1, 2, 0.9), hit(2, 0, 0.8), hit(1, 0, 0.7), hit(3, 1, 0.6), hit(2, 1, 0.5)]
    assert [h.chunk_id for h in dedupe_by_pmid(hits)] == ["1-2", "2-0", "3-1"]


def test_dedupe_empty():
    assert dedupe_by_pmid([]) == []


# --- reciprocal rank fusion -------------------------------------------------------


def test_rrf_scores_match_formula():
    dense = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(1)]
    fused = {h.pmid: h.score for h in rrf_fuse([dense, keyword], rrf_k=60)}

    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_over_a_single_top_rank():
    # 9 and 7 are each #1 in one list; 5 is #2 in both -> 5 wins, then 7/9 tie by pmid.
    fused = rrf_fuse([[hit(9), hit(5)], [hit(7), hit(5)]], rrf_k=60)
    assert fused[0].pmid == 5
    assert [h.pmid for h in fused] == [5, 7, 9]


def test_rrf_ties_broken_by_best_rank_then_pmid():
    # 2 and 1 each appear once at rank 1 -> equal scores; lower pmid first.
    fused = rrf_fuse([[hit(2)], [hit(1)]], rrf_k=60)
    assert [h.pmid for h in fused] == [1, 2]


def test_rrf_keeps_chunk_from_best_ranking():
    dense = [hit(8), hit(4, chunk=0)]  # pmid 4 at rank 2 via chunk 0
    keyword = [hit(4, chunk=3)]  # pmid 4 at rank 1 via chunk 3
    fused = {h.pmid: h for h in rrf_fuse([dense, keyword], rrf_k=60)}
    assert fused[4].chunk_id == "4-3"


def test_rrf_single_ranking_preserves_order():
    ranking = [hit(3), hit(1), hit(2)]
    assert [h.pmid for h in rrf_fuse([ranking], rrf_k=60)] == [3, 1, 2]


def test_rrf_empty_inputs():
    assert rrf_fuse([[], []], rrf_k=60) == []


# --- BM25 index ----------------------------------------------------------------------


@pytest.fixture
def tiny_bm25(tmp_path):
    chunks = pd.DataFrame(
        {
            "chunk_id": ["1-0", "2-0", "3-0", "3-1"],
            "pmid": [1, 2, 3, 3],
            "text": [
                "Pompe disease is caused by mutations in the GAA gene.",
                "BRCA1 mutations increase breast cancer risk.",
                "IL-6 drives inflammation in rheumatoid arthritis.",
                "Vitamin D deficiency and type 2 diabetes.",
            ],
        }
    )
    build_index(chunks["chunk_id"].tolist(), chunks["text"].tolist(), tmp_path)
    return load_index(tmp_path, chunks)


def test_bm25_ranks_matching_chunk_first(tiny_bm25):
    (results,) = tiny_bm25.search_many(["what causes pompe disease"], n=4)
    assert tiny_bm25.chunk_ids[results[0][0]] == "1-0"


def test_bm25_stems_query_terms(tiny_bm25):
    (results,) = tiny_bm25.search_many(["mutated BRCA1"], n=4)
    assert tiny_bm25.chunk_ids[results[0][0]] == "2-0"


def test_bm25_keeps_single_character_tokens(tiny_bm25):
    (results,) = tiny_bm25.search_many(["vitamin D"], n=4)
    assert tiny_bm25.chunk_ids[results[0][0]] == "3-1"
    assert "d" in tiny_bm25.tokenizer.get_vocab_dict()


def test_bm25_drops_zero_score_results(tiny_bm25):
    assert tiny_bm25.search_many(["zebrafish"], n=4) == [[]]


# --- integration: real indexes + Ollama ----------------------------------------------


def _indexes_ready() -> bool:
    try:
        import ollama

        ollama.Client(host=settings.ollama_host).list()
    except Exception:
        return False
    return (settings.bm25_dir / "chunk_ids.json").exists() and settings.chroma_dir.exists()


@pytest.mark.skipif(not _indexes_ready(), reason="needs built indexes and running Ollama")
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_return_results_for_sample_question(strategy):
    hits = search("What causes Pompe disease?", strategy, k=5)
    assert len(hits) == 5
    assert len({h.pmid for h in hits}) == 5
    assert all(isinstance(h.pmid, int) and h.text for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert sum("pompe" in h.text.lower() for h in hits) >= 4
