# BioASQ Local RAG Explorer

A fully local app for studying the `rag-mini-bioasq` biomedical dataset with an LLM: chat with cited sources, semantic search, a topic map of the corpus, and a retrieval evaluation dashboard. Everything runs on your machine through Ollama; no cloud APIs.

The full implementation plan is in [`bioasq-rag-plan.md`](bioasq-rag-plan.md).

> This is a research and study tool. It is not medical advice.

## Requirements

- macOS on Apple Silicon (developed on an M4 Pro, 24 GB)
- [Homebrew](https://brew.sh)
- About 30 GB of free disk space for models, data and indexes

## Setup

```bash
# 1. Tools
brew install ollama uv node

# 2. Start Ollama (background service, restarts at login)
brew services start ollama

# 3. Models (~23 GB download)
ollama pull qwen3-embedding:0.6b
ollama pull gemma4:26b-a4b-it-qat
ollama pull gemma4:12b-it-qat

# 4. Configuration
cp .env.example .env

# 5. Backend dependencies
cd backend && uv sync && cd ..

# 6. Frontend dependencies
cd frontend && npm install && cd ..
```

Check the setup:

```bash
ollama list
cd backend && uv run python -c "import chromadb, bm25s, umap"
cd frontend && npm run dev      # http://localhost:5173
```

## Models

| Role | Ollama tag | Size |
|---|---|---|
| Chat / answers / judge | `gemma4:26b-a4b-it-qat` | 15.6 GB |
| Chat fallback | `gemma4:12b-it-qat` | 7.2 GB |
| Embeddings | `qwen3-embedding:0.6b` | 0.6 GB |

**Change from the plan:** the plan names `gemma4:26b` and `gemma4:12b`. The default `gemma4:26b` tag is 18.7 GB, which is too large to run fully on the GPU of a 24 GB Mac (macOS reserves roughly a third of unified memory for the system). The QAT (quantization-aware trained) 4-bit variants are used instead: they fit the memory budget and lose less quality than standard 4-bit quantization. The plan also names React 18; the Vite template now ships React 19, which is used as-is.

If you see memory pressure turn yellow or red in Activity Monitor, set `CHAT_MODEL=gemma4:12b-it-qat` in `.env`.

Gemma 4 reasons ("thinks") before answering by default, which makes short answers about 10x slower. It is turned off with `CHAT_THINK=false`.

Measured on the M4 Pro (24 GB), `num_ctx=8192`:

| Model | Memory | On GPU | Prompt | Generation |
|---|---|---|---|---|
| `gemma4:26b-a4b-it-qat` | 14.9 GB | 100% | ~4,200 tok/s | ~58 tok/s |
| `gemma4:12b-it-qat` | 7.4 GB | 100% | ~4,000 tok/s | ~28 tok/s |
| `qwen3-embedding:0.6b` | 1.0 GB | 100% | ~13 chunks/s incl. Chroma writes | — |

All settings (models, paths, chunk sizes, top-k) live in `backend/app/config.py` and can be overridden in `.env`.

## Data pipeline

Pipeline scripts run as modules from `backend/`:

```bash
cd backend
uv run python -m scripts.01_download   # data/raw/*.parquet (~26 MB, ~10 s)
uv run python -m scripts.02_prepare    # data/processed/*.parquet (~2 s)
uv run python -m scripts.03_embed      # storage/chroma + embeddings.npy (~46 min, resumable)
uv run python -m scripts.04_bm25       # storage/bm25 keyword index (~3 s)
uv run python -m scripts.05_topics     # storage/topics map + labels (~6 min)
uv run python -m scripts.06_evaluate   # storage/eval (retrieval ~4 min + answers ~35-50 min)
```

`03_embed` is resumable: if it stops, rerun it and it continues from the last stored batch. Measured run: 36,076 chunks in 45.7 min (13.1 chunks/s), 515 MB Chroma collection, 148 MB `embeddings.npy`.

`02_prepare` result:

| | Count |
|---|---|
| Passages in dataset | 40,221 |
| Dropped: missing abstract (`"nan"`) | 12,220 |
| Dropped: stub, empty or citation-only (e.g. `"1."`, a bare journal reference) | 29 |
| **Passages kept** | **27,972** |
| Kept whole (≤ 2,000 chars) / split | 24,849 / 3,123 |
| **Chunks** | **36,076** |
| Questions | 4,719 |
| **Usable questions** (≥ 1 non-empty gold passage) | **4,386** |

Notes on the data, compared with the plan:

- The plan expects 4,387 usable questions. The extra one has a single relevant passage whose text is just `"1."`, so it is excluded.
- The plan's median passage length (~1,160 chars) includes the empty `"nan"` rows. Among real abstracts the median is ~1,450 chars, and 12,925 exceed the plan's 1,500-char "keep whole" limit. To keep the plan's intent ("most passages fit in one chunk") the limit is `MAX_WHOLE_PASSAGE_CHARS=2000`, so only the ~3,100 longest abstracts are split (36k chunks instead of 52k). Set it back to 1,500 in `.env` to compare.
- Long passages are split on sentence boundaries into balanced chunks of at most 1,200 chars, each starting with ~200 chars (whole sentences) from the previous chunk.

## Evaluation

`06_evaluate` has two parts (`--part retrieval|answers|all`):

1. **Retrieval** on all 4,386 usable questions for dense, BM25 and hybrid search. Metrics are at the PubMed-ID level: recall@1/5/10/20, MRR (first gold article within the top 20) and hit-rate@10. No LLM needed.
2. **Answers** on a fixed random sample (`--sample`, default 100, seed 42) using the best retrieval strategy and `k=8` sources. Each question is answered with RAG and without retrieval, and each answer is scored 1–5 by an LLM judge: correctness against the expert answer, and (RAG only) groundedness in the sources. The judge must return JSON matching a pydantic schema and is retried once if invalid. Records are appended to `answers.jsonl` as they finish, so an interrupted run resumes.

Retrieval results (4,386 questions):

| Strategy | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Hit@10 |
|---|---|---|---|---|---|---|
| Dense | 0.242 | 0.524 | 0.654 | 0.753 | 0.806 | 0.908 |
| BM25 | 0.254 | 0.562 | 0.709 | 0.813 | 0.829 | 0.938 |
| **Hybrid** | 0.252 | **0.576** | **0.724** | **0.835** | **0.840** | **0.943** |

Answer results (100 questions, hybrid retrieval, k=8, `gemma4:26b-a4b-it-qat`, ~50 min):

| | Correctness (1–5) | Correct (score ≥ 4) | Groundedness (1–5) | Seconds per answer |
|---|---|---|---|---|
| **RAG** | **4.68** | **89%** | 5.00 | 11.2 |
| No RAG | 3.92 | 69% | — | 6.6 |

- RAG rescued 26 questions the model got wrong on its own; the model alone did better on 6.
- RAG correctness was 4.74 when a gold article was among the sources (91 questions) and 4.11 when none was (9).
- 99 of 100 RAG answers cite sources (2.7 distinct sources on average). In 4 the model said the sources did not contain the answer instead of guessing.
- The judge returned valid JSON on the first try for all 200 judgments.

Caveats:

- **Groundedness is at its ceiling:** the judge gave 5 to every RAG answer, so this score does not discriminate. A stricter, claim-by-claim check or a different judge model would be needed to measure it.
- **Self-grading bias:** the judge is the same local model that wrote the answers, which tends to inflate scores. Use the scores to compare RAG with no-RAG, not as absolute quality.
- **Incomplete gold sets:** BioASQ gold lists are not exhaustive, so a "miss" can still be a relevant paper. Recall is also capped for questions with more gold articles than `k`.

## Topic map

`05_topics` places every abstract on a 2D map and groups them into topics:

1. One vector per article (the normalized mean of its chunk vectors).
2. UMAP to 2D for the map (`n_neighbors=15`, `min_dist=0.1`, cosine, fixed seed).
3. A second UMAP to 10D, clustered with HDBSCAN (`min_cluster_size=150`, `min_samples=10`). Clustering in 10D keeps groups apart that a 2D projection would merge.
4. The LLM names each cluster from its 8 most typical abstracts (closest to the cluster's mean direction) with a 2–5 word label and a one-sentence description, validated against a JSON schema.

Projections are cached in `storage/topics/projection.npz`; use `--fresh` to recompute them.

Result: **56 topics** for 27,972 abstracts, e.g. "Genetic Syndromes and Dysmorphology" (1,118), "Emerging Viral Pathogens and Vaccines" (857), "Alzheimer's Amyloid-Beta Therapeutics" (367), "CGRP-Targeted Migraine Therapies" (180).

Notes:

- **~39% of abstracts are unclustered** (HDBSCAN noise). The corpus spans all of biomedicine with many small topics, and noise stays at 32–39% for any setting that gives 25–85 clusters. Each unclustered point also stores its `nearest_cluster`.
- **Labels describe the cluster's core.** The LLM sees only the most central abstracts, so a mixed cluster can get a narrow label (e.g. "HER2-Targeted Prostate Cancer Therapies" also contains HER2 breast cancer papers).

## Backend API

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000   # docs at http://localhost:8000/docs
```

| Endpoint | Description |
|---|---|
| `POST /api/chat` | Body `{message, history, strategy, k, model?, think?}`. Retrieves sources, then streams the answer over Server-Sent Events: `sources` (first), `token` (text pieces), `done` (sources, model, token counts, time) or `error`. |
| `GET /api/search?q=&strategy=&k=` | Retrieval only (`dense`, `bm25` or `hybrid`), no LLM. |
| `GET /api/passages/{pmid}` | Full abstract, PubMed link and its topic (or nearest topic if unclustered). |
| `GET /api/questions?query=&page=&page_size=&usable_only=` | Browse and search the BioASQ questions. |
| `GET /api/questions/{qid}` | Gold answer, gold PMIDs, stored retrieval metrics and answer-evaluation record. |
| `GET /api/questions/{qid}/run?strategy=&k=` | Run one question: sources marked as in/out of the gold set, generated answer, gold answer. |
| `GET /api/topics/map`, `GET /api/topics/clusters` | Topic map points and cluster labels. |
| `GET /api/eval/results` | Retrieval and answer summaries, per-question answer records, worst retrieval questions. |
| `GET /api/health` | Ollama reachable, models installed, index and result files present (`ok` / `degraded` / `down`). |

Missing pipeline outputs return `503` naming the script to run. CORS allows the Vite dev server (`localhost:5173`); large responses are gzipped (topic map 2.4 MB → 390 KB).

Measured: search ~0.4 s warm; chat first token ~4 s warm (~16 s when the chat model must load first), ~7 s for a short answer. Chat retrieves sources for the latest message only; earlier turns are sent as plain text (last 6 messages).

## Project layout

```
backend/    FastAPI app, pipeline scripts, tests (Python 3.12, uv)
frontend/   Vite + React + TypeScript + Tailwind UI
data/       raw and processed dataset files (gitignored)
storage/    Chroma DB, BM25 index, topic map, eval results (gitignored)
```

## Dataset attribution

This project uses [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq), licensed under [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). It is derived from the [BioASQ](http://bioasq.org/) challenge data, built on PubMed abstracts.
