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

## Project layout

```
backend/    FastAPI app, pipeline scripts, tests (Python 3.12, uv)
frontend/   Vite + React + TypeScript + Tailwind UI
data/       raw and processed dataset files (gitignored)
storage/    Chroma DB, BM25 index, topic map, eval results (gitignored)
```

## Dataset attribution

This project uses [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq), licensed under [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). It is derived from the [BioASQ](http://bioasq.org/) challenge data, built on PubMed abstracts.
