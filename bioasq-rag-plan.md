# BioASQ Local RAG Explorer — Implementation Plan

A fully local app for studying the `rag-mini-bioasq` biomedical dataset with an LLM: chat with cited sources, semantic search, a topic map of the corpus, and a retrieval evaluation dashboard.

**Target machine:** Apple M4 Pro, 24 GB unified memory, macOS. Everything runs locally; no cloud APIs.

---

## 0. Instructions for Claude Code

- Work phase by phase. Finish each phase's acceptance criteria before starting the next, and run the tests at the end of every phase.
- Use `uv` for all Python dependency management. Do not use pip directly.
- Before using any Ollama model tag in this plan, verify it exists with `ollama pull <tag>` or `ollama list`. Model tags change. If a tag is missing, pick the closest equivalent and note the change in `README.md`.
- Keep all configuration (model names, paths, chunk sizes, top-k) in `backend/app/config.py` using pydantic-settings, overridable from `.env`.
- Never commit `data/` or `storage/` (add to `.gitignore`).
- Memory budget is tight (24 GB shared between CPU and GPU). Never load the chat LLM and run bulk embedding at the same time. Use Ollama's `keep_alive` to unload models between stages.

---

## 1. Technologies

### Local AI runtime
| Technology | Purpose |
|---|---|
| **Ollama** | Runs the chat LLM and embedding model locally on Apple Silicon (Metal), exposes a local HTTP API |
| **Gemma 4 26B A4B** (4-bit, ~14–15 GB) | Main chat / answer-generation model. Mixture-of-experts, so it is fast for its size and well suited to document Q&A |
| **Gemma 4 12B** (4-bit, ~7 GB) | Fallback chat model if the 26B causes memory pressure or swapping |
| **qwen3-embedding:0.6b** (~1.5 GB) | Embedding model that turns passages and questions into vectors |

### Backend (Python 3.12)
| Technology | Purpose |
|---|---|
| **uv** | Python version, virtualenv and dependency management |
| **huggingface_hub** | Downloads the dataset files |
| **pandas + pyarrow** | Reads parquet files, cleaning |
| **ollama** (Python client) | Calls Ollama for embeddings and chat |
| **ChromaDB** (persistent, embedded) | Vector database stored on disk, no server |
| **bm25s** | Keyword (BM25) search for hybrid retrieval |
| **NumPy** | Vector math, score fusion |
| **umap-learn** | Projects embeddings to 2D for the topic map |
| **scikit-learn** | HDBSCAN clustering, metrics |
| **FastAPI + Uvicorn** | REST API and streaming chat endpoint (Server-Sent Events) |
| **pydantic / pydantic-settings** | Request/response models and config |
| **tqdm** | Progress bars for long ingestion jobs |
| **pytest** | Tests |
| **ruff** | Linting and formatting |

### Frontend
| Technology | Purpose |
|---|---|
| **Node.js (LTS) + npm** | JavaScript runtime and package manager |
| **Vite** | Dev server and build tool |
| **React 18 + TypeScript** | UI |
| **Tailwind CSS** | Styling |
| **React Router** | Page navigation |
| **TanStack Query** | Data fetching and caching from the API |
| **react-markdown** | Render LLM answers |
| **Plotly.js (react-plotly.js, `scattergl`)** | Interactive topic map with ~28k points |
| **Recharts** | Evaluation charts (recall@k, comparisons) |

### Data source
- `rag-datasets/rag-mini-bioasq` on Hugging Face (CC BY 2.5 license — keep attribution in the README).

---

## 2. Dataset facts (verified — design around these)

Two files:

- **Passages:** `data/passages.parquet/part.0.parquet`
  - 40,221 rows. Single column `passage`. The **PubMed ID is stored as the parquet index named `id`**, not as a column. Use `pd.read_parquet(...).reset_index()` to get it as a column.
  - **~12,200 rows have the literal string `"nan"` as passage text** (articles with no abstract), plus a few rows like `"1."`. These must be filtered out. About **28,000 usable passages** remain.
  - Text length: median ~1,160 characters; ~3,350 passages are over 2,000 characters; 28 are over 8,000 characters (max ~33,000). Most passages fit in one chunk; long ones need splitting.
  - Text contains hard line breaks (`\n`) mid-sentence. Normalize whitespace.
- **Questions:** `data/test.parquet/part.0.parquet`
  - 4,719 rows. Columns: `question`, `answer` (gold answer), `relevant_passage_ids`.
  - `relevant_passage_ids` is a **JSON string** like `"[20598273, 6650562]"` — parse with `json.loads`. Median 6 relevant IDs per question, max 157.
  - Many relevant IDs point to the empty `"nan"` passages. **332 questions have no usable relevant passage at all.** For evaluation, remove empty passages from each question's gold set and skip questions whose gold set becomes empty.

PubMed IDs can be linked as `https://pubmed.ncbi.nlm.nih.gov/<id>/` in the UI for citations.

---

## 3. Project structure

```
bioasq-rag/
├── README.md
├── .env.example
├── .gitignore
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── config.py          # settings (models, paths, k, chunk sizes)
│   │   ├── main.py            # FastAPI app, CORS for Vite dev server
│   │   ├── api/
│   │   │   ├── chat.py        # POST /api/chat (SSE stream)
│   │   │   ├── search.py      # GET /api/search
│   │   │   ├── passages.py    # GET /api/passages/{pmid}
│   │   │   ├── questions.py   # GET /api/questions, /api/questions/{i}
│   │   │   ├── topics.py      # GET /api/topics/map, /api/topics/clusters
│   │   │   └── eval.py        # GET /api/eval/results
│   │   ├── core/
│   │   │   ├── llm.py         # Ollama chat wrapper, streaming
│   │   │   ├── embed.py       # Ollama embedding wrapper, batching
│   │   │   ├── store.py       # Chroma access
│   │   │   ├── bm25.py        # BM25 index load/query
│   │   │   ├── retrieve.py    # dense, bm25, hybrid (RRF)
│   │   │   └── prompts.py     # all prompt templates
│   │   └── models.py          # pydantic schemas
│   ├── scripts/
│   │   ├── 01_download.py
│   │   ├── 02_prepare.py      # clean + chunk -> data/processed/chunks.parquet
│   │   ├── 03_embed.py        # embed chunks -> Chroma
│   │   ├── 04_bm25.py         # build BM25 index
│   │   ├── 05_topics.py       # UMAP + HDBSCAN + LLM cluster labels
│   │   └── 06_evaluate.py     # retrieval + answer evaluation
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts         # proxy /api -> http://localhost:8000
│   └── src/
│       ├── pages/ (Chat, Search, Explorer, TopicMap, Evaluation)
│       ├── components/
│       └── api/client.ts
├── data/        # raw + processed files (gitignored)
└── storage/     # chroma db, bm25 index, topic map json, eval results (gitignored)
```

---

## 4. Phases

### Phase 1 — Environment setup
1. Install Homebrew packages if missing: `ollama`, `uv`, `node`.
2. Start Ollama and pull models (verify tags first):
   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama pull gemma4:26b      # verify exact tag for Gemma 4 26B A4B
   ollama pull gemma4:12b      # fallback, verify exact tag
   ```
3. Create `backend/` with `uv init`, add the backend dependencies from section 1.
4. Create `frontend/` with `npm create vite@latest frontend -- --template react-ts`, add Tailwind and the frontend dependencies.
5. Write `README.md` with setup steps.

**Acceptance:** `ollama list` shows the models; `uv run python -c "import chromadb, bm25s, umap"` works; `npm run dev` serves a blank app.

### Phase 2 — Download and prepare data
1. `01_download.py`: download both parquet files with `huggingface_hub.hf_hub_download` into `data/raw/`.
2. `02_prepare.py`:
   - Load passages, `reset_index()` to get `pmid`.
   - Drop rows whose stripped text is `"nan"`, empty, or shorter than 50 characters. Log how many were dropped.
   - Normalize whitespace (collapse `\n` and repeated spaces).
   - Chunk: passages ≤ 1,500 characters stay whole. Longer passages split on sentence boundaries into ~1,200-character chunks with ~200 characters of overlap.
   - Chunk ID format: `"{pmid}-{chunk_index}"`. Keep `pmid`, `chunk_index`, `text`, `char_len`.
   - Load questions, parse `relevant_passage_ids` with `json.loads`, compute `usable_relevant_ids` (intersection with non-empty passages), flag questions with none.
   - Save `data/processed/chunks.parquet` and `data/processed/questions.parquet`.
   - Print a summary: passages kept, chunks created, questions usable.

**Acceptance:** about 28,000 passages kept, chunk count printed, 4,387 usable questions. Unit tests for cleaning and chunking (including a 33,000-character passage).

### Phase 3 — Embeddings and vector store
1. `03_embed.py`:
   - Embed chunks with `qwen3-embedding:0.6b` through Ollama's `/api/embed` in batches of 64.
   - Store in a persistent Chroma collection `bioasq_chunks` at `storage/chroma`, cosine distance, metadata `pmid` and `chunk_index`.
   - Make it resumable: skip IDs already in the collection, so a crash does not restart from zero.
   - Also save all vectors to `storage/embeddings.npy` with a matching ID list (used by the topic map).
   - Unload the embedding model at the end (`keep_alive=0`).
2. Note for queries: Qwen3-Embedding works best when queries get an instruction prefix, for example `"Instruct: Given a biomedical question, retrieve relevant PubMed abstracts\nQuery: {question}"`, while documents are embedded as-is. Put this in `prompts.py`.

**Acceptance:** collection count equals chunk count. A quick script query for "What causes Pompe disease?" returns obviously relevant passages. Report total embedding time.

### Phase 4 — Retrieval
1. `04_bm25.py`: build a `bm25s` index over chunk texts (English stopwords, simple stemming), save to `storage/bm25/`.
2. `retrieve.py` with three strategies, all returning `[{chunk_id, pmid, text, score}]`:
   - `dense(query, k)` — Chroma.
   - `bm25(query, k)` — bm25s.
   - `hybrid(query, k)` — Reciprocal Rank Fusion of dense and BM25 top-50 lists (`score = Σ 1/(60 + rank)`).
3. Deduplicate by `pmid` when returning sources, keeping the best chunk per article.

**Acceptance:** tests for RRF math and deduplication; all three strategies return results for a sample question.

### Phase 5 — Evaluation (the "study" core)
`06_evaluate.py`, results saved as JSON in `storage/eval/`:
1. **Retrieval evaluation** on all usable questions, for dense, BM25 and hybrid:
   - Recall@1, @5, @10, @20 (at the PubMed-ID level), MRR, and hit-rate@10.
   - This needs no LLM and runs fast. Run it first.
2. **Answer evaluation** on a random sample of 100 questions (configurable, fixed seed), because local generation is slow:
   - Generate an answer with the best retrieval strategy.
   - LLM-as-judge with the same local model: compare generated answer to the gold `answer`, score 1–5 for correctness and 1–5 for groundedness in the retrieved sources, with a short reason. Force JSON output and validate with pydantic; retry once on invalid JSON.
   - Also run the same 100 questions **without retrieval** (model alone) to show what RAG adds.
3. Save per-question records (question, gold answer, retrieved IDs, hits, generated answer, judge scores) for the UI.

**Acceptance:** printed metrics table comparing the three retrievers; answer eval completes for 100 questions with no crashes; results JSON loads in the API.

### Phase 6 — Topic map
`05_topics.py`:
1. Load `embeddings.npy` (one vector per passage: use chunk 0, or the mean of a passage's chunks).
2. UMAP to 2D (`n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`, fixed `random_state`).
3. HDBSCAN on a 10–15-dimension UMAP projection (better clusters than clustering in 2D). Tune `min_cluster_size` (start at 50) so you get roughly 30–80 clusters.
4. For each cluster, send 8 representative passages (closest to the cluster centroid) to the LLM and ask for a 2–5 word topic label and a one-sentence description. Return JSON.
5. Save `storage/topics/map.json` with `{pmid, x, y, cluster_id}` per point and `clusters.json` with labels, sizes and descriptions.

**Acceptance:** map file has one point per passage; cluster labels look like real biomedical topics (e.g. "Breast cancer genetics").

### Phase 7 — Backend API
FastAPI with CORS for `http://localhost:5173`:

| Endpoint | Description |
|---|---|
| `POST /api/chat` | Body: `{message, history, strategy, k}`. Retrieves, builds prompt, streams tokens over SSE. Final event contains the sources (pmid, snippet, score). |
| `GET /api/search?q=&strategy=&k=` | Retrieval only, no LLM. |
| `GET /api/passages/{pmid}` | Full passage text and its cluster. |
| `GET /api/questions?query=&page=` | Browse/search the BioASQ questions. |
| `GET /api/questions/{i}/run?strategy=` | Run one question: retrieved passages marked hit/miss against gold, generated answer, gold answer. |
| `GET /api/topics/map`, `GET /api/topics/clusters` | Topic map data. |
| `GET /api/eval/results` | Stored evaluation results. |
| `GET /api/health` | Ollama reachable, models present, collection count. |

**RAG prompt rules (in `prompts.py`):** answer only from the provided numbered sources; cite sources inline as `[1]`, `[2]`; say clearly when the sources do not contain the answer; this is a research/study tool, not medical advice. Keep the model's context at `num_ctx=8192` to fit in memory.

**Acceptance:** `pytest` passes; `/api/health` is green; a curl to `/api/chat` streams an answer with citations.

### Phase 8 — Frontend
Pages (sidebar navigation):
1. **Chat** — streaming answers rendered with react-markdown, clickable `[n]` citations that open a source panel showing the passage and a PubMed link. Settings popover: retrieval strategy, k, model.
2. **Search** — query box, side-by-side results from dense vs BM25 vs hybrid, so you can see how they differ.
3. **Question Explorer** — browse the 4,719 BioASQ questions; run one and see gold answer vs generated answer, and retrieved passages colored green (in gold set) or grey.
4. **Topic Map** — Plotly `scattergl` of all passages colored by cluster, hover shows topic label, click opens the passage. Cluster list with sizes and descriptions; clicking a cluster highlights it. A "Chat about this topic" button pre-fills the chat.
5. **Evaluation** — Recharts bar/line charts: recall@k per strategy, MRR, RAG vs no-RAG judge scores, and a table of the worst-performing questions to study failures.

Support light and dark mode. Show a clear banner if `/api/health` reports Ollama is down.

**Acceptance:** all five pages work end to end against the running backend; the topic map stays smooth with ~28k points.

### Phase 9 — Polish
- `Makefile` (or `justfile`) with targets: `setup`, `data`, `index`, `topics`, `eval`, `dev`.
- README section "How to study the dataset" describing suggested experiments (below).
- Log timings for every pipeline step.

---

## 5. Memory and performance notes (24 GB M4 Pro)

- Approximate budget: chat model 14–15 GB + embedding model 1.5 GB + macOS, browser and app. This is tight. Run ingestion steps with only the embedding model loaded, and chat with only the LLM loaded.
- If you see heavy swapping (Activity Monitor → Memory Pressure turns yellow/red) or slow tokens, switch `CHAT_MODEL` to the Gemma 4 12B fallback in `.env`.
- Keep `num_ctx` at 8192 and `k` ≤ 8 sources in chat to limit memory use.
- Expect embedding ~28k passages to take a while (tens of minutes). This is why Phase 3 must be resumable.

---

## 6. Suggested study experiments

Once built, use the app to answer questions like:
- How much better is hybrid retrieval than dense alone? Where does BM25 win (gene names, acronyms, drug names)?
- Does a larger `k` improve answers, or just add noise?
- How does the model do *without* retrieval vs with it?
- Which topic clusters have the worst retrieval recall, and why?
- What happens with chunk size 600 vs 1,200 vs whole passages? (Re-run Phases 2–5 with a different config and compare the eval JSON.)
- Try a different embedding model (e.g. `qwen3-embedding:4b` or `bge-m3`) and compare recall@10. Note: switching embedding models requires a full re-index.

---

## 7. Definition of done

- One command sequence (`make setup data index topics eval dev`) takes a fresh clone to a running app.
- All five UI pages work, fully offline after models and data are downloaded.
- Evaluation report exists comparing dense, BM25 and hybrid retrieval, plus RAG vs no-RAG answer quality.
- Tests pass and the README documents setup, models used, and dataset attribution (rag-mini-bioasq, CC BY 2.5, derived from BioASQ).
