from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "gemma4:26b-a4b-it-qat"
    fallback_chat_model: str = "gemma4:12b-it-qat"
    embed_model: str = "qwen3-embedding:0.6b"
    num_ctx: int = 8192
    # Gemma 4 "thinks" before answering by default; ~10x slower for short answers.
    chat_think: bool = False
    embed_batch_size: int = 64

    # Dataset
    hf_repo_id: str = "rag-datasets/rag-mini-bioasq"
    passages_file: str = "data/passages.parquet/part.0.parquet"
    questions_file: str = "data/test.parquet/part.0.parquet"

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    storage_dir: Path = ROOT_DIR / "storage"

    # Cleaning and chunking
    min_passage_chars: int = 50
    max_whole_passage_chars: int = 2000
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200

    # Retrieval
    collection_name: str = "bioasq_chunks"
    fusion_candidates: int = 50
    rrf_k: int = 60
    default_k: int = 8
    max_chat_k: int = 8

    # Evaluation
    eval_sample_size: int = 100
    eval_seed: int = 42

    # Topic map
    topics_seed: int = 42
    topics_n_neighbors: int = 15
    topics_cluster_dims: int = 10
    # Tuned on this corpus: 55 clusters, ~36% noise (noise stays 32-36% for nearby values).
    topics_min_cluster_size: int = 150
    topics_min_samples: int = 10
    topics_representatives: int = 8
    topics_representative_chars: int = 800

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.storage_dir / "bm25"

    @property
    def topics_dir(self) -> Path:
        return self.storage_dir / "topics"

    @property
    def eval_dir(self) -> Path:
        return self.storage_dir / "eval"


settings = Settings()
