"""Download the rag-mini-bioasq parquet files into data/raw/."""

import shutil
import time

from huggingface_hub import hf_hub_download

from app.config import settings


def main() -> None:
    start = time.perf_counter()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in [
        ("passages", settings.passages_file),
        ("questions", settings.questions_file),
    ]:
        cached = hf_hub_download(
            repo_id=settings.hf_repo_id, filename=filename, repo_type="dataset"
        )
        target = settings.raw_dir / f"{name}.parquet"
        shutil.copyfile(cached, target)
        print(f"{name:<10} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")

    print(f"Done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
