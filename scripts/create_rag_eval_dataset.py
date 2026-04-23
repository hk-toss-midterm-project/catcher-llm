"""RAG 평가 예제를 위한 LangSmith 데이터셋을 생성하거나 재사용한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from catcher_llm.config.settings import get_settings
from catcher_llm.evaluation.dataset import ensure_rag_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--examples-path", default=None)
    args = parser.parse_args()

    settings = get_settings()
    dataset = ensure_rag_dataset(
        settings=settings,
        dataset_name=args.dataset_name,
        examples_path=Path(args.examples_path) if args.examples_path else None,
    )
    print(f"Dataset ready: {dataset.name}")
    print(f"Dataset id: {dataset.id}")


if __name__ == "__main__":
    main()
