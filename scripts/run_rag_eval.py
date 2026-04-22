from __future__ import annotations

import argparse
from pathlib import Path

from catcher_llm.config.settings import get_settings
from catcher_llm.evaluation.runner import run_rag_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--examples-path", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    results = run_rag_evaluation(
        settings=settings,
        dataset_name=args.dataset_name,
        examples_path=Path(args.examples_path) if args.examples_path else None,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
    )

    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    print("RAG evaluation submitted to LangSmith.")


if __name__ == "__main__":
    main()
