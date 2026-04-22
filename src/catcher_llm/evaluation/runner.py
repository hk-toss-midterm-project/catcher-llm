from __future__ import annotations

from pathlib import Path

from langsmith import evaluate

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.evaluation.dataset import build_langsmith_client, ensure_rag_dataset
from catcher_llm.evaluation.evaluators import RAG_EVALUATORS
from catcher_llm.services.rag_service import rag_target


def run_rag_evaluation(
    *,
    settings: Settings | None = None,
    dataset_name: str | None = None,
    examples_path: Path | None = None,
    experiment_prefix: str | None = None,
    max_concurrency: int = 4,
):
    config = settings or get_settings()
    if not config.has_langsmith_key:
        raise ValueError("LANGSMITH_API_KEY is not set.")

    dataset = ensure_rag_dataset(
        settings=config,
        dataset_name=dataset_name,
        examples_path=examples_path,
    )
    client = build_langsmith_client(config)

    return evaluate(
        lambda inputs: rag_target(inputs, settings=config),
        data=dataset.name,
        evaluators=RAG_EVALUATORS,
        experiment_prefix=experiment_prefix or config.langsmith_experiment_prefix,
        description="Starter RAG evaluation for the Catcher LLM project.",
        max_concurrency=max_concurrency,
        client=client,
    )
