from __future__ import annotations

import json
from pathlib import Path

from langsmith import Client
from langsmith.schemas import Dataset
from langsmith.utils import LangSmithNotFoundError

from catcher_llm.config.settings import Settings, get_settings


def load_eval_examples(path: Path) -> list[dict]:
    """평가 예제 JSON 파일을 읽고 LangSmith 예제 형식인지 검증한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation examples must be a list of {inputs, outputs} records.")

    for record in payload:
        if "inputs" not in record or "outputs" not in record:
            raise ValueError("Each evaluation record must include inputs and outputs.")
    return payload


def build_langsmith_client(settings: Settings | None = None) -> Client:
    """현재 설정값으로 LangSmith API 클라이언트를 생성한다."""
    config = settings or get_settings()
    return Client(
        api_key=config.langsmith_api_key or None,
        api_url=config.langsmith_endpoint or None,
    )


def ensure_rag_dataset(
    *,
    settings: Settings | None = None,
    dataset_name: str | None = None,
    examples_path: Path | None = None,
) -> Dataset:
    """RAG 평가 데이터셋을 조회하거나 생성하고, 비어 있으면 예제를 업로드한다."""
    config = settings or get_settings()
    if not config.has_langsmith_key:
        raise ValueError("LANGSMITH_API_KEY is not set.")

    client = build_langsmith_client(config)
    target_name = dataset_name or config.langsmith_dataset_name
    target_path = examples_path or (config.eval_data_dir / "rag_examples.json")

    try:
        dataset = client.read_dataset(dataset_name=target_name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=target_name,
            description="Starter RAG evaluation dataset for the Catcher LLM project.",
        )

    if next(client.list_examples(dataset_id=dataset.id, limit=1), None) is None:
        client.create_examples(
            dataset_id=dataset.id,
            examples=load_eval_examples(target_path),
        )

    return dataset
