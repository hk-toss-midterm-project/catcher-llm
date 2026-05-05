from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.llm.models import get_chat_model, get_embeddings_model
from catcher_llm.services.rag.welfare import generate_welfare_rag_reply
from catcher_llm.services.ragas.welfare_eval_dataset import (
    get_welfare_eval_questions,
    get_welfare_ground_truths,
)

if TYPE_CHECKING:
    from pandas import DataFrame


def _extract_contexts(rag_response: Any) -> list[str]:
    """RAG 응답 객체에서 ragas 평가용 context 문자열 목록을 추출한다."""
    if hasattr(rag_response, "contexts"):
        return [
            getattr(doc, "page_content", getattr(doc, "content", str(doc)))
            for doc in rag_response.contexts
        ]

    if hasattr(rag_response, "source_documents"):
        return [
            getattr(doc, "page_content", getattr(doc, "content", str(doc)))
            for doc in rag_response.source_documents
        ]

    if hasattr(rag_response, "documents"):
        return [
            getattr(doc, "page_content", getattr(doc, "content", str(doc)))
            for doc in rag_response.documents
        ]

    return []


def _extract_answer(rag_response: Any) -> str:
    """RAG 응답 객체에서 평가용 답변 문자열을 추출한다."""
    if hasattr(rag_response, "answer"):
        return str(rag_response.answer)

    if hasattr(rag_response, "content"):
        return str(rag_response.content)

    if hasattr(rag_response, "response"):
        return str(rag_response.response)

    return str(rag_response)


def build_welfare_ragas_dataset(settings: Settings | None = None) -> Dataset:
    """복지 정책 질문 세트와 RAG 응답으로 ragas 평가 데이터셋을 구성한다."""
    questions = get_welfare_eval_questions()
    ground_truths = get_welfare_ground_truths()

    answers: list[str] = []
    contexts: list[list[str]] = []

    for question in questions:
        rag_response = generate_welfare_rag_reply(question, settings=settings)
        answers.append(_extract_answer(rag_response))
        contexts.append(_extract_contexts(rag_response))

    return Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        }
    )


def run_welfare_ragas_eval(settings: Settings | None = None) -> EvaluationResult:
    """프로젝트 설정 모델로 복지 정책 ragas 평가를 실행한다."""
    config = settings or get_settings()
    dataset = build_welfare_ragas_dataset(settings=config)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=3),
        ContextPrecision(),
        ContextRecall(),
    ]

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_chat_model(config),
        embeddings=get_embeddings_model(config),
    )


def save_welfare_ragas_result(
    output_path: str = "welfare_ragas_result.csv",
    settings: Settings | None = None,
) -> DataFrame:
    """복지 정책 ragas 평가 결과를 CSV로 저장하고 데이터프레임으로 반환한다."""
    result = run_welfare_ragas_eval(settings=settings)
    dataframe = result.to_pandas()
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return dataframe
