from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.llm.models import get_chat_model, get_embeddings_model
from catcher_llm.services.rag.kca_report import generate_kca_report_rag_reply
from catcher_llm.services.ragas.kca_eval_dataset import (
    get_kca_eval_questions,
    get_kca_ground_truths,
)

if TYPE_CHECKING:
    from pandas import DataFrame


def _extract_contexts(rag_response: Any) -> list[str]:
    """RAG 응답 객체에서 ragas가 기대하는 문자열 컨텍스트 목록만 추출한다."""
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
    """RAG 응답 객체에서 평가 대상 답변 문자열을 우선순위에 따라 추출한다."""
    if hasattr(rag_response, "answer"):
        return str(rag_response.answer)

    if hasattr(rag_response, "content"):
        return str(rag_response.content)

    if hasattr(rag_response, "response"):
        return str(rag_response.response)

    return str(rag_response)


def build_kca_ragas_dataset(settings: Settings | None = None) -> Dataset:
    """KCA 질문 세트를 실제 RAG 응답으로 채워 ragas 평가용 데이터셋으로 구성한다."""
    questions = get_kca_eval_questions()
    ground_truths = get_kca_ground_truths()

    answers: list[str] = []
    contexts: list[list[str]] = []

    for question in questions:
        rag_response = generate_kca_report_rag_reply(question, settings=settings)
        answers.append(_extract_answer(rag_response))
        contexts.append(_extract_contexts(rag_response))

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }

    return Dataset.from_dict(data)


def run_kca_ragas_eval(settings: Settings | None = None) -> EvaluationResult:
    """프로젝트 설정의 LLM과 임베딩 모델을 사용해 KCA ragas 평가를 실행한다."""
    config = settings or get_settings()
    dataset = build_kca_ragas_dataset(settings=config)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_chat_model(config),
        embeddings=get_embeddings_model(config),
    )


def save_kca_ragas_result(
    output_path: str = "kca_ragas_result.csv",
    settings: Settings | None = None,
) -> DataFrame:
    """KCA ragas 평가 결과를 CSV 파일로 저장하고 데이터프레임으로 반환한다."""
    result = run_kca_ragas_eval(settings=settings)
    df = result.to_pandas()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    dataframe = save_kca_ragas_result()
    print(dataframe)
