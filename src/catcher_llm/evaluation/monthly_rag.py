from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd
from langchain_core.messages import BaseMessage
from langsmith import evaluate as langsmith_evaluate
from langsmith.evaluation import EvaluationResult
from langsmith.utils import LangSmithNotFoundError
from pydantic import BaseModel

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.evaluation.dataset import build_langsmith_client
from catcher_llm.llm.models import get_chat_model, get_embeddings_model
from catcher_llm.schemas.consumption_feedback import (
    MonthlyFeedbackAction,
    MonthlyFeedbackEvidence,
    MonthlyFeedbackResult,
    MonthlyFeedbackServiceResult,
    RetrievedAdviceContext,
)
from catcher_llm.services.rag.config import DocumentKind, get_rag_pipeline_config
from catcher_llm.services.rag.core import generate_rag_reply

RAGAS_METRICS: tuple[str, ...] = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)
RAGAS_METRIC_LABELS: dict[str, str] = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}

_MONTHLY_ALLOWED_DOCUMENT_KINDS: tuple[str, ...] = (
    "user_report",
    "catcher_consumption_benchmark",
    "kca_report",
)
_DOCUMENT_SIGNAL_TERMS: tuple[str, ...] = (
    "전체 사용자",
    "사용자 보고서",
    "소비 동향",
    "소비자원",
    "벤치마크",
    "카드 결제",
    "USER_REPORT",
    "CATCHER_CONSUMPTION_BENCHMARK",
    "KCA_REPORT",
)
_ACTION_TERMS: tuple[str, ...] = (
    "결제 전",
    "확인",
    "점검",
    "보류",
    "목록",
    "작성",
    "기록",
    "비교",
    "알림",
    "분리",
    "정리",
)
_GENERIC_GOAL_TERMS: tuple[str, ...] = (
    "목표 세",
    "목표를 세",
    "목표를 설정",
    "줄이기",
    "줄이는 목표",
    "절감 목표",
    "감축 목표",
    "각각 줄",
)
_PERCENT_REDUCTION_RE = re.compile(r"\d+(?:\.\d+)?\s*%[^.\n]*(?:줄|절감|감축)")
_RAW_PERCENT_RE = re.compile(r"\d+\.\d{3,}\s*%")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_USABLE_CLAIM_RE = re.compile(r"usable_claim:\s*(.+)")


@dataclass(frozen=True, slots=True)
class MonthlyRagasRecord:
    """RAGAS 평가에 전달할 월간 문서 RAG 단일 샘플을 표현한다."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    document_kind: str | None = None
    source_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MonthlyRagQuerySpec:
    """월간 피드백 생성 중 사용된 문서 RAG 검색 쿼리와 기준 답변 후보를 표현한다."""

    query: str
    document_kind: str | None
    contexts: list[RetrievedAdviceContext]
    reference: str


class MonthlyRuleEvaluation(BaseModel):
    """월간 피드백 RAG 규칙 하나의 통과 여부와 설명을 표현한다."""

    key: str
    label: str
    passed: bool
    detail: str


class MonthlyRuleSummary(BaseModel):
    """월간 피드백 RAG 규칙 기반 평가 결과 전체를 표현한다."""

    pass_count: int
    total_count: int
    pass_rate: float
    rule_evaluations: list[MonthlyRuleEvaluation]


class MonthlyLlmJudgeResult(BaseModel):
    """LLM Judge의 원문 결과와 파싱 가능한 점수 요약을 표현한다."""

    raw_output: str
    total_score: float | None = None
    groundedness_score: float | None = None
    product_fit_score: float | None = None
    retrieval_use_score: float | None = None
    reason: str = ""


class MonthlyLangSmithEvaluationResult(BaseModel):
    """LangSmith Experiment 실행 결과의 화면 표시용 요약을 표현한다."""

    dataset_name: str
    experiment_prefix: str
    example_count: int
    experiment_name: str = ""
    experiment_url: str = ""


def _clean_text(value: str) -> str:
    """반복 비교와 규칙 평가를 위해 공백을 단일 공백으로 정규화한다."""
    return " ".join(value.split())


def _normalize_eval_text(value: str) -> str:
    """LangSmith code evaluator 비교를 위해 특수문자와 중복 공백을 정규화한다."""
    lowered = value.lower()
    normalized = re.sub(r"[^0-9a-z가-힣\s/_.%-]+", " ", lowered)
    return " ".join(normalized.split())


def _compact_text(value: str) -> str:
    """문장 반복 검사를 위해 공백과 일부 문장부호를 제거한 문자열을 만든다."""
    return re.sub(r"[\s.,!?。！？]+", "", value)


def _format_evidence(evidence: MonthlyFeedbackEvidence) -> str:
    """월간 피드백 근거 모델을 평가 입력용 한 줄 문자열로 변환한다."""
    parts = [
        f"- evidence_type={evidence.evidence_type}",
        f"title={evidence.title}",
        f"detail={evidence.detail}",
    ]
    if evidence.source_json_path:
        parts.append(f"source_json_path={evidence.source_json_path}")
    if evidence.source:
        parts.append(f"source={evidence.source}")
    if evidence.page_number is not None:
        parts.append(f"page={evidence.page_number}")
    return " | ".join(parts)


def _format_action(action: MonthlyFeedbackAction) -> str:
    """월간 피드백 행동 항목 모델을 평가 입력용 한 줄 문자열로 변환한다."""
    parts = [
        f"- title={action.title}",
        f"detail={action.detail}",
        f"target_json_path={action.target_json_path}",
        f"urgency={action.urgency}",
    ]
    if action.related_source:
        parts.append(f"related_source={action.related_source}")
    return " | ".join(parts)


def format_monthly_feedback_response(feedback: MonthlyFeedbackResult) -> str:
    """월간 피드백 결과를 RAGAS와 LLM Judge가 읽기 쉬운 텍스트로 변환한다."""
    evidence_lines = "\n".join(_format_evidence(item) for item in feedback.key_evidences)
    action_lines = "\n".join(_format_action(item) for item in feedback.action_items)
    return "\n".join(
        [
            f"summary_title: {feedback.summary_title}",
            f"feedback_message: {feedback.feedback_message}",
            f"next_month_mission: {feedback.next_month_mission}",
            "key_evidences:",
            evidence_lines or "- 없음",
            "action_items:",
            action_lines or "- 없음",
        ]
    )


def _format_monthly_reference(feedback: MonthlyFeedbackResult) -> str:
    """RAGAS reference로 사용할 월간 피드백의 핵심 근거와 기대 산출물을 구성한다."""
    document_evidences = [
        evidence for evidence in feedback.key_evidences if evidence.evidence_type == "document"
    ]
    evidence_source = document_evidences if document_evidences else feedback.key_evidences
    evidence_lines = "\n".join(_format_evidence(item) for item in evidence_source)
    return "\n".join(
        [
            f"피드백 본문: {feedback.feedback_message}",
            f"다음 달 미션: {feedback.next_month_mission}",
            "핵심 근거:",
            evidence_lines or "- 없음",
        ]
    )


def _extract_context_reference(context: RetrievedAdviceContext) -> str:
    """검색 문서 본문에서 쿼리별 RAGAS reference로 쓸 핵심 claim을 추출한다."""
    match = _USABLE_CLAIM_RE.search(context.content)
    if match is not None:
        return _clean_text(match.group(1))

    paragraphs = [
        _clean_text(paragraph) for paragraph in context.content.split("\n\n") if paragraph.strip()
    ]
    if not paragraphs:
        return ""
    first_paragraph = paragraphs[0]
    sentences = re.split(r"(?<=[.!?。！？다])\s+", first_paragraph)
    return " ".join(sentence for sentence in sentences[:2] if sentence).strip()


def _build_query_spec_reference(contexts: Sequence[RetrievedAdviceContext]) -> str:
    """같은 검색 쿼리로 찾은 문서들의 핵심 claim을 합쳐 기준 답변을 만든다."""
    references: list[str] = []
    for context in contexts:
        reference = _extract_context_reference(context)
        if reference and reference not in references:
            references.append(reference)
    return "\n".join(references)


def build_monthly_rag_query_specs(
    result: MonthlyFeedbackServiceResult,
) -> list[MonthlyRagQuerySpec]:
    """월간 피드백 생성 중 검색된 문서 컨텍스트를 쿼리별 RAGAS 평가 단위로 묶는다."""
    grouped_contexts: dict[tuple[str, str | None], list[RetrievedAdviceContext]] = {}
    for context in result.retrieved_contexts:
        if not context.query.strip() or not context.content.strip():
            continue
        key = (context.query, context.document_kind)
        grouped_contexts.setdefault(key, []).append(context)

    specs: list[MonthlyRagQuerySpec] = []
    for (query, document_kind), contexts in grouped_contexts.items():
        reference = _build_query_spec_reference(contexts)
        if not reference:
            continue
        specs.append(
            MonthlyRagQuerySpec(
                query=query,
                document_kind=document_kind,
                contexts=contexts,
                reference=reference,
            )
        )
    return specs


def _coerce_document_kind(document_kind: str | None) -> DocumentKind | None:
    """문서 종류 문자열을 RAG 파이프라인 설정에 쓸 enum으로 변환한다."""
    if document_kind is None:
        return None
    try:
        return DocumentKind(document_kind)
    except ValueError:
        return None


def _build_ragas_record_from_rag_response(
    spec: MonthlyRagQuerySpec,
    *,
    response: str,
    contexts: list[str],
    error: str | None = None,
) -> MonthlyRagasRecord:
    """generate_rag_reply 결과를 쿼리별 RAGAS 레코드로 변환한다."""
    return MonthlyRagasRecord(
        user_input=spec.query,
        response=response,
        retrieved_contexts=contexts,
        reference=spec.reference,
        document_kind=spec.document_kind,
        source_count=len(contexts),
        error=error,
    )


def run_monthly_rag_query_generation(
    specs: Sequence[MonthlyRagQuerySpec],
    *,
    settings: Settings | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 6,
    temperature: float = 0.0,
) -> list[MonthlyRagasRecord]:
    """월간 피드백 RAG 검색 쿼리마다 generate_rag_reply를 실행해 RAGAS 레코드를 만든다."""
    config = settings or get_settings()
    records: list[MonthlyRagasRecord] = []
    for spec in specs:
        document_kind = _coerce_document_kind(spec.document_kind)
        try:
            if document_kind is None:
                rag_response = generate_rag_reply(
                    spec.query,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    settings=config,
                    temperature=temperature,
                )
            else:
                pipeline_config = get_rag_pipeline_config(document_kind, settings=config)
                rag_response = generate_rag_reply(
                    spec.query,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    raw_data_dir=pipeline_config.raw_data_dir,
                    source_files=pipeline_config.source_files,
                    settings=config,
                    prompt=pipeline_config.prompt,
                    temperature=temperature,
                )
            contexts = [context.content for context in rag_response.contexts]
            records.append(
                _build_ragas_record_from_rag_response(
                    spec,
                    response=rag_response.answer,
                    contexts=contexts,
                    error=rag_response.error,
                )
            )
        except Exception as exc:
            records.append(
                _build_ragas_record_from_rag_response(
                    spec,
                    response="",
                    contexts=[],
                    error=str(exc),
                )
            )
    return records


def build_monthly_langsmith_examples(
    specs: Sequence[MonthlyRagQuerySpec],
) -> list[dict[str, object]]:
    """월간 문서 RAG 쿼리 스펙을 LangSmith dataset example 형식으로 변환한다."""
    examples: list[dict[str, object]] = []
    for spec in specs:
        examples.append(
            {
                "inputs": {
                    "query": spec.query,
                    "document_kind": spec.document_kind or "",
                },
                "outputs": {
                    "reference": spec.reference,
                },
                "metadata": {
                    "document_kind": spec.document_kind or "",
                    "source_count": len(spec.contexts),
                },
            }
        )
    return examples


def monthly_rag_langsmith_target(
    inputs: dict[str, object],
    *,
    settings: Settings | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 6,
    temperature: float = 0.0,
) -> dict[str, object]:
    """LangSmith evaluate가 호출할 월간 문서 RAG target 함수를 실행한다."""
    query = str(inputs["query"])
    document_kind = _coerce_document_kind(str(inputs.get("document_kind") or ""))
    config = settings or get_settings()

    if document_kind is None:
        rag_response = generate_rag_reply(
            query,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            settings=config,
            temperature=temperature,
        )
    else:
        pipeline_config = get_rag_pipeline_config(document_kind, settings=config)
        rag_response = generate_rag_reply(
            query,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=pipeline_config.raw_data_dir,
            source_files=pipeline_config.source_files,
            settings=config,
            prompt=pipeline_config.prompt,
            temperature=temperature,
        )

    return {
        "answer": rag_response.answer,
        "contexts": [
            {
                "source": context.source,
                "content": context.content,
                "page_number": context.page_number,
            }
            for context in rag_response.contexts
        ],
        "sources": rag_response.sources,
        "document_kind": document_kind.value if document_kind is not None else "",
        "error": rag_response.error,
    }


def _extract_langsmith_context_blob(outputs: dict[str, object]) -> str:
    """LangSmith target 출력에서 검색 컨텍스트 본문을 하나의 문자열로 합친다."""
    contexts = outputs.get("contexts", [])
    if not isinstance(contexts, list):
        return ""

    parts: list[str] = []
    for item in contexts:
        if isinstance(item, dict):
            parts.append(str(item.get("content", "")))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def monthly_rag_answer_nonempty(
    inputs: dict[str, object],
    outputs: dict[str, object],
) -> EvaluationResult:
    """LangSmith code evaluator로 문서 RAG 답변이 비어 있지 않은지 평가한다."""
    answer = str(outputs.get("answer", "")).strip()
    return EvaluationResult(
        key="answer_nonempty",
        score=bool(answer),
        comment="문서 RAG 답변이 생성되었습니다." if answer else "문서 RAG 답변이 비어 있습니다.",
    )


def monthly_rag_has_retrieved_contexts(
    inputs: dict[str, object],
    outputs: dict[str, object],
) -> EvaluationResult:
    """LangSmith code evaluator로 문서 RAG 출력에 검색 컨텍스트가 있는지 평가한다."""
    contexts = outputs.get("contexts", [])
    has_contexts = isinstance(contexts, list) and bool(contexts)
    return EvaluationResult(
        key="has_retrieved_contexts",
        score=has_contexts,
        comment="검색 컨텍스트가 있습니다." if has_contexts else "검색 컨텍스트가 없습니다.",
    )


def monthly_rag_context_recall(
    inputs: dict[str, object],
    outputs: dict[str, object],
    reference_outputs: dict[str, object],
) -> EvaluationResult:
    """LangSmith code evaluator로 reference 핵심 문구가 검색 컨텍스트에 포함되는지 평가한다."""
    reference = _normalize_eval_text(str(reference_outputs.get("reference", "")))
    context_blob = _normalize_eval_text(_extract_langsmith_context_blob(outputs))
    has_reference = bool(reference and context_blob and reference in context_blob)
    return EvaluationResult(
        key="reference_in_contexts",
        score=has_reference,
        comment=(
            "reference가 검색 컨텍스트에 포함됩니다."
            if has_reference
            else "reference 전체 문구가 검색 컨텍스트에서 그대로 확인되지 않습니다."
        ),
    )


def monthly_rag_answer_reference_overlap(
    inputs: dict[str, object],
    outputs: dict[str, object],
    reference_outputs: dict[str, object],
) -> EvaluationResult:
    """LangSmith code evaluator로 답변과 reference의 토큰 겹침 비율을 계산한다."""
    answer_tokens = set(_normalize_eval_text(str(outputs.get("answer", ""))).split())
    reference_tokens = set(
        _normalize_eval_text(str(reference_outputs.get("reference", ""))).split()
    )
    if not reference_tokens:
        return EvaluationResult(
            key="answer_reference_token_overlap",
            score=None,
            comment="reference 토큰이 없어 겹침 비율을 계산하지 않았습니다.",
        )

    overlap_score = len(answer_tokens & reference_tokens) / len(reference_tokens)
    return EvaluationResult(
        key="answer_reference_token_overlap",
        score=round(overlap_score, 3),
        comment=f"reference 토큰 {len(reference_tokens)}개 중 {len(answer_tokens & reference_tokens)}개가 답변에 포함되었습니다.",
    )


MONTHLY_RAG_LANGSMITH_EVALUATORS = [
    monthly_rag_answer_nonempty,
    monthly_rag_has_retrieved_contexts,
    monthly_rag_context_recall,
    monthly_rag_answer_reference_overlap,
]


def run_monthly_langsmith_evaluation(
    specs: Sequence[MonthlyRagQuerySpec],
    *,
    settings: Settings | None = None,
    dataset_name: str = "catcher-monthly-feedback-rag-eval",
    experiment_prefix: str = "monthly-feedback-rag",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 6,
    max_concurrency: int = 2,
) -> MonthlyLangSmithEvaluationResult:
    """월간 문서 RAG 쿼리 평가를 LangSmith Dataset/Experiment로 실행한다."""
    config = settings or get_settings()
    if not config.has_langsmith_key:
        raise ValueError("LANGSMITH_API_KEY is not set.")

    examples = build_monthly_langsmith_examples(specs)
    if not examples:
        raise ValueError("LangSmith에 업로드할 월간 RAG 평가 예제가 없습니다.")

    client = build_langsmith_client(config)
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name,
            description="월간 피드백 생성에 사용되는 문서 RAG 쿼리별 평가 데이터셋.",
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    experiment = langsmith_evaluate(
        lambda inputs: monthly_rag_langsmith_target(
            inputs,
            settings=config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        ),
        data=dataset.name,
        evaluators=MONTHLY_RAG_LANGSMITH_EVALUATORS,
        experiment_prefix=experiment_prefix,
        description="월간 피드백 문서 RAG 쿼리별 LangSmith 평가.",
        max_concurrency=max_concurrency,
        client=client,
    )

    return MonthlyLangSmithEvaluationResult(
        dataset_name=dataset.name,
        experiment_prefix=experiment_prefix,
        example_count=len(examples),
        experiment_name=str(getattr(experiment, "experiment_name", "")),
        experiment_url=str(getattr(experiment, "url", "")),
    )


def build_monthly_ragas_frame(records: Sequence[MonthlyRagasRecord]) -> pd.DataFrame:
    """월간 RAGAS 레코드 목록을 ragas 입력 컬럼을 가진 DataFrame으로 변환한다."""
    return pd.DataFrame(
        {
            "user_input": [record.user_input for record in records],
            "response": [record.response for record in records],
            "retrieved_contexts": [record.retrieved_contexts for record in records],
            "reference": [record.reference for record in records],
        }
    )


def run_monthly_ragas_evaluation(
    records: Sequence[MonthlyRagasRecord],
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """RAGAS 4대 지표로 월간 문서 RAG 쿼리별 레코드를 평가하고 결과 DataFrame을 반환한다."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    valid_records = [
        record
        for record in records
        if record.response.strip()
        and record.retrieved_contexts
        and record.reference.strip()
        and record.error is None
    ]
    if not valid_records:
        raise ValueError("RAGAS 평가 레코드가 비어 있습니다.")
    config = settings or get_settings()
    frame = build_monthly_ragas_frame(valid_records)
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(strictness=3),
        ],
        llm=get_chat_model(config),
        embeddings=get_embeddings_model(config),
    )
    result_frame = cast(pd.DataFrame, evaluation_result.to_pandas())
    result_frame.insert(0, "document_kind", [record.document_kind for record in valid_records])
    result_frame.insert(1, "source_count", [record.source_count for record in valid_records])
    return result_frame


def summarize_ragas_scores(frame: pd.DataFrame) -> dict[str, float]:
    """RAGAS 결과 DataFrame에서 주요 지표 평균 점수를 추출한다."""
    summary: dict[str, float] = {}
    for metric in RAGAS_METRICS:
        if metric in frame.columns:
            summary[metric] = float(pd.to_numeric(frame[metric], errors="coerce").mean())
    return summary


def _has_document_evidence(feedback: MonthlyFeedbackResult) -> bool:
    """최종 피드백 근거에 문서 evidence가 최소 하나 있는지 확인한다."""
    return any(evidence.evidence_type == "document" for evidence in feedback.key_evidences)


def _context_document_kind(context: RetrievedAdviceContext) -> str:
    """검색 컨텍스트의 문서 종류를 소문자 문자열로 안전하게 추출한다."""
    return (context.document_kind or "").strip().lower()


def _uses_saving_tips_context(contexts: Sequence[RetrievedAdviceContext]) -> bool:
    """월간 피드백 RAG에서 금지한 saving tips 문서가 검색 결과에 섞였는지 확인한다."""
    for context in contexts:
        blob = " ".join(
            [
                context.query,
                context.source,
                context.document_kind or "",
            ]
        ).lower()
        if "saving_tips" in blob or "saving tips" in blob:
            return True
    return False


def _has_allowed_monthly_context(contexts: Sequence[RetrievedAdviceContext]) -> bool:
    """월간 피드백 근거로 허용된 동향·벤치마크 문서가 검색되었는지 확인한다."""
    for context in contexts:
        document_kind = _context_document_kind(context)
        if document_kind in _MONTHLY_ALLOWED_DOCUMENT_KINDS:
            return True
        source = context.source.lower()
        if any(kind in source for kind in _MONTHLY_ALLOWED_DOCUMENT_KINDS):
            return True
    return False


def _message_uses_document_signal(feedback: MonthlyFeedbackResult) -> bool:
    """피드백 본문이 전체 사용자 비교나 소비 동향 문서 해석을 반영했는지 확인한다."""
    message = feedback.feedback_message
    if any(term in message for term in _DOCUMENT_SIGNAL_TERMS):
        return True
    return any(
        evidence.evidence_type == "document" and evidence.detail and evidence.detail in message
        for evidence in feedback.key_evidences
    )


def _mission_is_goal_like(mission: str) -> bool:
    """월간 미션이 구체 행동이 아닌 비율 감축·목표형 문장인지 판단한다."""
    normalized = _clean_text(mission)
    if _PERCENT_REDUCTION_RE.search(normalized) is not None:
        return True
    return any(term in normalized for term in _GENERIC_GOAL_TERMS)


def _mission_is_actionable(mission: str) -> bool:
    """월간 미션에 시점·대상·행동 중 실제 실행 행동이 드러나는지 판단한다."""
    normalized = _clean_text(mission)
    if not normalized or _mission_is_goal_like(normalized):
        return False
    return any(term in normalized for term in _ACTION_TERMS)


def _message_and_mission_are_not_repeated(feedback: MonthlyFeedbackResult) -> bool:
    """피드백 본문과 다음 달 미션이 같은 문장을 반복하지 않는지 확인한다."""
    message = _compact_text(feedback.feedback_message)
    mission = _compact_text(feedback.next_month_mission)
    if not message or not mission:
        return False
    return message != mission and mission not in message


def _contains_raw_percent_noise(feedback: MonthlyFeedbackResult) -> bool:
    """피드백 본문이나 미션에 읽기 어려운 원시 소수 퍼센트가 있는지 확인한다."""
    target_text = "\n".join([feedback.feedback_message, feedback.next_month_mission])
    return _RAW_PERCENT_RE.search(target_text) is not None


def _has_payment_fixed_cost_overclaim(
    feedback: MonthlyFeedbackResult,
    contexts: Sequence[RetrievedAdviceContext],
) -> bool:
    """납부 카테고리를 근거 없이 고정비로 단정했는지 확인한다."""
    target_text = "\n".join([feedback.feedback_message, feedback.next_month_mission])
    if "납부" not in target_text or "고정비" not in target_text:
        return False

    evidence_blob = "\n".join(
        [
            *[
                " ".join(
                    [
                        evidence.title,
                        evidence.detail,
                        evidence.source_json_path or "",
                    ]
                )
                for evidence in feedback.key_evidences
            ],
            *[context.content for context in contexts],
        ]
    )
    return not any(term in evidence_blob for term in ("반복", "자동이체", "고정비"))


def _make_rule(
    key: str,
    label: str,
    passed: bool,
    passed_detail: str,
    failed_detail: str,
) -> MonthlyRuleEvaluation:
    """규칙 평가 결과 모델을 일관된 형식으로 생성한다."""
    return MonthlyRuleEvaluation(
        key=key,
        label=label,
        passed=passed,
        detail=passed_detail if passed else failed_detail,
    )


def evaluate_monthly_feedback_rules(result: MonthlyFeedbackServiceResult) -> MonthlyRuleSummary:
    """월간 피드백 RAG 산출물을 프로젝트 전용 규칙 기반 평가로 채점한다."""
    if result.feedback is None:
        evaluations = [
            MonthlyRuleEvaluation(
                key="feedback_exists",
                label="피드백 생성",
                passed=False,
                detail="월간 피드백 결과가 없습니다.",
            )
        ]
        return MonthlyRuleSummary(
            pass_count=0,
            total_count=1,
            pass_rate=0.0,
            rule_evaluations=evaluations,
        )

    feedback = result.feedback
    contexts = result.retrieved_contexts
    has_contexts = bool(contexts)
    has_document_evidence = _has_document_evidence(feedback)
    no_saving_tips = not _uses_saving_tips_context(contexts)
    has_allowed_context = _has_allowed_monthly_context(contexts)
    uses_document_signal = _message_uses_document_signal(feedback)
    mission_actionable = _mission_is_actionable(feedback.next_month_mission)
    not_repeated = _message_and_mission_are_not_repeated(feedback)
    no_raw_percent = not _contains_raw_percent_noise(feedback)
    no_fixed_cost_overclaim = not _has_payment_fixed_cost_overclaim(feedback, contexts)

    evaluations = [
        _make_rule(
            "has_retrieved_contexts",
            "검색 문서 존재",
            has_contexts,
            f"{len(contexts)}개의 검색 문서가 있습니다.",
            "검색된 문서 근거가 없습니다.",
        ),
        _make_rule(
            "has_allowed_monthly_context",
            "월간 허용 문서 사용",
            has_allowed_context,
            "USER_REPORT, CATCHER_CONSUMPTION_BENCHMARK, KCA_REPORT 중 하나를 사용했습니다.",
            "월간 피드백 근거로 허용된 소비 동향 문서가 검색되지 않았습니다.",
        ),
        _make_rule(
            "no_saving_tips_context",
            "saving tips 배제",
            no_saving_tips,
            "saving tips 문서를 월간 피드백 RAG 근거로 사용하지 않았습니다.",
            "월간 피드백 RAG에서 제외해야 하는 saving tips 문서가 포함되었습니다.",
        ),
        _make_rule(
            "has_document_evidence",
            "문서 근거 포함",
            has_document_evidence,
            "key_evidences에 evidence_type=document 근거가 있습니다.",
            "key_evidences에 evidence_type=document 근거가 없습니다.",
        ),
        _make_rule(
            "message_uses_document_signal",
            "본문 문서 해석 반영",
            uses_document_signal,
            "feedback_message에 전체 사용자 비교나 소비 동향 문서 해석이 반영되었습니다.",
            "feedback_message에서 전체 사용자 비교나 소비 동향 문서 해석을 찾기 어렵습니다.",
        ),
        _make_rule(
            "mission_is_actionable",
            "실행형 미션",
            mission_actionable,
            "next_month_mission이 결제 전 확인·점검 같은 실행 행동으로 작성되었습니다.",
            "next_month_mission이 비율 감축 목표이거나 실행 행동이 부족합니다.",
        ),
        _make_rule(
            "message_mission_not_repeated",
            "본문/미션 역할 분리",
            not_repeated,
            "feedback_message와 next_month_mission이 같은 문장을 반복하지 않습니다.",
            "feedback_message와 next_month_mission이 같은 문장을 반복합니다.",
        ),
        _make_rule(
            "no_raw_percent_noise",
            "퍼센트 반올림",
            no_raw_percent,
            "원시 소수 퍼센트가 노출되지 않았습니다.",
            "읽기 어려운 원시 소수 퍼센트가 노출되었습니다.",
        ),
        _make_rule(
            "no_payment_fixed_cost_overclaim",
            "납부 고정비 단정 방지",
            no_fixed_cost_overclaim,
            "납부 카테고리를 근거 없이 고정비로 단정하지 않았습니다.",
            "납부 카테고리를 반복 납부나 자동이체 근거 없이 고정비로 단정했습니다.",
        ),
    ]
    pass_count = sum(1 for evaluation in evaluations if evaluation.passed)
    total_count = len(evaluations)
    return MonthlyRuleSummary(
        pass_count=pass_count,
        total_count=total_count,
        pass_rate=round(pass_count / total_count, 3),
        rule_evaluations=evaluations,
    )


def _format_context(context: RetrievedAdviceContext, index: int) -> str:
    """검색 문서 컨텍스트를 LLM Judge 프롬프트에 넣을 블록 문자열로 변환한다."""
    page = f", page={context.page_number}" if context.page_number is not None else ""
    return "\n".join(
        [
            f"[{index}] query={context.query}",
            f"source={context.source}, document_kind={context.document_kind or 'unknown'}{page}",
            f"content={context.content}",
        ]
    )


def build_monthly_llm_judge_prompt(result: MonthlyFeedbackServiceResult) -> str:
    """월간 피드백 RAG 산출물을 채점할 LLM Judge 한국어 프롬프트를 생성한다."""
    feedback_text = (
        format_monthly_feedback_response(result.feedback)
        if result.feedback is not None
        else "피드백 없음"
    )
    context_text = "\n\n".join(
        _format_context(context, index)
        for index, context in enumerate(result.retrieved_contexts, start=1)
    )
    query_text = "\n".join(f"- {query}" for query in result.retrieval_queries) or "- 없음"
    return f"""
너는 월간 피드백 RAG 평가자다. 검색 질의, 검색 문서, 최종 월간 피드백을 보고
아래 기준으로 0.0~1.0 점수를 채점한다.

평가 기준:
1. groundedness_score: 최종 피드백이 검색 문서와 JSON 근거에 기반했는가.
2. retrieval_use_score: USER_REPORT, CATCHER_CONSUMPTION_BENCHMARK, KCA_REPORT 근거를
   월간 소비 비교와 소비 동향 해석에 실제로 사용했는가.
3. product_fit_score: 월간 피드백 규칙을 지켰는가.
   - key_evidences에 evidence_type=document 근거가 최소 1개 있어야 한다.
   - saving tips 문서는 월간 피드백 RAG 근거로 사용하지 않는다.
   - USER_REPORT의 개인 변화와 전체 사용자 변화를 방향과 크기로 분리해 해석한다.
   - 납부 카테고리는 반복 납부, 자동이체, 고정 항목 근거가 있을 때만 고정비로 말한다.
   - next_month_mission은 목표형 문장이 아니라 결제 전 확인, 하루 보류, 목록 작성,
     자동이체 목록 점검 같은 실행 행동이어야 한다.
   - feedback_message와 next_month_mission은 같은 문장을 반복하지 않는다.

반드시 JSON 객체만 출력한다:
{{
  "groundedness_score": 0.0,
  "retrieval_use_score": 0.0,
  "product_fit_score": 0.0,
  "total_score": 0.0,
  "reason": "핵심 판단 이유"
}}

[검색 질의]
{query_text}

[검색 문서]
{context_text or "검색 문서 없음"}

[최종 월간 피드백]
{feedback_text}
""".strip()


def _message_content_to_text(message: BaseMessage) -> str:
    """LangChain 메시지 content 값을 사람이 읽을 수 있는 문자열로 변환한다."""
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _coerce_score(value: object) -> float | None:
    """LLM Judge JSON 값에서 0.0~1.0 범위 점수를 안전하게 추출한다."""
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            return None
    return None


def _parse_llm_judge_output(raw_output: str) -> MonthlyLlmJudgeResult:
    """LLM Judge 원문에서 JSON 점수와 판단 이유를 파싱한다."""
    match = _JSON_OBJECT_RE.search(raw_output)
    if match is None:
        return MonthlyLlmJudgeResult(raw_output=raw_output)

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return MonthlyLlmJudgeResult(raw_output=raw_output)
    if not isinstance(payload, dict):
        return MonthlyLlmJudgeResult(raw_output=raw_output)

    payload_dict = cast(dict[str, object], payload)
    return MonthlyLlmJudgeResult(
        raw_output=raw_output,
        total_score=_coerce_score(payload_dict.get("total_score")),
        groundedness_score=_coerce_score(payload_dict.get("groundedness_score")),
        product_fit_score=_coerce_score(payload_dict.get("product_fit_score")),
        retrieval_use_score=_coerce_score(payload_dict.get("retrieval_use_score")),
        reason=str(payload_dict.get("reason", "")),
    )


def run_monthly_llm_judge(
    result: MonthlyFeedbackServiceResult,
    *,
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> MonthlyLlmJudgeResult:
    """현재 설정의 채팅 모델을 LLM Judge로 사용해 월간 피드백 RAG를 평가한다."""
    config = settings or get_settings()
    prompt = build_monthly_llm_judge_prompt(result)
    response = get_chat_model(config, temperature=temperature).invoke(prompt)
    return _parse_llm_judge_output(_message_content_to_text(response))
