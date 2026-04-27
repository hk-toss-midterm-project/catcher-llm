from __future__ import annotations

import re
from collections.abc import Sequence

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.rag_prompt import get_self_report_prompt
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.rag.config import DocumentKind, get_rag_pipeline_config
from catcher_llm.services.rag.core import generate_rag_reply


def _collect_self_report_context_text(contexts: Sequence[RetrievedChunk]) -> str:
    """소비 자기진단 리포트 후처리에 사용할 context 본문을 하나의 문자열로 합친다."""
    return " ".join(chunk.content for chunk in contexts)


def _normalize_self_report_answer(question: str, rag_response: RAGResponse) -> str:
    """질문 유형에 맞춰 소비 자기진단 리포트 답변을 보고서 사실형 문장으로 정규화한다."""
    context_text = _collect_self_report_context_text(rag_response.contexts)

    if "회원월당 소비성 금액" in question:
        trend_amount_match = re.search(
            r"회원월당\s+소비성\s+금액은?\s*([0-9.]+\s*만\s*원)이었고\s*201812\s+에는\s*([0-9.]+\s*만\s*원)",
            context_text,
        )
        if trend_amount_match:
            amount = re.sub(r"\s+", "", trend_amount_match.group(2)).replace("만원", "만 원")
            return f"201812 회원월당 소비성 금액은 {amount}이다."

        amount_match = re.search(
            r"201812.{0,80}?회원월당\s+소비성\s+금액(?:은|[^\d]{0,20})\s*([0-9.]+\s*만\s*원)",
            context_text,
        )
        if amount_match:
            amount = re.sub(r"\s+", "", amount_match.group(1)).replace("만원", "만 원")
            return f"201812 회원월당 소비성 금액은 {amount}이다."

    if "변동비 누수 후보 비율" in question:
        trend_ratio_match = re.search(
            r"변동비\s+누수\s+후보\s+비율은?\s*([0-9.]+%)\s*에서\s*([0-9.]+%)\s*로",
            context_text,
        )
        if trend_ratio_match:
            return f"201812 변동비 누수 후보 비율은 {trend_ratio_match.group(2)}이다."

        ratio_match = re.search(
            r"201812.{0,80}?변동비\s+누수\s+후보\s+비율(?:은|[^\d]{0,20})\s*([0-9.]+%)",
            context_text,
        )
        if ratio_match:
            return f"201812 변동비 누수 후보 비율은 {ratio_match.group(1)}이다."

    if "절약 타깃" in question:
        priority_match = re.search(
            r"절약\s+타깃은?\s*([가-힣A-Za-z0-9]+)\s*,\s*([가-힣A-Za-z0-9]+)\s*,\s*([가-힣A-Za-z0-9]+)\s*순",
            context_text,
        )
        if priority_match:
            first, second, third = priority_match.groups()
            return f"절약 타깃은 {first}, {second}, {third} 순이다."

    if "소비후잔액부담지수" in question:
        burden_match = re.search(
            r"소비후잔액부담지수는?\s*([0-9.]+)로?\s*([0-9]+개월\s*최고치)?", context_text
        )
        if burden_match:
            burden_value = burden_match.group(1)
            burden_suffix = burden_match.group(2)
            if burden_suffix:
                return f"소비후잔액부담지수는 {burden_value}로 {re.sub(r'\\s+', '', burden_suffix)}이다."
            return f"소비후잔액부담지수는 {burden_value}이다."

    return rag_response.answer


def generate_self_report_rag_reply(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 2,
    *,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    """소비 자기진단 리포트 문서 설정을 주입해 RAG 답변을 생성한다."""
    pipeline_config = get_rag_pipeline_config(DocumentKind.SELF_REPORT, settings=settings)
    prompt = get_self_report_prompt()

    rag_response = generate_rag_reply(
        question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        history=history,
        raw_data_dir=pipeline_config.raw_data_dir,
        source_files=pipeline_config.source_files,
        settings=settings,
        prompt=prompt,
        temperature=temperature,
    )
    if isinstance(rag_response, RAGResponse):
        rag_response.answer = _normalize_self_report_answer(question, rag_response)
    return rag_response
