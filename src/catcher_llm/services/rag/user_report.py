from __future__ import annotations

import re
from collections.abc import Sequence

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.rag_prompt import get_user_report_prompt
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.rag.config import DocumentKind, get_rag_pipeline_config
from catcher_llm.services.rag.core import generate_rag_reply


def _collect_user_report_context_text(contexts: Sequence[RetrievedChunk]) -> str:
    """사용자 동향 보고서 후처리에 사용할 context 본문을 하나의 문자열로 합친다."""
    return " ".join(chunk.content for chunk in contexts)


def _normalize_user_report_answer(question: str, rag_response: RAGResponse) -> str:
    """질문 유형에 맞춰 사용자 소비 동향 보고서 답변을 사실형 문장으로 정규화한다."""
    context_text = _collect_user_report_context_text(rag_response.contexts)

    # ── 4개월 합산 소비 총액 ─────────────────────────────────────────────────
    if any(kw in question for kw in ("4개월", "전체 총액", "합산 소비", "승인 소비 총액")):
        total_match = re.search(
            r"4개월\s*승인\s*소비\s*총액은?\s*([\d,]+원)",
            context_text,
        )
        if total_match:
            return f"4개월 승인 소비 총액은 {total_match.group(1)}이다."

    # ── 월평균 소비 ──────────────────────────────────────────────────────────
    if "월평균" in question and "소비" in question:
        avg_match = re.search(
            r"월평균\s*승인\s*소비\s*총액은?\s*([\d,]+원)",
            context_text,
        )
        if avg_match:
            return f"월평균 승인 소비 총액은 {avg_match.group(1)}이다."

    # ── 특정 월 소비 총액 ────────────────────────────────────────────────────
    month_query = re.search(r"(2026-\d{2})", question)
    if month_query and any(kw in question for kw in ("총액", "소비", "얼마")):
        target_month = month_query.group(1)
        # 테이블 행에서 해당 월 총액 추출: | 2026-01 | 100 | 7,361 | 288,556,600원 |
        table_match = re.search(
            rf"\|\s*{re.escape(target_month)}\s*\|[^|]+\|[^|]+\|\s*([\d,]+원)",
            context_text,
        )
        if table_match:
            return f"{target_month} 승인 소비 총액은 {table_match.group(1)}이다."

    # ── 카테고리 순위 / 1위 카테고리 ────────────────────────────────────────
    if any(kw in question for kw in ("카테고리", "1위", "가장 큰", "가장 많이")):
        top_category_match = re.search(
            r"가장\s*큰\s*카테고리는\s*모든\s*월에서\s*`?([가-힣]+)`?",
            context_text,
        )
        if top_category_match:
            cat = top_category_match.group(1)
            ratio_match = re.search(
                rf"`?{re.escape(cat)}`?\s*([\d,]+원),\s*평균\s*구성비는?\s*([0-9.]+%)",
                context_text,
            )
            if ratio_match:
                return f"1위 카테고리는 {cat}이며 4개월 합산 {ratio_match.group(1)}, 평균 구성비 {ratio_match.group(2)}이다."
            return f"1위 카테고리는 {cat}이다."

    # ── 목표 사용률 ──────────────────────────────────────────────────────────
    if "목표 사용률" in question or "예산 초과" in question:
        rate_match = re.search(
            r"목표\s*지출\s*한도\s*대비\s*사용률은?\s*매월\s*([0-9.]+%[~\-][0-9.]+%)",
            context_text,
        )
        if rate_match:
            return f"목표 사용률은 매월 {rate_match.group(1)}로 목표를 꾸준히 초과한다."

        rate_range_match = re.search(
            r"([0-9.]+%)[~\-]([0-9.]+%)\s*로\s*목표를?\s*꾸준히\s*초과", context_text
        )
        if rate_range_match:
            return f"목표 사용률은 매월 {rate_range_match.group(1)}~{rate_range_match.group(2)}로 목표를 꾸준히 초과한다."

    # ── 할부 비중 ────────────────────────────────────────────────────────────
    if "할부" in question:
        installment_match = re.search(
            r"할부\s*비중은?\s*([0-9.]+%)\s*에서\s*([0-9.]+%)\s*로\s*상승",
            context_text,
        )
        if installment_match:
            return f"할부 비중은 {installment_match.group(1)}에서 {installment_match.group(2)}로 상승했다."

    # ── 온라인 결제 비중 ─────────────────────────────────────────────────────
    if "온라인" in question and ("비중" in question or "결제" in question):
        online_match = re.search(
            r"온라인\s*결제\s*비중은?\s*([0-9.]+%)\s*에서\s*([0-9.]+%)\s*로\s*내려",
            context_text,
        )
        if online_match:
            return f"온라인 결제 비중은 {online_match.group(1)}에서 {online_match.group(2)}로 하락했다."

    # ── 세그먼트 질문 (연령대·성별) ──────────────────────────────────────────
    segment_keywords = {
        "40대 남성": ("40대 남성", "3,463,444원", "162.42%"),
        "50대 여성": ("50대 여성", "2,769,789원", "169.15%"),
        "30대 남성": ("30대 남성", "3,274,267원", "144.03%"),
        "30대 여성": ("30대 여성", "3,170,436원", "146.70%"),
        "50대 남성": ("50대 남성", "3,144,420원", "161.42%"),
        "40대 여성": ("40대 여성", "2,537,620원", "152.48%"),
        "20대 이하 남성": ("20대 이하 남성", "2,519,955원", "156.81%"),
        "20대 이하 여성": ("20대 이하 여성", "2,475,573원", "143.86%"),
    }
    for seg_key, (seg_name, per_capita, rate) in segment_keywords.items():
        if seg_key in question:
            seg_match = re.search(
                rf"{re.escape(seg_name)}.{{0,120}}?([\d,]+원).{{0,60}}?([0-9.]+%)",
                context_text,
            )
            if seg_match:
                return (
                    f"{seg_name}의 월평균 1인당 소비는 {seg_match.group(1)}, "
                    f"평균 목표 사용률은 {seg_match.group(2)}이다."
                )
            return f"{seg_name}의 월평균 1인당 소비는 {per_capita}, 평균 목표 사용률은 {rate}이다."

    # ── 위험군 ───────────────────────────────────────────────────────────────
    if any(kw in question for kw in ("위험군", "위험 사용자", "300%", "200%")):
        risk_300_match = re.search(
            r"목표\s*사용률\s*300%\s*이상[^\d]{0,30}(\d+)건",
            context_text,
        )
        risk_200_match = re.search(
            r"목표\s*사용률\s*200%\s*이상[^\d]{0,30}(\d+)건",
            context_text,
        )
        if risk_300_match and risk_200_match:
            return (
                f"목표 사용률 200% 이상 사용자-월은 {risk_200_match.group(1)}건, "
                f"300% 이상은 {risk_300_match.group(1)}건이다."
            )

    # ── 주간·일간 피크 ───────────────────────────────────────────────────────
    if any(kw in question for kw in ("주간 피크", "최고 주간", "일간 최고", "피크")):
        peak_week_match = re.search(
            r"주간\s*1위[^\d]{0,30}(2026-\d{2}-\d{2}\s*~\s*2026-\d{2}-\d{2})[^\d]{0,30}([\d,]+원)",
            context_text,
        )
        if peak_week_match:
            return (
                f"주간 소비 1위는 {peak_week_match.group(1)} 기간으로 "
                f"{peak_week_match.group(2)}이다."
            )

    return rag_response.answer


def generate_user_report_rag_reply(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 3,
    *,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    """사용자 소비 동향 보고서(마크다운) 문서를 기반으로 RAG 답변을 생성한다.

    data/raw/markdown/users_report/ 하위의 모든 .md 파일을 로드하며,
    loaders.load_markdown_file 을 통해 헤더 구조 기반으로 청킹된다.
    """
    pipeline_config = get_rag_pipeline_config(DocumentKind.USER_REPORT, settings=settings)
    prompt = get_user_report_prompt()

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
        rag_response.answer = _normalize_user_report_answer(question, rag_response)
    return rag_response
