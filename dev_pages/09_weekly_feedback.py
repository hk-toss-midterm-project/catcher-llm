from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.config.settings import get_settings
from catcher_llm.prompts.persona_prompt import PERSONAS
from catcher_llm.schemas.consumption_feedback import (
    RetrievedAdviceContext,
    UserProfileContext,
    WeeklyFeedbackAction,
    WeeklyFeedbackEvidence,
    WeeklySpendingData,
)
from catcher_llm.services.consumption_feedback.weekly_feedback import generate_weekly_feedback
from catcher_llm.ui.components import render_readonly_control
from catcher_llm.ui.date_picker import render_date_picker_styles, select_week_range

_WEEKLY_PERSONA_KEY = "weekly_persona"

settings = get_settings()


def _models_to_frame(models: Sequence[BaseModel]) -> pd.DataFrame:
    """Pydantic 모델 목록을 Streamlit 표로 렌더링할 DataFrame으로 변환한다."""
    return pd.DataFrame([model.model_dump() for model in models])


def _format_amount(value: int | float) -> str:
    """주간 분석 금액 지표를 원화 표시 문자열로 변환한다."""
    return f"{value:,.0f}원"


def _format_percent(value: int | float) -> str:
    """주간 분석 비율 지표를 퍼센트 표시 문자열로 변환한다."""
    return f"{value:,.2f}%"


def _render_evidence_table(evidences: Sequence[WeeklyFeedbackEvidence]) -> None:
    """주간 피드백 근거 목록을 표로 표시한다."""
    frame = _models_to_frame(evidences)
    if frame.empty:
        st.info("표시할 피드백 근거가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_action_table(actions: Sequence[WeeklyFeedbackAction]) -> None:
    """주간 피드백 행동 항목 목록을 표로 표시한다."""
    frame = _models_to_frame(actions)
    if frame.empty:
        st.info("표시할 행동 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_contexts(contexts: Sequence[RetrievedAdviceContext]) -> None:
    """RAG에서 수집한 문서 청크를 쿼리와 출처별 expander로 표시한다."""
    if not contexts:
        st.info("검색된 문서 근거가 없습니다.")
        return

    for index, context in enumerate(contexts, start=1):
        page_label = f" | p.{context.page_number}" if context.page_number is not None else ""
        with st.expander(
            f"{index}. {context.query} | {context.source}{page_label}",
            expanded=index == 1,
        ):
            st.write(context.content)


def _render_weekly_analysis_summary(weekly_analysis: WeeklySpendingData | None) -> None:
    """최종 피드백에 사용된 주간 분석 핵심 지표를 화면에 요약 표시한다."""
    if weekly_analysis is None:
        return

    weekly_summary = weekly_analysis.weekly_summary
    waste_detection = weekly_analysis.waste_detection
    repeat_patterns = weekly_analysis.repeat_patterns

    st.subheader("주간 소비 핵심 지표")
    metric_columns = st.columns(5)
    metric_columns[0].metric("이번 주 총 소비", _format_amount(weekly_summary.this_week_total))
    metric_columns[1].metric("전주 대비", _format_percent(weekly_summary.diff_rate_percent))
    metric_columns[2].metric("결제 건수", f"{weekly_summary.transaction_count}건")
    metric_columns[3].metric("배달 소비", _format_amount(repeat_patterns.delivery.total_amount))
    metric_columns[4].metric("고액 결제", f"{waste_detection.high_spending.count}건")


def _render_profile_context(user_profile: UserProfileContext | None) -> None:
    """서비스가 최종 피드백에 전달한 사용자 프로필을 JSON으로 표시한다."""
    with st.expander("사용자 프로필 JSON"):
        st.json(user_profile.model_dump() if user_profile is not None else {})


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("주간 분석, 해석, RAG 조회, 최종 주간 소비 피드백을 한 번에 실행합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Raw data: `{settings.raw_data_dir}`")


st.title("🧾 주간 피드백")
st.caption("generate_weekly_feedback 서비스를 실행해 최종 주간 소비 피드백 결과를 점검합니다.")

render_date_picker_styles()
controls = st.columns(3)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    week_start, week_end = select_week_range(
        "분석 주",
        default_start=date(2024, 4, 1),
        key="weekly_feedback_week",
    )
with controls[2]:
    render_readonly_control("분석 주 종료일", week_end)

retrieval_controls = st.columns(4)
chunk_size = retrieval_controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_controls[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_controls[3].number_input("Max queries", min_value=1, value=4, step=1)

_WEEKLY_RADIO_KEY = f"{_WEEKLY_PERSONA_KEY}_radio"
_weekly_persona_label_to_key = {info["label"]: key for key, info in PERSONAS.items()}
_weekly_persona_labels = list(_weekly_persona_label_to_key.keys())

# 라디오 위젯 자체의 session_state를 읽어야 타이밍 없이 즉시 제목에 반영된다
_weekly_selected_label = st.session_state.get(_WEEKLY_RADIO_KEY)
_weekly_selected_key = _weekly_persona_label_to_key.get(_weekly_selected_label) if _weekly_selected_label else None
_weekly_expander_title = (
    "🎭 페르소나"
    if _weekly_selected_key is None
    else f"🎭 페르소나 — {_weekly_selected_label}"
)
with st.expander(_weekly_expander_title, expanded=False):
    st.radio(
        "피드백을 전달할 페르소나를 선택하세요",
        options=_weekly_persona_labels,
        index=None,
        key=_WEEKLY_RADIO_KEY,
    )

if st.button("주간 피드백 생성", width="stretch"):
    with st.spinner("주간 피드백 생성 중..."):
        result = generate_weekly_feedback(
            member_id=int(member_id),
            week_start=week_start,
            week_end=week_end,
            settings=settings,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            top_k=int(top_k),
            max_queries=int(max_queries),
            persona_key=_weekly_selected_key,
        )

    if result.error:
        st.error(f"주간 피드백 생성 실패: {result.error}")
        if result.retrieval_queries:
            st.subheader("생성된 RAG 검색 질의")
            st.write(result.retrieval_queries)
        _render_weekly_analysis_summary(result.weekly_analysis)
        if result.weekly_analysis is not None:
            with st.expander("주간 분석 JSON"):
                st.json(result.weekly_analysis.model_dump())
        if result.interpretation_result:
            with st.expander("소비 해석 JSON"):
                st.json(result.interpretation_result)
        _render_profile_context(result.user_profile)
        st.stop()

    if result.feedback is None:
        st.warning("피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback

    st.subheader(feedback.summary_title)
    st.write(feedback.feedback_message)
    st.info(feedback.next_week_mission)

    _render_weekly_analysis_summary(result.weekly_analysis)

    st.subheader("피드백 근거")
    _render_evidence_table(feedback.key_evidences)

    st.subheader("다음 주 할 일")
    _render_action_table(feedback.action_items)

    st.subheader("RAG 검색 질의")
    st.write(result.retrieval_queries)

    st.subheader("검색된 문서 근거")
    _render_contexts(result.retrieved_contexts)

    _render_profile_context(result.user_profile)

    with st.expander("주간 분석 JSON"):
        if result.weekly_analysis is not None:
            st.json(result.weekly_analysis.model_dump())

    with st.expander("소비 해석 JSON"):
        st.json(result.interpretation_result or {})

    with st.expander("최종 피드백 JSON"):
        st.json(feedback.model_dump())
