from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import BaseModel

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import DailyFeedbackServiceResult, UserSpendingData
from catcher_llm.services.consumption_feedback.daily_feedback import (
    DailyFeedbackTimingRecord,
    generate_daily_feedback,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

_UNIFIED_RESULT_KEY = "daily_feedback_unified_result"
_UNIFIED_TIMING_KEY = "daily_feedback_unified_timing"


def _format_amount(value: int | float) -> str:
    """금액 값을 원화 표시 문자열로 변환한다."""
    return f"{value:,.0f}원"


def _format_percent(value: int | float) -> str:
    """비율 값을 퍼센트 표시 문자열로 변환한다."""
    return f"{value:,.2f}%"


def _models_to_frame(models: Sequence[BaseModel]) -> pd.DataFrame:
    """Pydantic 모델 목록을 개발 확인용 DataFrame으로 변환한다."""
    return pd.DataFrame([model.model_dump() for model in models])


def _timing_records_to_frame(records: Sequence[DailyFeedbackTimingRecord]) -> pd.DataFrame:
    """통합 피드백 생성 단계별 타이밍 기록을 표와 차트용 DataFrame으로 변환한다."""
    total_seconds = sum(record.elapsed_seconds for record in records)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        rows.append(
            {
                "순서": index,
                "단계": record.step_name,
                "키": record.step_key,
                "상태": "실패" if record.status == "error" else "성공",
                "소요 시간(초)": round(record.elapsed_seconds, 3),
                "비중(%)": round(record.elapsed_seconds / total_seconds * 100, 1)
                if total_seconds
                else 0.0,
                "상세": record.detail or "-",
                "오류": record.error or "",
            }
        )
    return pd.DataFrame(rows)


def _render_daily_analysis_summary(daily_analysis: UserSpendingData | None) -> None:
    """통합 피드백 생성에 사용된 일일 분석 핵심 지표를 표시한다."""
    if daily_analysis is None:
        return

    frictionless_spending = daily_analysis.payment_behavior_analysis.frictionless_spending
    transaction_density = daily_analysis.payment_behavior_analysis.transaction_density
    stable_metrics = daily_analysis.stable_metrics

    st.subheader("일일 분석 요약")
    metric_columns = st.columns(5)
    metric_columns[0].metric("오늘 총 소비", _format_amount(stable_metrics.today_total))
    metric_columns[1].metric(
        "평소 대비 증감률",
        _format_percent(stable_metrics.increase_rate_percent),
    )
    metric_columns[2].metric(
        "마찰력 없는 지출 비중",
        _format_percent(frictionless_spending.ratio_percent),
    )
    metric_columns[3].metric(
        "마찰력 없는 지출액",
        _format_amount(frictionless_spending.total_amount),
    )
    metric_columns[4].metric(
        "건당 평균 금액",
        _format_amount(transaction_density.average_amount_per_transaction),
    )


def _render_timing_records(records: Sequence[DailyFeedbackTimingRecord]) -> None:
    """통합 방식 일일 피드백 생성 단계별 실행 시간을 표시한다."""
    if not records:
        return

    frame = _timing_records_to_frame(records)
    st.subheader("통합 방식 단계별 소요 시간")
    st.dataframe(frame, width="stretch", hide_index=True)

    fig = px.bar(
        frame,
        x="소요 시간(초)",
        y="단계",
        orientation="h",
        text="소요 시간(초)",
        color="상태",
        color_discrete_map={"성공": "#2563eb", "실패": "#dc2626"},
    )
    fig.update_traces(texttemplate="%{text:.3f}s", textposition="outside")
    fig.update_layout(
        height=max(360, 42 * len(frame)),
        margin=dict(t=20, b=20, l=20, r=40),
        xaxis_title="소요 시간(초)",
        yaxis_title=None,
        yaxis=dict(autorange="reversed"),
        legend_title_text=None,
    )
    st.plotly_chart(fig, width="stretch")


def _render_generation_result(result: DailyFeedbackServiceResult) -> None:
    """통합 방식 일일 피드백 결과와 세부 JSON을 표시한다."""
    st.subheader("통합 방식 피드백 결과")
    if result.error:
        st.error(f"통합 방식 일일 피드백 생성 실패: {result.error}")
    elif result.feedback is None:
        st.warning("피드백 결과가 비어 있습니다.")
    else:
        st.success(result.feedback.summary_title)
        st.write(result.feedback.scolding_message)
        st.info(result.feedback.tomorrow_mission)

        st.subheader("피드백 근거")
        evidence_frame = _models_to_frame(result.feedback.key_evidences)
        if evidence_frame.empty:
            st.info("표시할 피드백 근거가 없습니다.")
        else:
            st.dataframe(evidence_frame, width="stretch", hide_index=True)

        st.subheader("오늘 할 일")
        action_frame = _models_to_frame(result.feedback.action_items)
        if action_frame.empty:
            st.info("표시할 행동 항목이 없습니다.")
        else:
            st.dataframe(action_frame, width="stretch", hide_index=True)

    _render_daily_analysis_summary(result.daily_analysis)

    with st.expander("소비 해석 JSON"):
        st.json(result.interpretation_result or {})
    with st.expander("최종 피드백 JSON"):
        st.json(result.feedback.model_dump() if result.feedback is not None else {})
    with st.expander("RAG 검색 질의"):
        st.write(result.retrieval_queries)
    with st.expander("메모리/프로필 컨텍스트 JSON"):
        st.json(
            {
                "user_profile": result.user_profile.model_dump()
                if result.user_profile is not None
                else {},
                "memory_context": result.memory_context.model_dump()
                if result.memory_context is not None
                else {},
            }
        )


settings = get_settings()

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("통합 해석 체인으로 최종 일일 피드백을 생성합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Chat model: `{settings.chat_provider} / {settings.chat_model_name}`")

st.title("📣 일일 통합 피드백")
st.caption("기존 페이지를 유지한 채, 통합 해석 모드(unified)로 일일 피드백 결과를 생성합니다.")

render_date_picker_styles()
control_columns = st.columns(3)
member_id = control_columns[0].number_input("Member ID", min_value=1, value=1, step=1)
with control_columns[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_feedback_unified_day",
    )
with control_columns[2]:
    previous_day = select_daily_date(
        "전일 기준일",
        default=DEFAULT_CALENDAR_DATE - timedelta(days=1),
        key="daily_feedback_unified_previous_day",
    )

retrieval_columns = st.columns(4)
chunk_size = retrieval_columns[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_columns[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_columns[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_columns[3].number_input("Max queries", min_value=1, value=4, step=1)

st.warning(
    "이 페이지는 통합 방식 결과 확인용이므로 기존 session 캐시를 읽어 건너뛰지 않고 "
    "통합 해석 모드로 generate_daily_feedback을 실행합니다."
)

if st.button("통합 방식 일일 피드백 생성", width="stretch"):
    timing_records: list[DailyFeedbackTimingRecord] = []
    with st.spinner("통합 방식 일일 피드백 생성 중..."):
        result = generate_daily_feedback(
            member_id=int(member_id),
            analysis_date=analysis_day,
            previous_date=previous_day,
            settings=settings,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            top_k=int(top_k),
            max_queries=int(max_queries),
            interpretation_mode="unified",
            timing_callback=timing_records.append,
        )

    st.session_state[_UNIFIED_RESULT_KEY] = result
    st.session_state[_UNIFIED_TIMING_KEY] = timing_records

result = cast(DailyFeedbackServiceResult | None, st.session_state.get(_UNIFIED_RESULT_KEY))
timing_records = cast(
    Sequence[DailyFeedbackTimingRecord],
    st.session_state.get(_UNIFIED_TIMING_KEY, []),
)

if result is None:
    st.info("Member ID와 날짜를 선택한 뒤 통합 방식 일일 피드백 생성을 눌러주세요.")
    st.stop()

_render_generation_result(result)
_render_timing_records(timing_records)
