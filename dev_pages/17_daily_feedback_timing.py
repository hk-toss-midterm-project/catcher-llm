from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import DailyFeedbackServiceResult
from catcher_llm.services.consumption_feedback.daily_feedback import (
    DailyFeedbackTimingRecord,
    generate_daily_feedback,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

_TIMING_RECORDS_KEY = "daily_feedback_timing_records"
_TIMING_RESULT_KEY = "daily_feedback_timing_result"


def _format_seconds(seconds: float) -> str:
    """초 단위 실행 시간을 개발 화면에 표시할 문자열로 변환한다."""
    return f"{seconds:.3f}s"


def _status_label(status: str) -> str:
    """타이밍 기록 상태 코드를 한글 화면 표시값으로 변환한다."""
    return "성공" if status == "success" else "실패"


def _records_to_frame(records: Sequence[DailyFeedbackTimingRecord]) -> pd.DataFrame:
    """일일 피드백 타이밍 기록 목록을 표와 차트에 쓸 DataFrame으로 변환한다."""
    total_seconds = sum(record.elapsed_seconds for record in records)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        ratio = record.elapsed_seconds / total_seconds * 100 if total_seconds else 0.0
        rows.append(
            {
                "순서": index,
                "단계": record.step_name,
                "키": record.step_key,
                "상태": _status_label(record.status),
                "소요 시간(초)": round(record.elapsed_seconds, 3),
                "비중(%)": round(ratio, 1),
                "이유/상세": record.detail or "-",
                "오류": record.error or "",
            }
        )
    return pd.DataFrame(rows)


def _render_timing_summary(records: Sequence[DailyFeedbackTimingRecord]) -> None:
    """단계별 타이밍 기록의 총합과 최장 단계를 요약해 표시한다."""
    total_seconds = sum(record.elapsed_seconds for record in records)
    slowest_record = max(records, key=lambda record: record.elapsed_seconds)
    failed_count = sum(1 for record in records if record.status == "error")

    metric_columns = st.columns(3)
    metric_columns[0].metric("총 소요 시간", _format_seconds(total_seconds))
    metric_columns[1].metric("가장 오래 걸린 단계", slowest_record.step_name)
    metric_columns[2].metric("실패 단계", f"{failed_count}개")


def _render_timing_chart(frame: pd.DataFrame) -> None:
    """단계별 소요 시간을 가로 막대 차트로 표시한다."""
    if frame.empty:
        return

    fig = px.bar(
        frame,
        x="소요 시간(초)",
        y="단계",
        color="상태",
        orientation="h",
        text="소요 시간(초)",
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
    """일일 피드백 생성 결과의 성공/실패와 최종 JSON을 표시한다."""
    if result.error:
        st.error(f"일일 피드백 생성 실패: {result.error}")
    elif result.feedback is not None:
        st.success(result.feedback.summary_title)
        st.write(result.feedback.scolding_message)
        st.info(result.feedback.tomorrow_mission)
    else:
        st.warning("피드백 결과가 비어 있습니다.")

    with st.expander("최종 피드백 JSON"):
        st.json(result.feedback.model_dump() if result.feedback is not None else {})

    with st.expander("RAG 검색 질의"):
        st.write(result.retrieval_queries)


def _render_timing_records(records: Sequence[DailyFeedbackTimingRecord]) -> None:
    """수집된 타이밍 기록을 요약, 표, 차트 순서로 표시한다."""
    if not records:
        st.info("아직 수집된 타이밍 기록이 없습니다.")
        return

    _render_timing_summary(records)
    frame = _records_to_frame(records)
    st.subheader("단계별 소요 시간")
    st.dataframe(frame, width="stretch", hide_index=True)
    _render_timing_chart(frame)


settings = get_settings()

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("일일 피드백 생성 서비스의 단계별 실행 시간을 측정합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Raw data: `{settings.raw_data_dir}`")

st.title("⏱️ 일일 피드백 소요 시간")
st.caption(
    "generate_daily_feedback 내부의 분석, 해석 체인, RAG, 최종 피드백, 저장 단계를 측정합니다."
)

render_date_picker_styles()
control_columns = st.columns(3)
member_id = control_columns[0].number_input("Member ID", min_value=1, value=1, step=1)
with control_columns[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_feedback_timing_day",
    )
with control_columns[2]:
    previous_day = select_daily_date(
        "전일 기준일",
        default=DEFAULT_CALENDAR_DATE - timedelta(days=1),
        key="daily_feedback_timing_previous_day",
    )

retrieval_columns = st.columns(4)
chunk_size = retrieval_columns[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_columns[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_columns[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_columns[3].number_input("Max queries", min_value=1, value=4, step=1)

st.warning(
    "이 페이지는 실제 시간을 측정하기 위해 기존 session 캐시를 읽어 건너뛰지 않고 "
    "generate_daily_feedback을 실행합니다."
)

if st.button("타이밍 측정 실행", width="stretch"):
    timing_records: list[DailyFeedbackTimingRecord] = []
    with st.spinner("일일 피드백 생성 시간을 측정 중입니다..."):
        result = generate_daily_feedback(
            member_id=int(member_id),
            analysis_date=analysis_day,
            previous_date=previous_day,
            settings=settings,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            top_k=int(top_k),
            max_queries=int(max_queries),
            timing_callback=timing_records.append,
        )

    st.session_state[_TIMING_RECORDS_KEY] = timing_records
    st.session_state[_TIMING_RESULT_KEY] = result

records = cast(
    Sequence[DailyFeedbackTimingRecord],
    st.session_state.get(_TIMING_RECORDS_KEY, []),
)
result = cast(DailyFeedbackServiceResult | None, st.session_state.get(_TIMING_RESULT_KEY))

if result is None:
    st.info("Member ID와 날짜를 선택한 뒤 타이밍 측정 실행을 눌러주세요.")
    st.stop()

_render_timing_records(records)
_render_generation_result(result)
