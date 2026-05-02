from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import cast

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.chains.consumption_feedback import build_spending_analysis_chain
from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import (
    CategoryShiftIndicator,
    HighSpendingItem,
    SpendingIndicatorPayload,
    SpendingMetric,
    TimeSlotComparison,
    UserSpendingData,
)
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_spending_indicators,
    make_spending_analysis_input,
    parse_user_spending_data,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

settings = get_settings()


def _models_to_frame(
    models: Sequence[BaseModel],
    *,
    string_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Pydantic 모델 목록을 Streamlit 표로 렌더링할 DataFrame으로 변환한다."""
    columns_to_stringify = string_columns or set()
    rows: list[dict[str, object]] = []
    for model in models:
        row = cast(dict[str, object], model.model_dump())
        for column_name in columns_to_stringify:
            if column_name in row:
                row[column_name] = str(row[column_name])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_sqlite_daily_user_data(
    *,
    member_id: int,
    analysis_day: date,
    previous_day: date,
) -> UserSpendingData:
    """SQLite 일일 분석 JSON을 생성한 뒤 소비 해석 입력 모델로 변환한다."""
    raw_result = build_daily_consumption_analysis_json(
        member_id=member_id,
        analysis_date=analysis_day,
        previous_date=previous_day,
        settings=settings,
    )
    return parse_user_spending_data(raw_result)


def _render_metric_table(metrics: Sequence[SpendingMetric]) -> None:
    """핵심 소비 지표 목록을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(metrics, string_columns={"value"})
    if frame.empty:
        st.info("표시할 핵심 지표가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_category_table(changes: Sequence[CategoryShiftIndicator]) -> None:
    """카테고리 비중 변화 지표를 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(changes)
    if frame.empty:
        st.info("표시할 카테고리 변화 지표가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_high_spending_table(items: Sequence[HighSpendingItem]) -> None:
    """고액 지출 항목을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(items)
    if frame.empty:
        st.info("고액 지출 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_time_slot_table(items: Sequence[TimeSlotComparison]) -> None:
    """시간대별 소비 차이 지표를 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(items)
    if frame.empty:
        st.info("시간대별 소비 지표가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _find_metric(metrics: Sequence[SpendingMetric], name: str) -> SpendingMetric | None:
    """지표 이름으로 핵심 소비 지표를 찾아 화면 요약에 사용할 수 있게 반환한다."""
    for metric in metrics:
        if metric.name == name:
            return metric
    return None


def _metric_value_text(metrics: Sequence[SpendingMetric], name: str) -> str:
    """지표 이름에 해당하는 값을 Streamlit metric 값 문자열로 변환한다."""
    metric = _find_metric(metrics, name)
    if metric is None:
        return "-"
    if metric.unit == "KRW" and isinstance(metric.value, int | float):
        return f"{metric.value:,.0f}원"
    if metric.unit == "percent" and isinstance(metric.value, int | float):
        return f"{metric.value:,.2f}%"
    if metric.unit == "count":
        return f"{metric.value}건"
    return str(metric.value)


def _render_payment_behavior_summary(metrics: Sequence[SpendingMetric]) -> None:
    """지출 마찰력과 결제 밀도 지표를 개발 확인용 메트릭으로 표시한다."""
    st.subheader("지출 마찰력 및 결제 밀도")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "마찰력 없는 지출 비중",
        _metric_value_text(metrics, "마찰력 없는 지출 비중"),
    )
    metric_columns[1].metric(
        "마찰력 없는 지출액",
        _metric_value_text(metrics, "마찰력 없는 지출액"),
    )
    metric_columns[2].metric(
        "마찰력 없는 결제 건수",
        _metric_value_text(metrics, "마찰력 없는 지출 건수"),
    )
    metric_columns[3].metric(
        "오늘 결제 횟수",
        _metric_value_text(metrics, "오늘 결제 횟수"),
    )
    metric_columns[4].metric(
        "건당 평균 금액",
        _metric_value_text(metrics, "1회 결제당 평균 금액"),
    )


def _render_financial_context_summary(metrics: Sequence[SpendingMetric]) -> None:
    """소득과 목표 소비 기반 일일 해석 지표를 개발 확인용 메트릭으로 표시한다."""
    st.subheader("소득·목표 소비 지표")
    budget_columns = st.columns(4)
    budget_columns[0].metric(
        "일일 잔여 예산",
        _metric_value_text(metrics, "일일 잔여 예산"),
    )
    budget_columns[1].metric(
        "일일 예산 초과액",
        _metric_value_text(metrics, "일일 예산 초과액"),
    )
    budget_columns[2].metric(
        "일 환산 소득 대비 소비율",
        _metric_value_text(metrics, "일 환산 소득 대비 소비율"),
    )
    budget_columns[3].metric(
        "월 누적 목표 사용률",
        _metric_value_text(metrics, "월 누적 목표 사용률"),
    )

    projection_columns = st.columns(3)
    projection_columns[0].metric(
        "월말 예상 소비",
        _metric_value_text(metrics, "월말 예상 소비"),
    )
    projection_columns[1].metric(
        "월말 예상 목표 사용률",
        _metric_value_text(metrics, "월말 예상 목표 사용률"),
    )
    projection_columns[2].metric(
        "월말까지 남은 하루 허용 소비",
        _metric_value_text(metrics, "월말까지 남은 하루 허용 소비"),
    )


def _render_indicator_summary(indicators: SpendingIndicatorPayload) -> None:
    """추출된 소비 지표 묶음의 핵심 요약과 상세 표를 표시한다."""
    metric_columns = st.columns(4)
    metric_columns[0].metric("Member ID", indicators.member_id)
    metric_columns[1].metric("분석 기준일", indicators.analysis_date)
    metric_columns[2].metric(
        "최대 증가 카테고리",
        indicators.largest_category_increase.category
        if indicators.largest_category_increase
        else "-",
    )
    metric_columns[3].metric(
        "최대 감소 카테고리",
        indicators.largest_category_decrease.category
        if indicators.largest_category_decrease
        else "-",
    )

    st.subheader("핵심 소비 지표")
    _render_metric_table(indicators.metrics)

    _render_payment_behavior_summary(indicators.metrics)

    _render_financial_context_summary(indicators.metrics)

    st.subheader("카테고리 비중 변화")
    _render_category_table(indicators.category_ratio_changes)

    st.subheader("고액 지출 항목")
    _render_high_spending_table(indicators.high_spending_items)

    st.subheader("시간대별 소비 차이")
    _render_time_slot_table(indicators.time_slot_diffs)


def _render_chain_result(result: dict[str, object]) -> None:
    """구조화 분석 체인 결과를 단계별 JSON으로 표시한다."""
    sections = [
        ("pattern_result", "소비 패턴 분석"),
        ("problem_result", "문제 소비 식별"),
        ("cause_result", "소비 원인 해석"),
        ("action_result", "행동 개선 제안"),
    ]
    for key, title in sections:
        if key not in result:
            continue
        value = result[key]
        st.subheader(title)
        if isinstance(value, BaseModel):
            st.json(value.model_dump())
        else:
            st.json(value)


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 일일 소비 분석 JSON 기반 지표 추출과 구조화 해석 체인을 점검합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("🧭 소비 해석 체인")
st.caption("일일 분석 결과를 소비 지표로 변환하고 구조화 해석 체인을 실행합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_interpretation_day",
    )
previous_day = analysis_day - timedelta(days=1)

try:
    user_data = _load_sqlite_daily_user_data(
        member_id=int(member_id),
        analysis_day=analysis_day,
        previous_day=previous_day,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

indicators = extract_spending_indicators(user_data)
analysis_input = make_spending_analysis_input(user_data)

_render_indicator_summary(indicators)

with st.expander("체인 입력 JSON"):
    st.code(analysis_input["indicator_json"], language="json")
    st.code(analysis_input["raw_json"], language="json")

chat_model_error = settings.chat_model_error
if chat_model_error is not None:
    st.warning(
        settings.get_chat_model_error_message("consumption interpretation flow")
        or "Chat model configuration is invalid."
    )
elif st.button("구조화 분석 체인 실행", width="stretch"):
    with st.spinner("소비 해석 체인 실행 중..."):
        try:
            chain = build_spending_analysis_chain(settings=settings)
            chain_result = chain.invoke(analysis_input)
        except Exception as error:
            st.error(f"체인 실행에 실패했습니다: {error}")
            st.stop()

    _render_chain_result(chain_result)
