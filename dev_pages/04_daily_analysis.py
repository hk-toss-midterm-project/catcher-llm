from __future__ import annotations

from datetime import timedelta
from typing import cast

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

settings = get_settings()


def _format_amount(value: object) -> str:
    """일일 소비 분석 JSON의 금액 값을 원화 표시 문자열로 변환한다."""
    if isinstance(value, int | float):
        return f"{value:,.0f}원"
    return "-"


def _format_percent(value: object) -> str:
    """일일 소비 분석 JSON의 비율 값을 퍼센트 표시 문자열로 변환한다."""
    if isinstance(value, int | float):
        return f"{value:,.2f}%"
    return "-"


def _json_rows_to_frame(rows: object) -> pd.DataFrame:
    """JSON 목록 값을 Streamlit 표 렌더링에 사용할 DataFrame으로 변환한다."""
    if not isinstance(rows, list):
        return pd.DataFrame()
    json_rows = [row for row in rows if isinstance(row, dict)]
    return pd.DataFrame(json_rows)


def _render_document_daily_metrics(daily_metrics: JsonObject) -> None:
    """기존 요약과 겹치지 않는 문서 기준 일일 추가 지표를 화면에 표시한다."""
    st.subheader("문서 기준 일일 추가 지표")
    special_metrics = cast(JsonObject, daily_metrics["special_metrics"])
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "야간 소비 비중",
        _format_percent(daily_metrics["late_night_ratio_percent"]),
    )
    metric_columns[1].metric(
        "일일 예산 소진율",
        _format_percent(daily_metrics["daily_budget_usage_rate_percent"]),
    )
    metric_columns[2].metric("무소비일 여부", "Y" if daily_metrics["no_spending_day"] else "N")
    metric_columns[3].metric("충동소비 점수", str(special_metrics["impulse_spending_score"]))
    metric_columns[4].metric("하루 소비 위험도", f"{special_metrics['daily_spending_risk']}x")


def _render_daily_comparison_card(
    column: DeltaGenerator,
    label: str,
    comparison: JsonObject,
) -> None:
    """일일 비교 결과 하나를 증감액과 기준일 맥락이 있는 카드로 표시한다."""
    column.metric(
        label,
        _format_amount(comparison["amount_diff"]),
        _format_percent(comparison["amount_diff_rate_percent"]),
    )
    column.caption(
        f"기준 {comparison['reference_date']} · "
        f"{_format_amount(comparison['reference_total'])} · "
        f"{comparison['reference_count']}건"
    )


def _render_daily_comparisons(daily_comparisons: JsonObject) -> None:
    """어제·지난주 같은 요일·최근 4주 같은 요일 평균 비교를 화면에 표시한다."""
    st.subheader("확장 일일 비교")
    previous_day = cast(JsonObject, daily_comparisons["previous_day"])
    same_weekday = cast(JsonObject, daily_comparisons["same_weekday_last_week"])
    recent_average = cast(JsonObject, daily_comparisons["recent_4week_same_weekday_average"])

    columns = st.columns(3)
    _render_daily_comparison_card(columns[0], "어제 대비", previous_day)
    _render_daily_comparison_card(columns[1], "지난주 같은 요일 대비", same_weekday)
    columns[2].metric(
        "최근 4주 같은 요일 평균 대비",
        _format_amount(recent_average["amount_diff"]),
        _format_percent(recent_average["amount_diff_rate_percent"]),
    )
    columns[2].caption(
        f"기준 {recent_average['reference_day_count']}일 평균 · "
        f"{_format_amount(recent_average['average_total'])} · "
        f"{recent_average['average_count']}건"
    )

    reference_day_frame = _json_rows_to_frame(recent_average["reference_days"])
    if reference_day_frame.empty:
        st.info("최근 4주 같은 요일 평균에 사용할 기준일 데이터가 없습니다.")
    else:
        st.dataframe(reference_day_frame, use_container_width=True, hide_index=True)


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 거래 데이터 기준 일일 소비 분석 JSON을 확인합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("📊 일일 소비 분석")
st.caption("consumption_feedback.daily_analysis 서비스를 실행해 화면에서 결과를 점검합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_analysis_day",
    )
previous_day = analysis_day - timedelta(days=1)

if st.button("일일 분석 실행", use_container_width=True):
    with st.spinner("일일 소비 분석 JSON 생성 중..."):
        try:
            result = build_daily_consumption_analysis_json(
                member_id=int(member_id),
                analysis_date=analysis_day,
                previous_date=previous_day,
                settings=settings,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    stable_metrics = cast(JsonObject, result["stable_metrics"])
    anomaly_detection = cast(JsonObject, result["anomaly_detection"])
    previous_day_comparison = cast(JsonObject, result["previous_day_comparison"])
    daily_comparisons = cast(JsonObject, result["daily_comparisons"])
    time_slot_analysis = cast(JsonObject, result["time_slot_analysis"])
    payment_behavior_analysis = cast(JsonObject, result["payment_behavior_analysis"])
    daily_metrics = cast(JsonObject, result["daily_metrics"])
    frictionless_spending = cast(
        JsonObject,
        payment_behavior_analysis["frictionless_spending"],
    )
    transaction_density = cast(
        JsonObject,
        payment_behavior_analysis["transaction_density"],
    )
    source_paths = cast(JsonObject, result["source_paths"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("오늘 총 소비", _format_amount(stable_metrics["today_total"]))
    metric_columns[1].metric(
        "전일 총 소비", _format_amount(previous_day_comparison["yesterday_total"])
    )
    metric_columns[2].metric(
        "평소 대비 증감률",
        _format_percent(stable_metrics["increase_rate_percent"]),
    )
    metric_columns[3].metric("피크 시간대", str(time_slot_analysis["peak_slot"] or "-"))

    _render_document_daily_metrics(daily_metrics)
    _render_daily_comparisons(daily_comparisons)

    st.subheader("지출 마찰력 및 결제 밀도")
    payment_columns = st.columns(5)
    payment_columns[0].metric(
        "마찰력 없는 지출 비중",
        _format_percent(frictionless_spending["ratio_percent"]),
    )
    payment_columns[1].metric(
        "마찰력 없는 지출액",
        _format_amount(frictionless_spending["total_amount"]),
    )
    payment_columns[2].metric(
        "마찰력 없는 결제 건수",
        f"{frictionless_spending['transaction_count']}건",
    )
    payment_columns[3].metric(
        "오늘 결제 횟수",
        f"{transaction_density['transaction_count']}건",
    )
    payment_columns[4].metric(
        "건당 평균 금액",
        _format_amount(transaction_density["average_amount_per_transaction"]),
    )
    st.caption(
        "마찰력 없는 지출 키워드: "
        + ", ".join(str(keyword) for keyword in frictionless_spending["keywords"])
    )

    st.subheader("이상 소비")
    anomaly_columns = st.columns(3)
    anomaly_columns[0].metric(
        "스파이크 여부",
        "Y" if anomaly_detection["is_spike"] else "N",
    )
    anomaly_columns[1].metric("스파이크 배율", f"{anomaly_detection['spike_ratio']}x")
    anomaly_columns[2].metric(
        "고액 결제 기준",
        _format_amount(anomaly_detection["high_spending_threshold"]),
    )

    high_spending_frame = _json_rows_to_frame(anomaly_detection["high_spending_items"])
    if high_spending_frame.empty:
        st.info("고액 결제 기준을 초과한 당일 거래가 없습니다.")
    else:
        st.dataframe(high_spending_frame, use_container_width=True, hide_index=True)

    st.subheader("카테고리 비중 변화")
    category_ratio_frame = _json_rows_to_frame(stable_metrics["category_ratio_changes"])
    if category_ratio_frame.empty:
        st.info("표시할 카테고리 비중 변화가 없습니다.")
    else:
        st.dataframe(category_ratio_frame, use_container_width=True, hide_index=True)

    st.subheader("시간대별 소비")
    time_slot_frame = _json_rows_to_frame(time_slot_analysis["time_slots"])
    if time_slot_frame.empty:
        st.info("표시할 시간대별 소비 데이터가 없습니다.")
    else:
        st.dataframe(time_slot_frame, use_container_width=True, hide_index=True)

    st.caption(
        f"past_source: `{source_paths['past_source']}` | "
        f"today_source: `{source_paths['today_source']}`"
    )
    with st.expander("Raw JSON"):
        st.json(result)
