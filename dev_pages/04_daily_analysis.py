from __future__ import annotations

from datetime import date, timedelta
from typing import cast

import pandas as pd
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
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


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 거래 데이터 기준 일일 소비 분석 JSON을 확인합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("📊 일일 소비 분석")
st.caption("consumption_feedback.daily_analysis 서비스를 실행해 화면에서 결과를 점검합니다.")

controls = st.columns(3)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
analysis_day_input = controls[1].date_input("분석 기준일", value=date(2024, 3, 31))
analysis_day = cast(date, analysis_day_input)
previous_day_input = controls[2].date_input(
    "전일 비교 기준일",
    value=analysis_day - timedelta(days=1),
)
previous_day = cast(date, previous_day_input)

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
    time_slot_analysis = cast(JsonObject, result["time_slot_analysis"])
    payment_behavior_analysis = cast(JsonObject, result["payment_behavior_analysis"])
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
