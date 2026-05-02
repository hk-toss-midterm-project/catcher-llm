from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pandas as pd

from catcher_llm.schemas.consumption_feedback import JsonObject, JsonValue

_REQUIRED_COLUMNS = {
    "멤버 id",
    "id",
    "사용 금액",
    "사용 시간",
    "결제 내역",
    "업종 카테고리",
    "결제 방식 (온/오프라인)",
}
_FRICTIONLESS_KEYWORDS = ["온라인", "간편결제", "앱결제", "배달"]
_ESSENTIAL_CATEGORIES = {"교통", "의료", "생활"}
_LATE_NIGHT_START_HOUR = 22
_TIME_SLOT_ORDER = {
    "1.새벽(00-06)": 1,
    "2.오전(06-11)": 2,
    "3.점심/오후(11-17)": 3,
    "4.저녁(17-21)": 4,
    "5.밤/야식(21-24)": 5,
}


def _validate_columns(frame: pd.DataFrame, label: str) -> None:
    """입력 데이터에 일일 소비 분석 필수 컬럼이 모두 있는지 확인한다."""
    missing_columns = sorted(_REQUIRED_COLUMNS.difference(str(column) for column in frame.columns))
    if missing_columns:
        raise ValueError(f"{label} 데이터에 필요한 컬럼이 없습니다: {missing_columns}")


def _parse_analysis_date(value: str | date) -> date:
    """문자열 또는 date 입력을 일 단위 분석 기준일로 변환한다."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _get_time_slot(hour: int) -> str:
    """결제 시각의 시간을 원본 노트북과 동일한 5개 시간대로 분류한다."""
    if 0 <= hour < 6:
        return "1.새벽(00-06)"
    if 6 <= hour < 11:
        return "2.오전(06-11)"
    if 11 <= hour < 17:
        return "3.점심/오후(11-17)"
    if 17 <= hour < 21:
        return "4.저녁(17-21)"
    return "5.밤/야식(21-24)"


def _safe_rate(numerator: float, denominator: float) -> float:
    """0으로 나누는 상황을 막고 비율 계산 결과를 반환한다."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round_float(value: float | int, digits: int = 4) -> float:
    """JSON 결과에서 사용하기 쉽게 실수 지표의 소수 자릿수를 제한한다."""
    return round(float(value), digits)


def _to_amount(value: float | int) -> int:
    """원화 합계처럼 정수로 표현할 금액 값을 JSON용 int로 변환한다."""
    return int(round(float(value)))


def _resolve_daily_budget(
    *,
    analysis_day: date,
    daily_budget: float | int | None,
    monthly_budget: float | int | None,
) -> float | None:
    """명시 일일 예산이 없으면 월 목표 소비 한도를 일 단위 예산으로 환산한다."""
    if daily_budget is not None:
        return float(daily_budget)
    if monthly_budget is None:
        return None
    month_day_count = calendar.monthrange(analysis_day.year, analysis_day.month)[1]
    return _safe_rate(float(monthly_budget), float(month_day_count))


def _build_daily_financial_metrics(
    past_member_frame: pd.DataFrame,
    *,
    analysis_day: date,
    today_total: float,
    daily_budget: float | int | None,
    monthly_budget: float | int | None,
    monthly_income: float | int | None,
) -> JsonObject:
    """사용자 목표 소비 한도와 월소득을 기준으로 일일 예산·소득 지표를 계산한다."""
    month_day_count = calendar.monthrange(analysis_day.year, analysis_day.month)[1]
    resolved_daily_budget = _resolve_daily_budget(
        analysis_day=analysis_day,
        daily_budget=daily_budget,
        monthly_budget=monthly_budget,
    )
    daily_income = (
        _safe_rate(float(monthly_income), float(month_day_count))
        if monthly_income is not None
        else None
    )
    month_start = date(analysis_day.year, analysis_day.month, 1)
    month_to_date_past_total = float(
        past_member_frame[
            (past_member_frame["date"] >= month_start) & (past_member_frame["date"] < analysis_day)
        ]["사용 금액"].sum()
    )
    month_to_date_total = month_to_date_past_total + today_total
    projected_monthly_spending = (
        _safe_rate(
            month_to_date_total,
            float(max(analysis_day.day, 1)),
        )
        * month_day_count
    )
    remaining_month_days = max(month_day_count - analysis_day.day, 0)
    remaining_month_budget = (
        max(float(monthly_budget) - month_to_date_total, 0.0)
        if monthly_budget is not None
        else None
    )

    return {
        "daily_budget_usage_rate_percent": None
        if resolved_daily_budget is None
        else _round_float(_safe_rate(today_total, resolved_daily_budget) * 100),
        "daily_remaining_budget": None
        if resolved_daily_budget is None
        else _to_amount(max(resolved_daily_budget - today_total, 0.0)),
        "daily_overspend_amount": None
        if resolved_daily_budget is None
        else _to_amount(max(today_total - resolved_daily_budget, 0.0)),
        "daily_income_usage_rate_percent": None
        if daily_income is None
        else _round_float(_safe_rate(today_total, daily_income) * 100),
        "month_to_date_budget_usage_rate_percent": None
        if monthly_budget is None
        else _round_float(_safe_rate(month_to_date_total, float(monthly_budget)) * 100),
        "projected_monthly_spending": _to_amount(projected_monthly_spending),
        "projected_monthly_budget_usage_rate_percent": None
        if monthly_budget is None
        else _round_float(_safe_rate(projected_monthly_spending, float(monthly_budget)) * 100),
        "required_daily_budget_until_month_end": None
        if remaining_month_budget is None
        else _to_amount(_safe_rate(remaining_month_budget, float(remaining_month_days)))
        if remaining_month_days > 0
        else 0,
    }


def _main_category(frame: pd.DataFrame) -> str | None:
    """주어진 거래 데이터에서 사용 금액 합계가 가장 큰 업종 카테고리를 찾는다."""
    if frame.empty:
        return None

    category_totals = frame.groupby("업종 카테고리")["사용 금액"].sum()
    if category_totals.empty:
        return None
    return str(category_totals.idxmax())


def _build_single_day_comparison(
    today_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    label: str,
    reference_date: date,
    today_total: float,
    today_count: int,
) -> JsonObject:
    """당일 소비를 특정 기준일 소비와 비교하는 공통 JSON 블록을 만든다."""
    reference_total = float(reference_frame["사용 금액"].sum())
    reference_count = int(len(reference_frame))
    amount_diff = today_total - reference_total

    return {
        "label": label,
        "reference_date": str(reference_date),
        "reference_total": _to_amount(reference_total),
        "today_total": _to_amount(today_total),
        "amount_diff": _to_amount(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, reference_total) * 100),
        "reference_count": reference_count,
        "today_count": today_count,
        "count_diff": today_count - reference_count,
        "reference_main_category": _main_category(reference_frame),
        "today_main_category": _main_category(today_frame),
    }


def _get_available_daily_reference_dates(
    past_member_frame: pd.DataFrame,
    candidate_dates: list[date],
) -> list[date]:
    """거래 데이터 범위 안에 있는 일일 비교 후보 날짜만 남긴다."""
    if past_member_frame.empty:
        return []

    first_available_date = cast(date, past_member_frame["date"].min())
    return [
        reference_date
        for reference_date in candidate_dates
        if reference_date >= first_available_date
    ]


def _build_same_weekday_average_comparison(
    past_member_frame: pd.DataFrame,
    today_frame: pd.DataFrame,
    *,
    analysis_day: date,
    today_total: float,
    today_count: int,
    week_count: int = 4,
) -> JsonObject:
    """최근 N주 같은 요일의 평균 소비와 당일 소비를 비교하는 JSON 블록을 만든다."""
    candidate_dates = [
        analysis_day - timedelta(days=7 * index) for index in range(1, week_count + 1)
    ]
    reference_dates = _get_available_daily_reference_dates(past_member_frame, candidate_dates)
    daily_totals = past_member_frame.groupby("date")["사용 금액"].sum()
    daily_counts = past_member_frame.groupby("date").size()

    reference_rows: list[JsonValue] = []
    reference_total_sum = 0.0
    reference_count_sum = 0
    for reference_date in reference_dates:
        reference_total = float(daily_totals.get(reference_date, 0.0))
        reference_count = int(daily_counts.get(reference_date, 0))
        reference_total_sum += reference_total
        reference_count_sum += reference_count
        reference_rows.append(
            {
                "date": str(reference_date),
                "total": _to_amount(reference_total),
                "transaction_count": reference_count,
            }
        )

    reference_day_count = len(reference_dates)
    average_total = _safe_rate(reference_total_sum, float(reference_day_count))
    average_count = _safe_rate(float(reference_count_sum), float(reference_day_count))
    amount_diff = today_total - average_total
    count_diff = float(today_count) - average_count

    return {
        "label": f"최근 {week_count}주 같은 요일 평균 대비",
        "reference_dates": [str(reference_date) for reference_date in reference_dates],
        "reference_days": reference_rows,
        "reference_day_count": reference_day_count,
        "average_total": _round_float(average_total),
        "today_total": _to_amount(today_total),
        "amount_diff": _round_float(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, average_total) * 100),
        "average_count": _round_float(average_count),
        "today_count": today_count,
        "count_diff": _round_float(count_diff),
        "today_main_category": _main_category(today_frame),
    }


def _build_category_ratio_changes(
    category_comparison: pd.DataFrame,
) -> list[JsonValue]:
    """평소 대비 당일 업종 카테고리 비중 변화 목록을 JSON 값 목록으로 만든다."""
    rows: list[JsonValue] = []
    for category, row in category_comparison.sort_values("diff_point", ascending=False).iterrows():
        rows.append(
            {
                "category": str(category),
                "usual_ratio_percent": _round_float(float(cast(float, row["usual_ratio_percent"]))),
                "today_ratio_percent": _round_float(float(cast(float, row["today_ratio_percent"]))),
                "diff_point": _round_float(float(cast(float, row["diff_point"]))),
            }
        )
    return rows


def _build_high_spending_items(
    high_spending_frame: pd.DataFrame,
) -> list[JsonValue]:
    """당일 거래 중 평소 상한선을 초과한 고액 결제 목록을 JSON 값 목록으로 만든다."""
    rows: list[JsonValue] = []
    for _, row in high_spending_frame.sort_values("사용 금액", ascending=False).iterrows():
        used_at = pd.to_datetime(row["사용 시간"]).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "used_at": str(used_at),
                "description": str(row["결제 내역"]),
                "amount": _to_amount(float(cast(float, row["사용 금액"]))),
                "category": str(row["업종 카테고리"]),
            }
        )
    return rows


def _build_time_slot_rows(time_comparison: pd.DataFrame) -> list[JsonValue]:
    """평소 대비 당일 시간대별 소비 금액 비교 목록을 JSON 값 목록으로 만든다."""
    rows: list[JsonValue] = []
    for slot, row in time_comparison.iterrows():
        today_amount = float(cast(float, row["today_amount"]))
        usual_average_amount = float(cast(float, row["usual_average_amount"]))
        rows.append(
            {
                "time_slot": str(slot),
                "today_amount": _to_amount(today_amount),
                "usual_average_amount": _round_float(usual_average_amount),
                "diff_amount": _round_float(today_amount - usual_average_amount),
            }
        )
    return rows


def _build_payment_behavior_analysis(
    today_frame: pd.DataFrame,
    *,
    today_total: float,
    today_count: int,
) -> JsonObject:
    """당일 지출 마찰력과 결제 밀도 지표를 노트북 계산 방식으로 만든다."""
    payment_channel = today_frame["결제 방식 (온/오프라인)"].astype("string")
    frictionless_pattern = "|".join(_FRICTIONLESS_KEYWORDS)
    frictionless_mask = payment_channel.str.contains(frictionless_pattern, na=False)
    frictionless_frame = today_frame[frictionless_mask]
    frictionless_total = float(frictionless_frame["사용 금액"].sum())
    frictionless_count = int(len(frictionless_frame))
    frictionless_ratio = _safe_rate(frictionless_total, today_total) * 100
    average_amount_per_transaction = _safe_rate(today_total, float(today_count))

    return {
        "frictionless_spending": {
            "keywords": list(_FRICTIONLESS_KEYWORDS),
            "transaction_count": frictionless_count,
            "total_amount": _to_amount(frictionless_total),
            "ratio_percent": _round_float(frictionless_ratio),
        },
        "transaction_density": {
            "transaction_count": today_count,
            "average_amount_per_transaction": _round_float(average_amount_per_transaction),
        },
    }


def _build_daily_category_spending(
    today_frame: pd.DataFrame, today_total: float
) -> list[JsonValue]:
    """문서형 일일 지표에 사용할 카테고리별 소비 금액·비중·건수를 만든다."""
    rows: list[JsonValue] = []
    category_amounts = today_frame.groupby("업종 카테고리")["사용 금액"].sum()
    category_counts = today_frame.groupby("업종 카테고리").size()
    for category, amount in category_amounts.sort_values(ascending=False).items():
        amount_float = float(amount)
        rows.append(
            {
                "category": str(category),
                "total_amount": _to_amount(amount_float),
                "ratio_percent": _round_float(_safe_rate(amount_float, today_total) * 100),
                "transaction_count": int(category_counts.get(category, 0)),
            }
        )
    return rows


def _build_daily_metrics(
    past_member_frame: pd.DataFrame,
    today_frame: pd.DataFrame,
    time_comparison: pd.DataFrame,
    *,
    today_total: float,
    today_count: int,
    past_daily_original_avg: float,
    spike_ratio: float,
    analysis_day: date,
    daily_budget: float | int | None,
    monthly_budget: float | int | None,
    monthly_income: float | int | None,
) -> JsonObject:
    """문서의 일일 소비 분석 10개 핵심 지표와 특수 지표를 계산한다."""
    daily_average_transaction_amount = _safe_rate(today_total, float(today_count))
    daily_max_transaction_amount = (
        float(today_frame["사용 금액"].max()) if not today_frame.empty else 0.0
    )

    if "hour" in today_frame.columns:
        late_night_frame = today_frame[today_frame["hour"] >= _LATE_NIGHT_START_HOUR]
    else:
        late_night_frame = today_frame[
            pd.to_datetime(today_frame["사용 시간"]).dt.hour >= _LATE_NIGHT_START_HOUR
        ]
    late_night_amount = float(late_night_frame["사용 금액"].sum())
    late_night_ratio = _safe_rate(late_night_amount, today_total)

    category_spending = _build_daily_category_spending(today_frame, today_total)
    financial_metrics = _build_daily_financial_metrics(
        past_member_frame,
        analysis_day=analysis_day,
        today_total=today_total,
        daily_budget=daily_budget,
        monthly_budget=monthly_budget,
        monthly_income=monthly_income,
    )

    nonessential_total = float(
        today_frame[~today_frame["업종 카테고리"].astype("string").isin(_ESSENTIAL_CATEGORIES)][
            "사용 금액"
        ].sum()
    )
    nonessential_ratio = _safe_rate(nonessential_total, today_total)
    past_daily_counts = past_member_frame.groupby("date").size()
    past_daily_count_avg = float(past_daily_counts.mean()) if not past_daily_counts.empty else 0.0
    count_increase_rate = _safe_rate(
        float(today_count) - past_daily_count_avg, past_daily_count_avg
    )
    impulse_spending_score = (
        late_night_ratio * 0.3 + nonessential_ratio * 0.4 + count_increase_rate * 0.3
    )

    return {
        "daily_total_amount": _to_amount(today_total),
        "daily_transaction_count": today_count,
        "daily_average_transaction_amount": _round_float(daily_average_transaction_amount),
        "daily_max_transaction_amount": _to_amount(daily_max_transaction_amount),
        "time_slot_amounts": _build_time_slot_rows(time_comparison),
        "late_night_ratio_percent": _round_float(late_night_ratio * 100),
        "category_spending": category_spending,
        "daily_budget_usage_rate_percent": financial_metrics["daily_budget_usage_rate_percent"],
        "daily_remaining_budget": financial_metrics["daily_remaining_budget"],
        "daily_overspend_amount": financial_metrics["daily_overspend_amount"],
        "daily_income_usage_rate_percent": financial_metrics["daily_income_usage_rate_percent"],
        "month_to_date_budget_usage_rate_percent": financial_metrics[
            "month_to_date_budget_usage_rate_percent"
        ],
        "projected_monthly_spending": financial_metrics["projected_monthly_spending"],
        "projected_monthly_budget_usage_rate_percent": financial_metrics[
            "projected_monthly_budget_usage_rate_percent"
        ],
        "required_daily_budget_until_month_end": financial_metrics[
            "required_daily_budget_until_month_end"
        ],
        "no_spending_day": bool(today_count == 0),
        "daily_anomaly_score": _round_float(spike_ratio),
        "special_metrics": {
            "impulse_spending_score": _round_float(impulse_spending_score),
            "daily_spending_risk": _round_float(_safe_rate(today_total, past_daily_original_avg)),
        },
    }


def _prepare_member_frames(
    past_frame: pd.DataFrame,
    today_full_frame: pd.DataFrame,
    *,
    member_id: int,
    analysis_day: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """과거와 기준일 DataFrame을 멤버/날짜 기준으로 필터링하고 계산 타입을 정리한다."""
    past_member_frame = past_frame[past_frame["멤버 id"] == member_id].copy()
    if past_member_frame.empty:
        raise ValueError(f"과거 데이터에서 멤버 {member_id}번 거래를 찾을 수 없습니다.")

    past_member_frame["사용 금액"] = pd.to_numeric(past_member_frame["사용 금액"])
    past_member_frame["사용 시간"] = pd.to_datetime(past_member_frame["사용 시간"])
    past_member_frame["date"] = past_member_frame["사용 시간"].dt.date

    today_frame = today_full_frame.copy()
    today_frame["사용 금액"] = pd.to_numeric(today_frame["사용 금액"])
    today_frame["사용 시간"] = pd.to_datetime(today_frame["사용 시간"])
    today_frame = today_frame[
        (today_frame["멤버 id"] == member_id) & (today_frame["사용 시간"].dt.date == analysis_day)
    ].copy()
    return past_member_frame, today_frame


def _build_category_comparison(
    past_member_frame: pd.DataFrame,
    today_frame: pd.DataFrame,
    *,
    today_total: float,
) -> pd.DataFrame:
    """평소와 당일 업종 카테고리별 소비 비중 비교 DataFrame을 만든다."""
    past_clipped_total = float(past_member_frame["사용 금액_clipped"].sum())
    past_category_ratio = (
        past_member_frame.groupby("업종 카테고리")["사용 금액_clipped"].sum()
        / past_clipped_total
        * 100
    )
    today_category_ratio = (
        today_frame.groupby("업종 카테고리")["사용 금액"].sum() / today_total * 100
        if today_total > 0
        else pd.Series(dtype="float64")
    )
    category_comparison = pd.DataFrame(
        {
            "usual_ratio_percent": past_category_ratio,
            "today_ratio_percent": today_category_ratio,
        }
    ).fillna(0)
    category_comparison["diff_point"] = (
        category_comparison["today_ratio_percent"] - category_comparison["usual_ratio_percent"]
    )
    return category_comparison


def _build_time_comparison(
    past_member_frame: pd.DataFrame,
    today_frame: pd.DataFrame,
) -> tuple[str | None, pd.DataFrame]:
    """평소와 당일 시간대별 소비 금액 비교 DataFrame과 피크 시간대를 만든다."""
    today_frame["hour"] = today_frame["사용 시간"].dt.hour
    today_frame["time_slot"] = today_frame["hour"].apply(_get_time_slot)
    past_member_frame["hour"] = past_member_frame["사용 시간"].dt.hour
    past_member_frame["time_slot"] = past_member_frame["hour"].apply(_get_time_slot)
    num_past_days = int(past_member_frame["date"].nunique())

    past_time_distribution = (
        past_member_frame.groupby("time_slot")["사용 금액"].sum() / num_past_days
    )
    today_time_distribution = today_frame.groupby("time_slot")["사용 금액"].sum()
    time_comparison = pd.DataFrame(
        {
            "today_amount": today_time_distribution,
            "usual_average_amount": past_time_distribution,
        }
    ).fillna(0)
    time_comparison["sort_order"] = [
        int(_TIME_SLOT_ORDER[str(slot)]) for slot in time_comparison.index
    ]
    time_comparison = time_comparison.sort_values("sort_order")
    peak_slot = None if today_time_distribution.empty else str(today_time_distribution.idxmax())
    return peak_slot, time_comparison


def build_daily_consumption_analysis_from_frames(
    past_frame: pd.DataFrame,
    today_full_frame: pd.DataFrame,
    *,
    member_id: int = 1,
    analysis_date: str | date = "2024-04-01",
    previous_date: str | date = "2024-03-31",
    past_source_path: str | Path | None = None,
    today_source_path: str | Path | None = None,
    daily_budget: float | int | None = None,
    monthly_budget: float | int | None = None,
    monthly_income: float | int | None = None,
) -> JsonObject:
    """과거/기준일 소비 DataFrame에서 파이프라인용 일일 소비 분석 JSON을 만든다."""
    _validate_columns(past_frame, "과거")
    _validate_columns(today_full_frame, "기준일")

    analysis_day = _parse_analysis_date(analysis_date)
    previous_day = _parse_analysis_date(previous_date)
    past_member_frame, today_frame = _prepare_member_frames(
        past_frame,
        today_full_frame,
        member_id=member_id,
        analysis_day=analysis_day,
    )

    q1 = float(past_member_frame["사용 금액"].quantile(0.25))
    q3 = float(past_member_frame["사용 금액"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    past_member_frame["사용 금액_clipped"] = past_member_frame["사용 금액"].clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    past_daily_stable_avg = float(
        past_member_frame.groupby("date")["사용 금액_clipped"].sum().mean()
    )
    today_total = float(today_frame["사용 금액"].sum())
    increase_rate = _safe_rate(today_total - past_daily_stable_avg, past_daily_stable_avg) * 100
    category_comparison = _build_category_comparison(
        past_member_frame,
        today_frame,
        today_total=today_total,
    )

    high_spending_frame = today_frame[today_frame["사용 금액"] > upper_bound].copy()
    past_daily_original_avg = float(past_member_frame.groupby("date")["사용 금액"].sum().mean())
    spike_ratio = _safe_rate(today_total, past_daily_original_avg)

    yesterday_frame = past_member_frame[past_member_frame["date"] == previous_day].copy()
    yesterday_total = float(yesterday_frame["사용 금액"].sum())
    yesterday_count = int(len(yesterday_frame))
    today_count = int(len(today_frame))
    day_diff = today_total - yesterday_total
    day_diff_rate = _safe_rate(day_diff, yesterday_total) * 100
    peak_slot, time_comparison = _build_time_comparison(past_member_frame, today_frame)

    source_paths: JsonObject = {
        "past_source": str(past_source_path) if past_source_path is not None else None,
        "today_source": str(today_source_path) if today_source_path is not None else None,
    }
    outlier_thresholds: JsonObject = {
        "q1": _round_float(q1),
        "q3": _round_float(q3),
        "iqr": _round_float(iqr),
        "lower_bound": _round_float(lower_bound),
        "upper_bound": _round_float(upper_bound),
    }
    stable_metrics: JsonObject = {
        "past_daily_stable_average": _round_float(past_daily_stable_avg),
        "today_total": _to_amount(today_total),
        "increase_rate_percent": _round_float(increase_rate),
        "category_ratio_changes": _build_category_ratio_changes(category_comparison),
    }
    anomaly_detection: JsonObject = {
        "past_daily_original_average": _round_float(past_daily_original_avg),
        "spike_ratio": _round_float(spike_ratio),
        "is_spike": bool(spike_ratio > 1.5),
        "high_spending_threshold": _round_float(upper_bound),
        "high_spending_items": _build_high_spending_items(high_spending_frame),
    }
    previous_day_comparison: JsonObject = {
        "yesterday_date": str(previous_day),
        "yesterday_total": _to_amount(yesterday_total),
        "today_total": _to_amount(today_total),
        "amount_diff": _to_amount(day_diff),
        "amount_diff_rate_percent": _round_float(day_diff_rate),
        "yesterday_count": yesterday_count,
        "today_count": today_count,
        "count_diff": today_count - yesterday_count,
        "yesterday_main_category": _main_category(yesterday_frame),
        "today_main_category": _main_category(today_frame),
    }
    same_weekday_last_week = analysis_day - timedelta(days=7)
    same_weekday_last_week_frame = past_member_frame[
        past_member_frame["date"] == same_weekday_last_week
    ].copy()
    daily_comparisons: JsonObject = {
        "previous_day": _build_single_day_comparison(
            today_frame,
            yesterday_frame,
            label="어제 대비",
            reference_date=previous_day,
            today_total=today_total,
            today_count=today_count,
        ),
        "same_weekday_last_week": _build_single_day_comparison(
            today_frame,
            same_weekday_last_week_frame,
            label="지난주 같은 요일 대비",
            reference_date=same_weekday_last_week,
            today_total=today_total,
            today_count=today_count,
        ),
        "recent_4week_same_weekday_average": _build_same_weekday_average_comparison(
            past_member_frame,
            today_frame,
            analysis_day=analysis_day,
            today_total=today_total,
            today_count=today_count,
            week_count=4,
        ),
    }
    time_slot_analysis: JsonObject = {
        "peak_slot": peak_slot,
        "time_slots": _build_time_slot_rows(time_comparison),
    }
    payment_behavior_analysis = _build_payment_behavior_analysis(
        today_frame,
        today_total=today_total,
        today_count=today_count,
    )
    daily_metrics = _build_daily_metrics(
        past_member_frame,
        today_frame,
        time_comparison,
        today_total=today_total,
        today_count=today_count,
        past_daily_original_avg=past_daily_original_avg,
        spike_ratio=spike_ratio,
        analysis_day=analysis_day,
        daily_budget=daily_budget,
        monthly_budget=monthly_budget,
        monthly_income=monthly_income,
    )
    return {
        "member_id": member_id,
        "analysis_date": str(analysis_day),
        "source_paths": source_paths,
        "outlier_thresholds": outlier_thresholds,
        "stable_metrics": stable_metrics,
        "anomaly_detection": anomaly_detection,
        "previous_day_comparison": previous_day_comparison,
        "daily_comparisons": daily_comparisons,
        "time_slot_analysis": time_slot_analysis,
        "payment_behavior_analysis": payment_behavior_analysis,
        "daily_metrics": daily_metrics,
    }
