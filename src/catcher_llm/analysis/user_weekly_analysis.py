from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from catcher_llm.analysis.ratio_context import build_ratio_context_warning
from catcher_llm.schemas.consumption_feedback import JsonObject, JsonValue

# ===== 가맹점 분류 키워드 (노트북과 동일) =====
_DELIVERY_KEYWORDS = ["배달의민족", "쿠팡이츠"]
_CAFE_KEYWORDS = [
    "스타벅스",
    "이디야",
    "투썸플레이스",
    "할리스",
    "커피빈",
    "폴바셋",
    "빽다방",
    "메가커피",
    "컴포즈",
]
_CONVENIENCE_KEYWORDS = ["CU", "GS25", "세븐일레븐", "미니스톱", "emart24", "이마트24"]
_TAXI_KEYWORDS = ["카카오택시", "타다", "우티"]

_MICRO_THRESHOLD: int = 10_000
_LATE_NIGHT_HOUR: int = 21
_WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

_REQUIRED_COLUMNS = {
    "멤버 id",
    "사용 금액",
    "사용 시간",
    "결제 내역",
    "업종 카테고리",
}


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------


def _validate_columns(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(str(c) for c in frame.columns))
    if missing:
        raise ValueError(f"{label} 데이터에 필요한 컬럼이 없습니다: {missing}")


def _round_float(value: float | int, digits: int = 4) -> float:
    return round(float(value), digits)


def _to_amount(value: float | int) -> int:
    return int(round(float(value)))


def _safe_rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _iter_period_dates(start_date: date, end_date: date) -> list[date]:
    """시작일과 종료일을 모두 포함하는 날짜 목록을 만든다."""
    day_count = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(max(day_count, 0))]


def _allocated_monthly_amount_for_range(
    monthly_amount: float | int | None,
    *,
    start_date: date,
    end_date: date,
) -> float | None:
    """월 단위 금액을 기간 내 각 날짜의 월 일수 기준으로 배분해 합산한다."""
    if monthly_amount is None:
        return None
    total = 0.0
    for current_date in _iter_period_dates(start_date, end_date):
        month_day_count = calendar.monthrange(current_date.year, current_date.month)[1]
        total += _safe_rate(float(monthly_amount), float(month_day_count))
    return total


def _resolve_weekly_budget(
    *,
    weekly_budget: float | int | None,
    monthly_budget: float | int | None,
    week_start: date,
    week_end: date,
) -> float | None:
    """명시 주간 예산이 없으면 월 목표 소비 한도를 주간 기간에 맞게 환산한다."""
    if weekly_budget is not None:
        return float(weekly_budget)
    return _allocated_monthly_amount_for_range(
        monthly_budget,
        start_date=week_start,
        end_date=week_end,
    )


def _build_weekly_financial_metrics(
    df_member: pd.DataFrame,
    *,
    this_total: float,
    week_start: date,
    week_end: date,
    weekly_budget: float | int | None,
    monthly_budget: float | int | None,
    monthly_income: float | int | None,
) -> JsonObject:
    """사용자 목표 소비 한도와 월소득을 기준으로 주간 예산·소득 지표를 계산한다."""
    resolved_weekly_budget = _resolve_weekly_budget(
        weekly_budget=weekly_budget,
        monthly_budget=monthly_budget,
        week_start=week_start,
        week_end=week_end,
    )
    weekly_income = _allocated_monthly_amount_for_range(
        monthly_income,
        start_date=week_start,
        end_date=week_end,
    )
    month_start = date(week_end.year, week_end.month, 1)
    month_to_date_total = float(
        df_member[(df_member["date"] >= month_start) & (df_member["date"] <= week_end)][
            "사용 금액"
        ].sum()
    )
    week_day_count = max((week_end - week_start).days + 1, 1)
    month_day_count = calendar.monthrange(week_end.year, week_end.month)[1]
    projected_monthly_spending = _safe_rate(this_total, float(week_day_count)) * month_day_count

    return {
        "weekly_budget_usage_rate_percent": None
        if resolved_weekly_budget is None
        else _round_float(_safe_rate(this_total, resolved_weekly_budget) * 100),
        "weekly_remaining_budget": None
        if resolved_weekly_budget is None
        else _to_amount(max(resolved_weekly_budget - this_total, 0.0)),
        "weekly_overspend_amount": None
        if resolved_weekly_budget is None
        else _to_amount(max(this_total - resolved_weekly_budget, 0.0)),
        "weekly_income_usage_rate_percent": None
        if weekly_income is None
        else _round_float(_safe_rate(this_total, weekly_income) * 100),
        "weekly_budget_burn_rate": None
        if resolved_weekly_budget is None
        else _round_float(_safe_rate(this_total, resolved_weekly_budget)),
        "month_to_date_budget_usage_rate_percent": None
        if monthly_budget is None
        else _round_float(_safe_rate(month_to_date_total, float(monthly_budget)) * 100),
        "projected_monthly_spending_from_weekly_pace": _to_amount(projected_monthly_spending),
    }


def _is_match(merchant: str, keywords: list[str]) -> bool:
    return any(kw in str(merchant) for kw in keywords)


def _merchant_summary(df: pd.DataFrame, keywords: list[str]) -> dict[str, Any]:
    """키워드 기반으로 가맹점을 필터링하고 건수·합계·건당 평균을 반환한다."""
    if df.empty:
        return {"count": 0, "total_amount": 0, "avg_per_transaction": 0.0}
    mask = df["결제 내역"].apply(lambda x: _is_match(x, keywords))
    sub = df[mask]
    return {
        "count": int(len(sub)),
        "total_amount": int(sub["사용 금액"].sum()),
        "avg_per_transaction": _round_float(float(sub["사용 금액"].mean()))
        if len(sub) > 0
        else 0.0,
    }


def _find_consecutive(df: pd.DataFrame, min_streak: int = 2) -> list[dict[str, Any]]:
    """N일 연속 소비된 가맹점 탐지."""
    if df.empty:
        return []
    by_date = df.groupby("결제 내역")["date"].apply(lambda x: sorted(x.unique()))
    result: list[dict[str, Any]] = []
    for merchant, dates in by_date.items():
        if len(dates) < min_streak:
            continue
        max_s = cur_s = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days == 1:
                cur_s += 1
                max_s = max(max_s, cur_s)
            else:
                cur_s = 1
        if max_s >= min_streak:
            result.append({"merchant": str(merchant), "max_consecutive_days": max_s})
    return sorted(result, key=lambda x: x["max_consecutive_days"], reverse=True)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _shift_month(year: int, month: int, month_delta: int) -> tuple[int, int]:
    """연도와 월에 월 단위 증감을 적용한 새 연도·월을 반환한다."""
    month_index = year * 12 + month - 1 + month_delta
    shifted_year = month_index // 12
    shifted_month = month_index % 12 + 1
    return shifted_year, shifted_month


def _build_week_period_comparison(
    frame: pd.DataFrame,
    *,
    label: str,
    reference_start: date,
    reference_end: date,
    current_start: date,
    current_end: date,
    current_total: float,
    current_count: int,
    week_num: int | None = None,
) -> JsonObject:
    """분석 주간 소비를 하나의 기준 주간 소비와 비교하는 JSON 블록을 만든다."""
    reference_frame = frame[(frame["date"] >= reference_start) & (frame["date"] <= reference_end)]
    reference_total = float(reference_frame["사용 금액"].sum())
    reference_count = int(len(reference_frame))
    amount_diff = current_total - reference_total
    comparison: JsonObject = {
        "label": label,
        "reference_start_date": str(reference_start),
        "reference_end_date": str(reference_end),
        "current_start_date": str(current_start),
        "current_end_date": str(current_end),
        "reference_total": _to_amount(reference_total),
        "current_total": _to_amount(current_total),
        "amount_diff": _to_amount(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, reference_total) * 100),
        "reference_count": reference_count,
        "current_count": current_count,
        "count_diff": current_count - reference_count,
    }
    if week_num is not None:
        comparison["week_num"] = week_num
    return comparison


def _get_available_week_periods(
    frame: pd.DataFrame,
    candidate_periods: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    """거래 데이터 범위 안에서 온전히 비교할 수 있는 주간 후보 기간만 반환한다."""
    if frame.empty:
        return []

    first_available_date = frame["date"].min()
    return [
        (period_start, period_end)
        for period_start, period_end in candidate_periods
        if period_start >= first_available_date
    ]


def _build_recent_week_average_comparison(
    frame: pd.DataFrame,
    *,
    week_start: date,
    week_end: date,
    current_total: float,
    current_count: int,
    week_count: int = 4,
) -> JsonObject:
    """최근 N개 주간의 평균 소비와 분석 주간 소비를 비교하는 JSON 블록을 만든다."""
    candidate_periods = [
        (week_start - timedelta(days=7 * index), week_end - timedelta(days=7 * index))
        for index in range(1, week_count + 1)
    ]
    reference_periods = _get_available_week_periods(frame, candidate_periods)

    period_rows: list[JsonValue] = []
    total_sum = 0.0
    count_sum = 0
    for period_start, period_end in reference_periods:
        period_frame = frame[(frame["date"] >= period_start) & (frame["date"] <= period_end)]
        period_total = float(period_frame["사용 금액"].sum())
        period_count = int(len(period_frame))
        total_sum += period_total
        count_sum += period_count
        period_rows.append(
            {
                "start_date": str(period_start),
                "end_date": str(period_end),
                "total": _to_amount(period_total),
                "transaction_count": period_count,
            }
        )

    reference_week_count = len(reference_periods)
    average_total = _safe_rate(total_sum, float(reference_week_count))
    average_count = _safe_rate(float(count_sum), float(reference_week_count))
    amount_diff = current_total - average_total
    count_diff = float(current_count) - average_count

    return {
        "label": f"최근 {week_count}주 평균 대비",
        "reference_periods": period_rows,
        "reference_week_count": reference_week_count,
        "average_total": _round_float(average_total),
        "current_total": _to_amount(current_total),
        "amount_diff": _round_float(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, average_total) * 100),
        "average_count": _round_float(average_count),
        "current_count": current_count,
        "count_diff": _round_float(count_diff),
    }


def _build_last_month_same_week_comparison(
    frame: pd.DataFrame,
    *,
    week_start: date,
    week_end: date,
    current_total: float,
    current_count: int,
) -> JsonObject:
    """분석 주차와 같은 지난달 주차의 소비를 비교하는 JSON 블록을 만든다."""
    week_num = (week_start.day - 1) // 7 + 1
    previous_year, previous_month = _shift_month(week_start.year, week_start.month, -1)
    last_day = calendar.monthrange(previous_year, previous_month)[1]
    reference_start_day = (week_num - 1) * 7 + 1
    if reference_start_day > last_day:
        reference_start_day = max(1, last_day - 6)
    reference_start = date(previous_year, previous_month, reference_start_day)
    reference_end = date(previous_year, previous_month, min(reference_start_day + 6, last_day))

    return _build_week_period_comparison(
        frame,
        label="지난달 같은 주차 대비",
        reference_start=reference_start,
        reference_end=reference_end,
        current_start=week_start,
        current_end=week_end,
        current_total=current_total,
        current_count=current_count,
        week_num=week_num,
    )


def _build_weekly_comparisons(
    frame: pd.DataFrame,
    *,
    week_start: date,
    week_end: date,
    current_total: float,
    current_count: int,
) -> JsonObject:
    """주간 분석에 필요한 전주·최근 4주 평균·지난달 같은 주차 비교를 묶는다."""
    previous_week_start = week_start - timedelta(days=7)
    previous_week_end = week_end - timedelta(days=7)

    return {
        "previous_week": _build_week_period_comparison(
            frame,
            label="전주 대비",
            reference_start=previous_week_start,
            reference_end=previous_week_end,
            current_start=week_start,
            current_end=week_end,
            current_total=current_total,
            current_count=current_count,
        ),
        "recent_4week_average": _build_recent_week_average_comparison(
            frame,
            week_start=week_start,
            week_end=week_end,
            current_total=current_total,
            current_count=current_count,
            week_count=4,
        ),
        "same_week_last_month": _build_last_month_same_week_comparison(
            frame,
            week_start=week_start,
            week_end=week_end,
            current_total=current_total,
            current_count=current_count,
        ),
    }


# ---------------------------------------------------------------------------
# 분석 섹션별 빌더
# ---------------------------------------------------------------------------


def _build_weekly_summary(
    df_this: pd.DataFrame,
    df_prev: pd.DataFrame,
    this_total: float,
    prev_total: float,
    week_start: date,
    week_end: date,
) -> JsonObject:
    """[1] 주간 총 지출 요약"""
    amount_diff = this_total - prev_total
    diff_rate = _safe_rate(amount_diff, prev_total) * 100
    active_days = int(df_this["date"].nunique())
    daily_avg = _safe_rate(this_total, active_days)

    daily_sums = df_this.groupby("date")["사용 금액"].sum()
    max_day_date = str(daily_sums.idxmax()) if not daily_sums.empty else None
    max_day_amount = _to_amount(daily_sums.max()) if not daily_sums.empty else 0
    min_day_date = str(daily_sums.idxmin()) if not daily_sums.empty else None
    min_day_amount = _to_amount(daily_sums.min()) if not daily_sums.empty else 0

    return {
        "week_start": str(week_start),
        "week_end": str(week_end),
        "this_week_total": _to_amount(this_total),
        "prev_week_total": _to_amount(prev_total),
        "amount_diff": _to_amount(amount_diff),
        "diff_rate_percent": _round_float(diff_rate),
        "daily_average": _round_float(daily_avg),
        "max_day_date": max_day_date,
        "max_day_amount": max_day_amount,
        "min_day_date": min_day_date,
        "min_day_amount": min_day_amount,
        "transaction_count": int(len(df_this)),
    }


def _build_category_summary(
    df_this: pd.DataFrame,
    df_prev: pd.DataFrame,
    this_total: float,
) -> list[JsonValue]:
    """[2] 카테고리별 주간 분석"""
    this_cat = df_this.groupby("업종 카테고리")["사용 금액"].sum()
    prev_cat = df_prev.groupby("업종 카테고리")["사용 금액"].sum()
    this_cat_cnt = df_this.groupby("업종 카테고리").size()
    prev_total = float(df_prev["사용 금액"].sum())

    all_cats = sorted(set(this_cat.index) | set(prev_cat.index))
    rows: list[JsonValue] = []
    for cat in all_cats:
        ta = float(this_cat.get(cat, 0))
        pa = float(prev_cat.get(cat, 0))
        diff = ta - pa
        ratio = _safe_rate(ta, this_total) * 100
        diff_rate = _safe_rate(diff, pa) * 100 if pa != 0 else 0.0
        transaction_count = int(this_cat_cnt.get(cat, 0))
        row: JsonObject = {
            "category": str(cat),
            "total_amount": _to_amount(ta),
            "ratio_percent": _round_float(ratio),
            "transaction_count": transaction_count,
            "prev_week_amount": _to_amount(pa),
            "diff_amount": _to_amount(diff),
            "diff_rate_percent": _round_float(diff_rate),
        }
        warning = build_ratio_context_warning(
            category=str(cat),
            period_label="이번 주",
            reference_label="전주",
            current_ratio_percent=ratio,
            current_amount=ta,
            reference_amount=pa,
            current_total=this_total,
            reference_total=prev_total,
            current_count=transaction_count,
            low_count_threshold=3,
        )
        if warning is not None:
            row["ratio_context_warning"] = warning
        rows.append(row)
    rows.sort(key=lambda r: r["total_amount"], reverse=True)  # type: ignore[arg-type]
    return rows


def _build_repeat_patterns(df_this: pd.DataFrame) -> JsonObject:
    """[3] 반복 소비 패턴 탐지"""
    top_merch = (
        df_this.groupby("결제 내역")
        .agg(visit_count=("사용 금액", "count"), total_amount=("사용 금액", "sum"))
        .reset_index()
        .sort_values("visit_count", ascending=False)
        .head(10)
    )
    top_merchants: list[JsonValue] = [
        {
            "merchant": str(r["결제 내역"]),
            "visit_count": int(r["visit_count"]),
            "total_amount": _to_amount(r["total_amount"]),
            "main_category": str(
                df_this[df_this["결제 내역"] == r["결제 내역"]]["업종 카테고리"].mode().iloc[0]
            )
            if not df_this[df_this["결제 내역"] == r["결제 내역"]].empty
            else None,
        }
        for _, r in top_merch.iterrows()
    ]

    consecutive = _find_consecutive(df_this)
    delivery = _merchant_summary(df_this, _DELIVERY_KEYWORDS)
    cafe = _merchant_summary(df_this, _CAFE_KEYWORDS)
    conv = _merchant_summary(df_this, _CONVENIENCE_KEYWORDS)
    taxi = _merchant_summary(df_this, _TAXI_KEYWORDS)

    return {
        "top_merchants": top_merchants,
        "consecutive_merchants": [
            {"merchant": c["merchant"], "max_consecutive_days": c["max_consecutive_days"]}
            for c in consecutive
        ],
        "delivery": delivery,
        "cafe": cafe,
        "convenience": conv,
        "taxi": taxi,
    }


def _build_weekday_pattern(df_this: pd.DataFrame, this_total: float) -> JsonObject:
    """[4] 요일별 소비 패턴"""
    wd_amt = df_this.groupby("weekday")["사용 금액"].sum()
    wd_cnt = df_this.groupby("weekday")["사용 금액"].count()

    weekday_rows: list[JsonValue] = []
    for i, name in enumerate(_WEEKDAY_NAMES):
        weekday_rows.append(
            {
                "weekday": name,
                "weekday_num": i,
                "total_amount": _to_amount(wd_amt.get(i, 0)),
                "transaction_count": int(wd_cnt.get(i, 0)),
            }
        )

    peak_wd_idx = int(wd_amt.idxmax()) if not wd_amt.empty else None
    peak_wd_name = _WEEKDAY_NAMES[peak_wd_idx] if peak_wd_idx is not None else None

    weekday_total = sum(float(wd_amt.get(i, 0)) for i in range(5))
    weekday_active = sum(1 for i in range(5) if wd_amt.get(i, 0) > 0)
    weekend_total = sum(float(wd_amt.get(i, 0)) for i in range(5, 7))
    weekend_active = sum(1 for i in range(5, 7) if wd_amt.get(i, 0) > 0)

    weekday_avg = _safe_rate(weekday_total, weekday_active)
    weekend_avg = _safe_rate(weekend_total, weekend_active)

    return {
        "weekday_breakdown": weekday_rows,
        "peak_weekday": peak_wd_name,
        "weekday_average": _round_float(weekday_avg),
        "weekend_average": _round_float(weekend_avg),
        "weekday_vs_weekend_diff": _round_float(weekend_avg - weekday_avg),
    }


def _build_weekly_metrics(
    df_member: pd.DataFrame,
    df_this: pd.DataFrame,
    cat_rows: list[JsonValue],
    repeat_patterns: JsonObject,
    this_total: float,
    prev_total: float,
    week_start: date,
    week_end: date,
    weekly_budget: float | int | None,
    monthly_budget: float | int | None,
    monthly_income: float | int | None,
) -> JsonObject:
    """문서의 주간 소비 분석 10개 핵심 지표와 특수 지표를 계산한다."""
    all_days = pd.date_range(week_start, week_end, freq="D").date
    daily_amounts = df_this.groupby("date")["사용 금액"].sum().reindex(all_days, fill_value=0.0)
    weekday_total = float(df_this[df_this["weekday"] <= 4]["사용 금액"].sum())
    weekend_total = float(df_this[df_this["weekday"] >= 5]["사용 금액"].sum())
    previous_week_change_rate = _safe_rate(this_total - prev_total, prev_total) * 100
    financial_metrics = _build_weekly_financial_metrics(
        df_member,
        this_total=this_total,
        week_start=week_start,
        week_end=week_end,
        weekly_budget=weekly_budget,
        monthly_budget=monthly_budget,
        monthly_income=monthly_income,
    )
    weekday_daily_average = _safe_rate(weekday_total, 5.0)
    weekend_daily_average = _safe_rate(weekend_total, 2.0)
    weekend_overspending_index = _safe_rate(weekend_daily_average, weekday_daily_average)
    max_day_ratio = _safe_rate(float(daily_amounts.max()), this_total) * 100

    return {
        "weekly_total_amount": _to_amount(this_total),
        "weekly_average_daily_amount": _round_float(_safe_rate(this_total, 7.0)),
        "weekly_transaction_count": int(len(df_this)),
        "weekday_spending_ratio_percent": _round_float(_safe_rate(weekday_total, this_total) * 100),
        "weekend_spending_ratio_percent": _round_float(_safe_rate(weekend_total, this_total) * 100),
        "weekday_spending_pattern": [
            {
                "weekday": _WEEKDAY_NAMES[int(day) if isinstance(day, int) else int(day)],
                "weekday_num": int(day),
                "total_amount": _to_amount(float(amount)),
            }
            for day, amount in df_this.groupby("weekday")["사용 금액"].sum().items()
        ],
        "category_spending": cat_rows,
        "previous_week_change_rate_percent": _round_float(previous_week_change_rate),
        "weekly_spending_volatility": _round_float(float(daily_amounts.std(ddof=0))),
        "weekly_budget_usage_rate_percent": financial_metrics["weekly_budget_usage_rate_percent"],
        "weekly_remaining_budget": financial_metrics["weekly_remaining_budget"],
        "weekly_overspend_amount": financial_metrics["weekly_overspend_amount"],
        "weekly_income_usage_rate_percent": financial_metrics["weekly_income_usage_rate_percent"],
        "weekly_budget_burn_rate": financial_metrics["weekly_budget_burn_rate"],
        "month_to_date_budget_usage_rate_percent": financial_metrics[
            "month_to_date_budget_usage_rate_percent"
        ],
        "projected_monthly_spending_from_weekly_pace": financial_metrics[
            "projected_monthly_spending_from_weekly_pace"
        ],
        "special_metrics": {
            "weekend_overspending_index": _round_float(weekend_overspending_index),
            "weekday_concentration_ratio_percent": _round_float(max_day_ratio),
            "routine_indicators": {
                "top_merchants": repeat_patterns.get("top_merchants", []),
                "consecutive_merchants": repeat_patterns.get("consecutive_merchants", []),
            },
        },
        "weekend_overspending_index": _round_float(weekend_overspending_index),
    }


def _build_waste_detection(
    df_this: pd.DataFrame,
    this_total: float,
    upper_bound: float,
) -> JsonObject:
    """[5] 낭비성 소비 탐지"""
    df_late = df_this[df_this["hour"] >= _LATE_NIGHT_HOUR].copy()
    df_micro = df_this[df_this["사용 금액"] < _MICRO_THRESHOLD].copy()
    df_high = df_this[df_this["사용 금액"] > upper_bound].copy()

    late_top = (
        df_late.groupby("업종 카테고리")["사용 금액"].sum().sort_values(ascending=False).head(3)
    )
    micro_top = (
        df_micro.groupby("업종 카테고리")["사용 금액"].sum().sort_values(ascending=False).head(3)
    )

    late_amount = float(df_late["사용 금액"].sum())
    micro_amount = float(df_micro["사용 금액"].sum())
    high_amount = float(df_high["사용 금액"].sum())

    return {
        "late_night": {
            "total_amount": _to_amount(late_amount),
            "count": int(len(df_late)),
            "ratio_percent": _round_float(_safe_rate(late_amount, this_total) * 100),
            "top_categories": [
                {"category": str(c), "amount": _to_amount(a)} for c, a in late_top.items()
            ],
        },
        "micro_spending": {
            "threshold": _MICRO_THRESHOLD,
            "total_amount": _to_amount(micro_amount),
            "count": int(len(df_micro)),
            "top_categories": [
                {"category": str(c), "amount": _to_amount(a)} for c, a in micro_top.items()
            ],
        },
        "high_spending": {
            "iqr_upper_bound": _round_float(upper_bound),
            "total_amount": _to_amount(high_amount),
            "count": int(len(df_high)),
            "items": [
                {
                    "used_at": str(r["사용 시간"]),
                    "merchant": str(r["결제 내역"]),
                    "amount": _to_amount(r["사용 금액"]),
                    "category": str(r["업종 카테고리"]),
                }
                for _, r in df_high.sort_values("사용 금액", ascending=False).iterrows()
            ],
        },
    }


def _build_saving_potential(
    cat_rows: list[JsonValue],
    delivery: dict[str, Any],
    cafe: dict[str, Any],
) -> JsonObject:
    """[6] 절약 가능성 추정"""
    delivery_1less = _to_amount(delivery["avg_per_transaction"]) if delivery["count"] > 0 else 0
    cafe_halved = (
        _to_amount(cafe["avg_per_transaction"] * (cafe["count"] // 2)) if cafe["count"] > 1 else 0
    )

    improved: list[JsonValue] = sorted(
        [r for r in cat_rows if r["diff_amount"] < 0],  # type: ignore[index]
        key=lambda x: x["diff_amount"],  # type: ignore[index]
    )[:3]
    worsened: list[JsonValue] = sorted(
        [r for r in cat_rows if r["diff_amount"] > 0],  # type: ignore[index]
        key=lambda x: x["diff_amount"],  # type: ignore[index]
        reverse=True,
    )[:3]

    return {
        "delivery_save_per_skip": delivery_1less,
        "cafe_save_half_visits": cafe_halved,
        "improved_categories": [
            {"category": r["category"], "diff_amount": r["diff_amount"]}  # type: ignore[index]
            for r in improved
        ],
        "worsened_categories": [
            {"category": r["category"], "diff_amount": r["diff_amount"]}  # type: ignore[index]
            for r in worsened
        ],
    }


def _build_elasticity_analysis(df: pd.DataFrame) -> JsonObject:
    """[8] 소비 탄성 및 심리적 반동 분석"""
    if df.empty:
        return {
            "correlation": None,
            "threshold": None,
            "rebound_avg": None,
            "normal_avg": None,
            "cheat_effective": None,
            "recommended_cheat_amount": None,
        }

    df_local = df.copy()
    df_local["week_id"] = df_local["사용 시간"].dt.isocalendar().week
    df_local["year"] = df_local["사용 시간"].dt.year

    weekly_stats: list[dict[str, Any]] = []
    for (_yr, _wk), group in df_local.groupby(["year", "week_id"]):
        weekday_data = group[group["weekday"] <= 4]
        weekend_data = group[group["weekday"] >= 5]

        wd_daily_avg = (
            weekday_data.groupby("date")["사용 금액"].sum().mean()
            if not weekday_data.empty
            else 0.0
        )
        we_total = weekend_data["사용 금액"].sum()
        friday_spending = group[group["weekday"] == 4]["사용 금액"].sum()

        weekly_stats.append(
            {
                "weekday_avg": float(wd_daily_avg),
                "weekend_total": float(we_total),
                "friday_spending": float(friday_spending),
            }
        )

    df_elastic = pd.DataFrame(weekly_stats)

    if len(df_elastic) <= 1:
        return {
            "correlation": None,
            "threshold": None,
            "rebound_avg": None,
            "normal_avg": None,
            "cheat_effective": None,
            "recommended_cheat_amount": None,
        }

    corr = df_elastic["weekday_avg"].corr(df_elastic["weekend_total"])
    if pd.isna(corr):
        corr = 0.0

    threshold = float(df_elastic["weekday_avg"].quantile(0.3))
    if pd.isna(threshold):
        threshold = 0.0

    rebound_weeks = df_elastic[df_elastic["weekday_avg"] <= threshold]
    normal_weeks = df_elastic[df_elastic["weekday_avg"] > threshold]

    rebound_avg = float(rebound_weeks["weekend_total"].mean()) if not rebound_weeks.empty else 0.0
    normal_avg = float(normal_weeks["weekend_total"].mean()) if not normal_weeks.empty else 0.0

    if pd.isna(rebound_avg):
        rebound_avg = 0.0
    if pd.isna(normal_avg):
        normal_avg = 0.0

    cheat_effective = df_elastic[
        (df_elastic["friday_spending"] > df_elastic["friday_spending"].median())
        & (df_elastic["weekend_total"] < df_elastic["weekend_total"].median())
    ]

    is_cheat_effective = not cheat_effective.empty
    recommended_cheat_amount = (
        float(df_elastic["friday_spending"].median()) if is_cheat_effective else None
    )

    return {
        "correlation": _round_float(corr),
        "threshold": _to_amount(threshold),
        "rebound_avg": _to_amount(rebound_avg),
        "normal_avg": _to_amount(normal_avg),
        "cheat_effective": is_cheat_effective,
        "recommended_cheat_amount": _to_amount(recommended_cheat_amount)
        if recommended_cheat_amount is not None
        else None,
    }


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def build_weekly_consumption_analysis_from_frames(
    all_frame: pd.DataFrame,
    *,
    member_id: int = 1,
    week_start: str | date = "2024-04-01",
    week_end: str | date = "2024-04-07",
    source_path: str | Path | None = None,
    weekly_budget: float | int | None = None,
    monthly_budget: float | int | None = None,
    monthly_income: float | int | None = None,
) -> JsonObject:
    """과거+당주 소비 DataFrame에서 주간 소비 분석 JSON을 만든다.

    Parameters
    ----------
    all_frame:
        멤버 전체 거래 내역 (과거 + 당주 포함).
        ``멤버 id``, ``사용 금액``, ``사용 시간``, ``결제 내역``, ``업종 카테고리`` 컬럼 필수.
    member_id:
        분석 대상 멤버 ID.
    week_start:
        분석할 주의 시작일 (YYYY-MM-DD 또는 date 객체).
    week_end:
        분석할 주의 종료일 (YYYY-MM-DD 또는 date 객체).
    source_path:
        데이터 원천 경로 (메타 정보용, 선택).

    Returns
    -------
    JsonObject
        주간 소비 분석 결과 딕셔너리.
    """
    _validate_columns(all_frame, "전체")

    ws = _parse_date(week_start)
    we = _parse_date(week_end)
    prev_ws = ws - timedelta(days=7)
    prev_we = we - timedelta(days=7)

    # 멤버 필터 & 타입 정리
    df = all_frame[all_frame["멤버 id"] == member_id].copy()
    if df.empty:
        raise ValueError(f"멤버 {member_id}번 거래를 찾을 수 없습니다.")

    df["사용 금액"] = pd.to_numeric(df["사용 금액"])
    df["사용 시간"] = pd.to_datetime(df["사용 시간"])
    df["date"] = df["사용 시간"].dt.date
    df["hour"] = df["사용 시간"].dt.hour
    df["weekday"] = df["사용 시간"].dt.dayofweek  # 0=월, 6=일

    df_this = df[(df["date"] >= ws) & (df["date"] <= we)].copy()
    df_prev = df[(df["date"] >= prev_ws) & (df["date"] <= prev_we)].copy()
    df_past = df[df["date"] < ws].copy()

    # IQR 상한선 (당주 제외 과거 전체 기준)
    if df_past.empty:
        df_base = df
    else:
        df_base = df_past
    q1 = float(df_base["사용 금액"].quantile(0.25))
    q3 = float(df_base["사용 금액"].quantile(0.75))
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    this_total = float(df_this["사용 금액"].sum())
    prev_total = float(df_prev["사용 금액"].sum())

    # 섹션별 빌드
    weekly_summary = _build_weekly_summary(df_this, df_prev, this_total, prev_total, ws, we)
    category_summary = _build_category_summary(df_this, df_prev, this_total)
    repeat_patterns = _build_repeat_patterns(df_this)
    weekday_pattern = _build_weekday_pattern(df_this, this_total)
    weekly_metrics = _build_weekly_metrics(
        df,
        df_this,
        category_summary,
        repeat_patterns,
        this_total,
        prev_total,
        ws,
        we,
        weekly_budget,
        monthly_budget,
        monthly_income,
    )
    waste_detection = _build_waste_detection(df_this, this_total, upper_bound)
    saving_potential = _build_saving_potential(
        category_summary,
        repeat_patterns["delivery"],  # type: ignore[arg-type]
        repeat_patterns["cafe"],  # type: ignore[arg-type]
    )
    weekly_comparisons = _build_weekly_comparisons(
        df,
        week_start=ws,
        week_end=we,
        current_total=this_total,
        current_count=int(len(df_this)),
    )
    elasticity_analysis = _build_elasticity_analysis(df)

    return {
        "member_id": member_id,
        "week_start": str(ws),
        "week_end": str(we),
        "source_path": str(source_path) if source_path is not None else None,
        "outlier_thresholds": {
            "q1": _round_float(q1),
            "q3": _round_float(q3),
            "iqr": _round_float(iqr),
            "upper_bound": _round_float(upper_bound),
        },
        "weekly_summary": weekly_summary,
        "category_summary": category_summary,
        "weekly_comparisons": weekly_comparisons,
        "weekly_metrics": weekly_metrics,
        "repeat_patterns": repeat_patterns,
        "weekday_pattern": weekday_pattern,
        "waste_detection": waste_detection,
        "saving_potential": saving_potential,
        "elasticity_analysis": elasticity_analysis,
    }
