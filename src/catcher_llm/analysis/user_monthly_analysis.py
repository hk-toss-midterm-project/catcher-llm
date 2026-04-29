from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

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

_ESSENTIAL_CATEGORIES = ["교통", "의료", "생활"]
_FRICTIONLESS_KEYWORDS = ["온라인", "간편결제", "앱결제", "배달"]

_FIXED_PAYMENT_KEYWORD = "자동이체"
_SUBSCRIPTION_KEYWORDS = ["구독", "OTT", "넷플릭스", "유튜브", "멤버십", "정기결제"]
_WASTE_CATEGORIES = ["식비", "쇼핑"]
_ESSENTIAL_CATEGORIES = ["교통", "의료", "생활"]

_MICRO_THRESHOLD: int = 10_000
_LATE_NIGHT_HOUR: int = 21
_SUBWAY_AVG_FARE: int = 1_400

_REQUIRED_COLUMNS = {
    "멤버 id",
    "사용 금액",
    "사용 시간",
    "결제 내역",
    "업종 카테고리",
    "결제 방식 (온/오프라인)",
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


def _is_match(merchant: str, keywords: list[str]) -> bool:
    return any(kw in str(merchant) for kw in keywords)


def _merchant_summary(df: pd.DataFrame, keywords: list[str]) -> dict[str, Any]:
    """키워드 기반으로 가맹점을 필터링하고 건수·합계·건당 평균을 반환한다."""
    if df.empty:
        return {"count": 0, "total_amount": 0, "avg_per_transaction": 0.0}
    sub = df[df["결제 내역"].apply(lambda x: _is_match(x, keywords))]
    return {
        "count": int(len(sub)),
        "total_amount": int(sub["사용 금액"].sum()),
        "avg_per_transaction": _round_float(float(sub["사용 금액"].mean()))
        if len(sub) > 0
        else 0.0,
    }


def _parse_ym(analysis_month: str) -> tuple[int, int]:
    """'YYYY-MM' 형식의 문자열에서 연도·월을 추출한다."""
    year, month = int(analysis_month[:4]), int(analysis_month[5:7])
    return year, month


def _shift_ym(year: int, month: int, month_delta: int) -> tuple[int, int, str]:
    """연도와 월에 월 단위 증감을 적용한 뒤 연도·월·문자열을 반환한다."""
    month_index = year * 12 + month - 1 + month_delta
    shifted_year = month_index // 12
    shifted_month = month_index % 12 + 1
    return shifted_year, shifted_month, f"{shifted_year}-{shifted_month:02d}"


def _prev_ym(year: int, month: int) -> tuple[int, int, str]:
    """전월 연도·월·'YYYY-MM' 문자열을 반환한다."""
    return _shift_ym(year, month, -1)


def _build_month_comparison(
    frame: pd.DataFrame,
    *,
    label: str,
    reference_month: str,
    current_month: str,
    current_total: float,
    current_count: int,
) -> JsonObject:
    """분석월 소비를 하나의 기준월 소비와 비교하는 JSON 블록을 만든다."""
    reference_frame = frame[frame["ym"] == reference_month]
    reference_total = float(reference_frame["사용 금액"].sum())
    reference_count = int(len(reference_frame))
    amount_diff = current_total - reference_total

    return {
        "label": label,
        "reference_month": reference_month,
        "current_month": current_month,
        "reference_total": _to_amount(reference_total),
        "current_total": _to_amount(current_total),
        "amount_diff": _to_amount(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, reference_total) * 100),
        "reference_count": reference_count,
        "current_count": current_count,
        "count_diff": current_count - reference_count,
    }


def _get_available_reference_months(
    frame: pd.DataFrame,
    candidate_months: list[str],
) -> list[str]:
    """거래 데이터 범위 안에 있는 월간 비교 후보월만 남긴다."""
    if frame.empty:
        return []

    first_available_month = str(frame["ym"].min())
    return [
        reference_month
        for reference_month in candidate_months
        if reference_month >= first_available_month
    ]


def _build_recent_month_average_comparison(
    frame: pd.DataFrame,
    *,
    year: int,
    month: int,
    current_month: str,
    current_total: float,
    current_count: int,
    month_count: int = 3,
) -> JsonObject:
    """최근 N개월 평균 소비와 분석월 소비를 비교하는 JSON 블록을 만든다."""
    candidate_months = [_shift_ym(year, month, -index)[2] for index in range(1, month_count + 1)]
    reference_months = _get_available_reference_months(frame, candidate_months)
    month_totals = frame.groupby("ym")["사용 금액"].sum()
    month_counts = frame.groupby("ym").size()

    month_rows: list[JsonValue] = []
    total_sum = 0.0
    count_sum = 0
    for reference_month in reference_months:
        reference_total = float(month_totals.get(reference_month, 0.0))
        reference_count = int(month_counts.get(reference_month, 0))
        total_sum += reference_total
        count_sum += reference_count
        month_rows.append(
            {
                "month": reference_month,
                "total": _to_amount(reference_total),
                "transaction_count": reference_count,
            }
        )

    reference_month_count = len(reference_months)
    average_total = _safe_rate(total_sum, float(reference_month_count))
    average_count = _safe_rate(float(count_sum), float(reference_month_count))
    amount_diff = current_total - average_total
    count_diff = float(current_count) - average_count

    return {
        "label": f"최근 {month_count}개월 평균 대비",
        "reference_months": reference_months,
        "reference_month_details": month_rows,
        "reference_month_count": reference_month_count,
        "average_total": _round_float(average_total),
        "current_month": current_month,
        "current_total": _to_amount(current_total),
        "amount_diff": _round_float(amount_diff),
        "amount_diff_rate_percent": _round_float(_safe_rate(amount_diff, average_total) * 100),
        "average_count": _round_float(average_count),
        "current_count": current_count,
        "count_diff": _round_float(count_diff),
    }


def _build_monthly_comparisons(
    frame: pd.DataFrame,
    *,
    year: int,
    month: int,
    analysis_month: str,
    prev_month: str,
    current_total: float,
    current_count: int,
) -> JsonObject:
    """월간 분석에 필요한 전월·최근 3개월 평균 비교를 묶는다."""
    return {
        "previous_month": _build_month_comparison(
            frame,
            label="전월 대비",
            reference_month=prev_month,
            current_month=analysis_month,
            current_total=current_total,
            current_count=current_count,
        ),
        "recent_3month_average": _build_recent_month_average_comparison(
            frame,
            year=year,
            month=month,
            current_month=analysis_month,
            current_total=current_total,
            current_count=current_count,
            month_count=3,
        ),
    }


# ---------------------------------------------------------------------------
# 분석 섹션별 빌더
# ---------------------------------------------------------------------------


def _build_monthly_summary(
    df_this: pd.DataFrame,
    df_prev: pd.DataFrame,
    this_total: float,
    prev_total: float,
    analysis_month: str,
) -> JsonObject:
    """[1] 월간 총 지출 요약"""
    amt_diff = this_total - prev_total
    amt_diff_r = _safe_rate(amt_diff, prev_total) * 100
    active_days = int(df_this["date"].nunique())
    daily_avg = _safe_rate(this_total, active_days)

    daily_sums = df_this.groupby("date")["사용 금액"].sum()
    max_day = (
        (str(daily_sums.idxmax()), _to_amount(daily_sums.max()))
        if not daily_sums.empty
        else (None, 0)
    )
    min_day = (
        (str(daily_sums.idxmin()), _to_amount(daily_sums.min()))
        if not daily_sums.empty
        else (None, 0)
    )

    return {
        "analysis_month": analysis_month,
        "this_month_total": _to_amount(this_total),
        "prev_month_total": _to_amount(prev_total),
        "amount_diff": _to_amount(amt_diff),
        "diff_rate_percent": _round_float(amt_diff_r),
        "daily_average": _round_float(daily_avg),
        "active_days": active_days,
        "max_day_date": max_day[0],
        "max_day_amount": max_day[1],
        "min_day_date": min_day[0],
        "min_day_amount": min_day[1],
        "transaction_count": int(len(df_this)),
    }


def _build_fixed_variable(
    df_this: pd.DataFrame,
    this_total: float,
) -> JsonObject:
    """[2] 고정비 vs 변동비 분석"""
    is_fixed = df_this["결제 방식 (온/오프라인)"].str.contains(_FIXED_PAYMENT_KEYWORD, na=False)
    df_fixed = df_this[is_fixed]
    df_variable = df_this[~is_fixed]

    fixed_total = float(df_fixed["사용 금액"].sum())
    variable_total = float(df_variable["사용 금액"].sum())
    fixed_ratio = _safe_rate(fixed_total, this_total) * 100

    fixed_items_df = (
        df_fixed.groupby("결제 내역")
        .agg(count=("사용 금액", "count"), total=("사용 금액", "sum"))
        .reset_index()
        .sort_values("total", ascending=False)
    )
    fixed_items: list[JsonValue] = [
        {
            "merchant": str(r["결제 내역"]),
            "count": int(r["count"]),
            "total_amount": _to_amount(r["total"]),
        }
        for _, r in fixed_items_df.iterrows()
    ]

    return {
        "fixed_total": _to_amount(fixed_total),
        "variable_total": _to_amount(variable_total),
        "fixed_ratio_percent": _round_float(fixed_ratio),
        "variable_ratio_percent": _round_float(100 - fixed_ratio),
        "fixed_items": fixed_items,
    }


def _build_category_deep(
    df_this: pd.DataFrame,
    df_prev: pd.DataFrame,
    this_total: float,
) -> tuple[list[JsonValue], list[JsonValue]]:
    """[3] 카테고리 심층 분석 → (cat_deep, top_savable)"""
    this_cat = df_this.groupby("업종 카테고리")["사용 금액"].sum()
    prev_cat = df_prev.groupby("업종 카테고리")["사용 금액"].sum()
    this_cat_cnt = df_this.groupby("업종 카테고리").size()

    all_cats = sorted(set(this_cat.index) | set(prev_cat.index))
    cat_deep: list[JsonValue] = []
    for cat in all_cats:
        ta = float(this_cat.get(cat, 0))
        pa = float(prev_cat.get(cat, 0))
        diff = ta - pa
        ratio = _safe_rate(ta, this_total) * 100
        diff_r = _safe_rate(diff, pa) * 100 if pa != 0 else 0.0
        cat_type = (
            "필수"
            if cat in _ESSENTIAL_CATEGORIES
            else "낭비성"
            if cat in _WASTE_CATEGORIES
            else "기타"
        )
        cat_deep.append(
            {
                "category": str(cat),
                "type": cat_type,
                "total_amount": _to_amount(ta),
                "ratio_percent": _round_float(ratio),
                "transaction_count": int(this_cat_cnt.get(cat, 0)),
                "prev_month_amount": _to_amount(pa),
                "diff_amount": _to_amount(diff),
                "diff_rate_percent": _round_float(diff_r),
            }
        )
    cat_deep.sort(key=lambda r: r["total_amount"], reverse=True)  # type: ignore[arg-type]

    top_savable: list[JsonValue] = sorted(
        [r for r in cat_deep if r["type"] == "낭비성"],  # type: ignore[index]
        key=lambda x: x["total_amount"],  # type: ignore[index]
        reverse=True,
    )[:3]

    return cat_deep, top_savable


def _build_repeat_monthly(df_this: pd.DataFrame) -> JsonObject:
    """[4] 반복 소비 누적 분석"""
    top5_df = (
        df_this.groupby("결제 내역")
        .agg(visit_count=("사용 금액", "count"), total_amount=("사용 금액", "sum"))
        .reset_index()
        .sort_values("visit_count", ascending=False)
        .head(5)
    )
    top5_list: list[JsonValue] = [
        {
            "merchant": str(r["결제 내역"]),
            "visit_count": int(r["visit_count"]),
            "total_amount": _to_amount(r["total_amount"]),
        }
        for _, r in top5_df.iterrows()
    ]

    delivery = _merchant_summary(df_this, _DELIVERY_KEYWORDS)
    cafe = _merchant_summary(df_this, _CAFE_KEYWORDS)
    conv = _merchant_summary(df_this, _CONVENIENCE_KEYWORDS)
    taxi = _merchant_summary(df_this, _TAXI_KEYWORDS)

    return {
        "top5_merchants": top5_list,
        "delivery": delivery,
        "cafe": cafe,
        "convenience": conv,
        "taxi": taxi,
    }


def _build_weekly_trend(
    df_this: pd.DataFrame,
    year: int,
    month: int,
) -> JsonObject:
    """[5] 주차별 소비 추이"""
    df_this = df_this.copy()
    df_this["week_num"] = df_this["date"].apply(lambda d: (d.day - 1) // 7 + 1)

    weekly_raw = (
        df_this.groupby("week_num")
        .agg(total_amount=("사용 금액", "sum"), count=("사용 금액", "count"))
        .reset_index()
    )

    last_day_num = calendar.monthrange(year, month)[1]

    def week_range(wn: int) -> tuple[str, str]:
        start = date(year, month, (wn - 1) * 7 + 1)
        end = date(year, month, min(wn * 7, last_day_num))
        return str(start), str(end)

    rows: list[JsonValue] = []
    for _, r in weekly_raw.iterrows():
        wn = int(r["week_num"])
        s, e = week_range(wn)
        rows.append(
            {
                "week_num": wn,
                "start_date": s,
                "end_date": e,
                "total_amount": _to_amount(r["total_amount"]),
                "count": int(r["count"]),
            }
        )

    amounts = [r["total_amount"] for r in rows]  # type: ignore[index]
    if len(amounts) >= 2:
        slope = amounts[-1] - amounts[0]
        trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    else:
        trend = "stable"

    return {"weekly_breakdown": rows, "trend_direction": trend}


def _build_micro_monthly(df_this: pd.DataFrame, this_total: float) -> JsonObject:
    """[6] 소액 다빈도 누적 분석"""
    df_micro = df_this[df_this["사용 금액"] < _MICRO_THRESHOLD].copy()
    micro_total = float(df_micro["사용 금액"].sum())
    micro_count = int(len(df_micro))

    micro_cat = (
        df_micro.groupby("업종 카테고리")["사용 금액"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .head(5)
    )
    micro_cat_list: list[JsonValue] = [
        {"category": str(cat), "total_amount": _to_amount(row["sum"]), "count": int(row["count"])}
        for cat, row in micro_cat.iterrows()
    ]

    return {
        "threshold": _MICRO_THRESHOLD,
        "total_amount": _to_amount(micro_total),
        "count": micro_count,
        "ratio_percent": _round_float(_safe_rate(micro_total, this_total) * 100),
        "top_categories": micro_cat_list,
    }


def _build_late_night_monthly(df_this: pd.DataFrame, this_total: float) -> JsonObject:
    """[7] 야간 소비 월간 분석"""
    df_late = df_this[df_this["hour"] >= _LATE_NIGHT_HOUR].copy()
    late_total = float(df_late["사용 금액"].sum())
    late_count = int(len(df_late))

    late_cat = (
        df_late.groupby("업종 카테고리")["사용 금액"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .head(5)
    )
    late_cat_list: list[JsonValue] = [
        {"category": str(cat), "total_amount": _to_amount(row["sum"]), "count": int(row["count"])}
        for cat, row in late_cat.iterrows()
    ]

    return {
        "late_night_hour": _LATE_NIGHT_HOUR,
        "total_amount": _to_amount(late_total),
        "count": late_count,
        "ratio_percent": _round_float(_safe_rate(late_total, this_total) * 100),
        "top_categories": late_cat_list,
    }


def _build_high_spending_monthly(df_this: pd.DataFrame, upper_bound: float) -> JsonObject:
    """[8] 이상 지출 월간 누적 집계"""
    df_high = df_this[df_this["사용 금액"] > upper_bound].copy()
    high_total = float(df_high["사용 금액"].sum())
    high_count = int(len(df_high))

    items: list[JsonValue] = [
        {
            "used_at": str(r["사용 시간"]),
            "merchant": str(r["결제 내역"]),
            "amount": _to_amount(r["사용 금액"]),
            "category": str(r["업종 카테고리"]),
        }
        for _, r in df_high.sort_values("사용 금액", ascending=False).iterrows()
    ]

    return {
        "iqr_upper_bound": _round_float(upper_bound),
        "total_amount": _to_amount(high_total),
        "count": high_count,
        "items": items,
    }


def _build_saving_monthly(
    delivery: dict[str, Any],
    cafe: dict[str, Any],
    taxi: dict[str, Any],
    micro_total: float,
    this_total: float,
) -> JsonObject:
    """[9] 절약 가능 금액 추정"""
    delivery_30pct = _to_amount(delivery["total_amount"] * 0.30)
    cafe_half = (
        _to_amount(cafe["avg_per_transaction"] * (cafe["count"] // 2)) if cafe["count"] > 1 else 0
    )
    taxi_to_transit = (
        max(0, _to_amount((taxi["avg_per_transaction"] - _SUBWAY_AVG_FARE) * (taxi["count"] // 2)))
        if taxi["count"] > 0
        else 0
    )
    micro_20pct = _to_amount(micro_total * 0.20)
    total_potential = delivery_30pct + cafe_half + taxi_to_transit + micro_20pct
    next_month_target = _to_amount(this_total * 0.90)

    return {
        "delivery_reduce_30pct": delivery_30pct,
        "cafe_every_other_day": cafe_half,
        "taxi_to_transit": taxi_to_transit,
        "micro_reduce_20pct": micro_20pct,
        "total_potential_saving": total_potential,
        "next_month_recommended_target": next_month_target,
    }


def _build_cash_flow_volatility(weekly_trend: JsonObject) -> JsonObject:
    """[10] 현금 흐름 변동성 지수 (Cash Flow Volatility)"""
    weekly_breakdown = weekly_trend.get("weekly_breakdown", [])
    weekly_amounts = (
        [r["total_amount"] for r in weekly_breakdown] if isinstance(weekly_breakdown, list) else []
    )

    if not weekly_amounts:
        return {
            "mean_weekly": 0,
            "std_weekly": 0,
            "cv_index": 0.0,
            "pace_status": "주차별 데이터가 부족합니다.",
            "weekly_ratios": [],
        }

    s = pd.Series(weekly_amounts)
    mean_weekly = float(s.mean())
    std_weekly = float(s.std(ddof=0)) if len(s) > 0 else 0.0

    cv_index = (std_weekly / mean_weekly) if mean_weekly > 0 else 0.0

    if cv_index > 0.5:
        pace_status = "위험 (초반 과소비 후 후반 쪼들림 등 변동성이 매우 큼)"
    elif cv_index > 0.3:
        pace_status = "주의 (주차별 소비 편차가 꽤 있는 편)"
    else:
        pace_status = "안정 (매주 일정한 페이스로 소비 중)"

    total_amount = sum(weekly_amounts)
    weekly_ratios = [
        {
            "week_num": i + 1,
            "amount": int(amt),  # type: ignore[arg-type]
            "ratio_percent": _round_float((amt / total_amount * 100) if total_amount > 0 else 0.0),  # type: ignore[operator]
        }
        for i, amt in enumerate(weekly_amounts)
    ]

    return {
        "mean_weekly": _to_amount(mean_weekly),
        "std_weekly": _to_amount(std_weekly),
        "cv_index": _round_float(cv_index),
        "pace_status": pace_status,
        "weekly_ratios": weekly_ratios,  # type: ignore[dict-item]
    }


def _build_spending_concentration(df_this: pd.DataFrame) -> JsonObject:
    """[11] 파레토 지출 쏠림 지수 (Spending Concentration Index)"""
    df_variable = df_this[~df_this["업종 카테고리"].isin(_ESSENTIAL_CATEGORIES)]

    if df_variable.empty:
        return {
            "total_variable_amount": 0,
            "top_1_category": None,
            "top_1_amount": 0,
            "top_1_ratio_percent": 0.0,
            "top_2_category": None,
            "top_2_amount": 0,
            "top_2_ratio_percent": 0.0,
            "top_2_combined_ratio_percent": 0.0,
            "concentration_status": "변동비 카테고리 지출 내역이 없습니다.",
        }

    var_total = float(df_variable["사용 금액"].sum())
    var_cat = df_variable.groupby("업종 카테고리")["사용 금액"].sum().sort_values(ascending=False)

    top_1_cat = str(var_cat.index[0])
    top_1_amt = float(var_cat.iloc[0])
    top_1_ratio = (top_1_amt / var_total * 100) if var_total > 0 else 0.0

    top_2_cat = None
    top_2_amt = 0.0
    top_2_ratio = 0.0
    if len(var_cat) > 1:
        top_2_cat = str(var_cat.index[1])
        top_2_amt = float(var_cat.iloc[1])
        top_2_ratio = (top_2_amt / var_total * 100) if var_total > 0 else 0.0

    top_2_combined_ratio = top_1_ratio + top_2_ratio

    if top_1_ratio >= 60:
        concentration_status = "극심한 쏠림 (1위 항목 하나만 통제해도 예산 절감 효과 극대화)"
    elif top_2_combined_ratio >= 80:
        concentration_status = "파레토 쏠림 (상위 2개 항목이 전체 변동비의 80% 차지)"
    else:
        concentration_status = "분산 소비 (비교적 여러 항목에 골고루 지출 중)"

    return {
        "total_variable_amount": _to_amount(var_total),
        "top_1_category": top_1_cat,
        "top_1_amount": _to_amount(top_1_amt),
        "top_1_ratio_percent": _round_float(top_1_ratio),
        "top_2_category": top_2_cat,
        "top_2_amount": _to_amount(top_2_amt),
        "top_2_ratio_percent": _round_float(top_2_ratio),
        "top_2_combined_ratio_percent": _round_float(top_2_combined_ratio),
        "concentration_status": concentration_status,
    }


def _build_frictionless_and_density(df_this: pd.DataFrame, this_total: float) -> JsonObject:
    """[12] 월간 지출 마찰력 및 밀도 분석"""
    if "결제 방식 (온/오프라인)" not in df_this.columns:
        return {
            "frictionless_spending": {"total_amount": 0, "count": 0, "ratio_percent": 0.0},
            "transaction_density": {"avg_daily_count": 0.0, "avg_per_transaction": 0},
        }

    is_fric = df_this["결제 방식 (온/오프라인)"].str.contains(
        "|".join(_FRICTIONLESS_KEYWORDS), na=False
    )
    df_fric = df_this[is_fric]

    fric_total = float(df_fric["사용 금액"].sum())
    fric_count = int(len(df_fric))
    fric_ratio = (fric_total / this_total * 100) if this_total > 0 else 0.0

    daily_counts = df_this.groupby("date").size()
    avg_daily_count = float(daily_counts.mean()) if not daily_counts.empty else 0.0
    avg_per_swipe = (this_total / len(df_this)) if len(df_this) > 0 else 0.0

    return {
        "frictionless_spending": {
            "total_amount": _to_amount(fric_total),
            "count": fric_count,
            "ratio_percent": _round_float(fric_ratio),
        },
        "transaction_density": {
            "avg_daily_count": _round_float(avg_daily_count),
            "avg_per_transaction": _to_amount(avg_per_swipe),
        },
    }


def _build_installment_debt_pressure(df_this: pd.DataFrame, this_total: float) -> JsonObject:
    """[13] 할부 부채 압박 지수 (Installment Debt Pressure Index)"""
    if "할부 여부" not in df_this.columns:
        df_this_copy = df_this.copy()
        df_this_copy["할부 여부"] = "N"
    else:
        df_this_copy = df_this.copy()

    df_install = df_this_copy[df_this_copy["할부 여부"] == "Y"].copy()
    install_total = float(df_install["사용 금액"].sum()) if not df_install.empty else 0.0
    install_count = int(len(df_install))
    install_ratio = (install_total / this_total * 100) if this_total > 0 else 0.0

    avg_months = 0.0
    max_months = 0
    items: list[JsonValue] = []

    if not df_install.empty and "할부 개월" in df_install.columns:
        df_install["할부 개월"] = pd.to_numeric(df_install["할부 개월"], errors="coerce").fillna(0)
        avg_months = float(df_install["할부 개월"].mean())
        max_months = int(df_install["할부 개월"].max())

        items = [
            {
                "used_at": str(r["사용 시간"]),
                "merchant": str(r["결제 내역"]),
                "amount": _to_amount(r["사용 금액"]),
                "installment_months": int(r["할부 개월"]),
            }
            for _, r in df_install.sort_values("사용 금액", ascending=False).iterrows()
        ]

    return {
        "total_installment_amount": _to_amount(install_total),
        "installment_count": install_count,
        "installment_ratio_percent": _round_float(install_ratio),
        "avg_installment_months": _round_float(avg_months),
        "max_installment_months": max_months,
        "items": items,
    }


def _build_monthly_category_spending(cat_deep: list[JsonValue]) -> list[JsonValue]:
    """문서형 월간 지표에 사용할 카테고리별 소비 금액·비중 목록을 만든다."""
    rows: list[JsonValue] = []
    for raw_row in cat_deep:
        if not isinstance(raw_row, dict):
            continue
        rows.append(
            {
                "category": raw_row.get("category"),
                "total_amount": raw_row.get("total_amount", 0),
                "ratio_percent": raw_row.get("ratio_percent", 0.0),
                "transaction_count": raw_row.get("transaction_count", 0),
            }
        )
    return rows


def _build_subscription_total(df_this: pd.DataFrame) -> int:
    """가맹점명과 결제 방식 키워드로 월간 구독료 합계를 추정한다."""
    keyword_pattern = "|".join(_SUBSCRIPTION_KEYWORDS)
    merchant_match = df_this["결제 내역"].astype("string").str.contains(keyword_pattern, na=False)
    payment_match = (
        df_this["결제 방식 (온/오프라인)"].astype("string").str.contains(keyword_pattern, na=False)
    )
    return _to_amount(float(df_this[merchant_match | payment_match]["사용 금액"].sum()))


def _build_post_salary_spending_increase_rate(
    df_this: pd.DataFrame,
    *,
    year: int,
    month: int,
    salary_day: int | None,
) -> float | None:
    """급여일 직후 7일의 일평균 소비가 나머지 기간 대비 얼마나 증가했는지 계산한다."""
    if salary_day is None:
        return None

    last_day = calendar.monthrange(year, month)[1]
    normalized_salary_day = min(max(salary_day, 1), last_day)
    salary_start = date(year, month, normalized_salary_day)
    salary_end = date(year, month, min(normalized_salary_day + 6, last_day))
    post_salary_frame = df_this[(df_this["date"] >= salary_start) & (df_this["date"] <= salary_end)]
    baseline_frame = df_this[(df_this["date"] < salary_start) | (df_this["date"] > salary_end)]
    post_days = (salary_end - salary_start).days + 1
    baseline_days = max(last_day - post_days, 0)
    post_daily_average = _safe_rate(float(post_salary_frame["사용 금액"].sum()), float(post_days))
    baseline_daily_average = _safe_rate(
        float(baseline_frame["사용 금액"].sum()),
        float(baseline_days),
    )
    return _round_float(
        _safe_rate(post_daily_average - baseline_daily_average, baseline_daily_average) * 100
    )


def _build_month_end_pressure_index(
    df_this: pd.DataFrame, *, year: int, month: int
) -> float | None:
    """월말 7일 일평균 소비를 월말 이전 일평균 소비와 비교한 압박 지수를 계산한다."""
    last_day = calendar.monthrange(year, month)[1]
    month_end_start = date(year, month, max(1, last_day - 6))
    month_end_frame = df_this[df_this["date"] >= month_end_start]
    before_month_end_frame = df_this[df_this["date"] < month_end_start]
    end_days = last_day - month_end_start.day + 1
    before_days = month_end_start.day - 1
    end_daily_average = _safe_rate(float(month_end_frame["사용 금액"].sum()), float(end_days))
    before_daily_average = _safe_rate(
        float(before_month_end_frame["사용 금액"].sum()),
        float(before_days),
    )
    if before_daily_average == 0:
        return None
    return _round_float(_safe_rate(end_daily_average, before_daily_average))


def _build_monthly_metrics(
    df_this: pd.DataFrame,
    cat_deep: list[JsonValue],
    fixed_variable: JsonObject,
    this_total: float,
    prev_total: float,
    *,
    year: int,
    month: int,
    monthly_budget: float | int | None,
    monthly_income: float | int | None,
    salary_day: int | None,
) -> JsonObject:
    """문서의 월간 소비 분석 10개 핵심 지표와 특수 지표를 계산한다."""
    monthly_budget_usage_rate = (
        _safe_rate(this_total, float(monthly_budget)) * 100 if monthly_budget is not None else None
    )
    fixed_cost_amount = float(fixed_variable.get("fixed_total", 0))
    variable_cost_amount = float(fixed_variable.get("variable_total", 0))
    fixed_cost_burden_rate = (
        _safe_rate(fixed_cost_amount, float(monthly_income)) * 100
        if monthly_income is not None
        else None
    )
    essential_variable_total = float(
        df_this[
            df_this["업종 카테고리"].isin(_ESSENTIAL_CATEGORIES)
            & ~df_this["결제 방식 (온/오프라인)"]
            .astype("string")
            .str.contains(
                _FIXED_PAYMENT_KEYWORD,
                na=False,
            )
        ]["사용 금액"].sum()
    )
    spending_capacity = (
        _to_amount(float(monthly_income) - fixed_cost_amount - essential_variable_total)
        if monthly_income is not None
        else None
    )

    return {
        "monthly_total_amount": _to_amount(this_total),
        "monthly_budget_usage_rate_percent": None
        if monthly_budget_usage_rate is None
        else _round_float(monthly_budget_usage_rate),
        "previous_month_change_rate_percent": _round_float(
            _safe_rate(this_total - prev_total, prev_total) * 100
        ),
        "fixed_cost_amount": _to_amount(fixed_cost_amount),
        "fixed_cost_ratio_percent": fixed_variable.get("fixed_ratio_percent", 0.0),
        "variable_cost_amount": _to_amount(variable_cost_amount),
        "category_monthly_spending_ratio": _build_monthly_category_spending(cat_deep),
        "subscription_total": _build_subscription_total(df_this),
        "post_salary_spending_increase_rate_percent": _build_post_salary_spending_increase_rate(
            df_this,
            year=year,
            month=month,
            salary_day=salary_day,
        ),
        "month_end_pressure_index": _build_month_end_pressure_index(
            df_this,
            year=year,
            month=month,
        ),
        "special_metrics": {
            "fixed_cost_burden_rate_percent": None
            if fixed_cost_burden_rate is None
            else _round_float(fixed_cost_burden_rate),
            "spending_capacity": spending_capacity,
            "subscription_leakage_rate_percent": None,
        },
        "fixed_cost_burden_rate_percent": None
        if fixed_cost_burden_rate is None
        else _round_float(fixed_cost_burden_rate),
    }


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def build_monthly_consumption_analysis_from_frames(
    all_frame: pd.DataFrame,
    *,
    member_id: int = 1,
    analysis_month: str = "2024-04",
    source_path: str | Path | None = None,
    monthly_budget: float | int | None = None,
    monthly_income: float | int | None = None,
    salary_day: int | None = None,
) -> JsonObject:
    """과거+당월 소비 DataFrame에서 월간 소비 분석 JSON을 만든다.

    Parameters
    ----------
    all_frame:
        멤버 전체 거래 내역 (과거 + 당월 포함).
        ``멤버 id``, ``사용 금액``, ``사용 시간``, ``결제 내역``,
        ``업종 카테고리``, ``결제 방식 (온/오프라인)`` 컬럼 필수.
    member_id:
        분석 대상 멤버 ID.
    analysis_month:
        분석할 연-월 (``'YYYY-MM'`` 형식).
    source_path:
        데이터 원천 경로 (메타 정보용, 선택).

    Returns
    -------
    JsonObject
        월간 소비 분석 결과 딕셔너리.
    """
    _validate_columns(all_frame, "전체")

    year, month = _parse_ym(analysis_month)
    _, _, prev_month_str = _prev_ym(year, month)

    # 멤버 필터 & 타입 정리
    df = all_frame[all_frame["멤버 id"] == member_id].copy()
    if df.empty:
        raise ValueError(f"멤버 {member_id}번 거래를 찾을 수 없습니다.")

    df["사용 금액"] = pd.to_numeric(df["사용 금액"])
    df["사용 시간"] = pd.to_datetime(df["사용 시간"])
    df["date"] = df["사용 시간"].dt.date
    df["hour"] = df["사용 시간"].dt.hour
    df["weekday"] = df["사용 시간"].dt.dayofweek
    df["ym"] = df["사용 시간"].dt.to_period("M").astype(str)

    df_this = df[df["ym"] == analysis_month].copy()
    df_prev = df[df["ym"] == prev_month_str].copy()

    # IQR 상한선 (당월 제외 과거 전체 기준)
    df_base = df[df["ym"] != analysis_month].copy()
    if df_base.empty:
        df_base = df
    q1 = float(df_base["사용 금액"].quantile(0.25))
    q3 = float(df_base["사용 금액"].quantile(0.75))
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    this_total = float(df_this["사용 금액"].sum())
    prev_total = float(df_prev["사용 금액"].sum())

    # 섹션별 빌드
    monthly_summary = _build_monthly_summary(
        df_this, df_prev, this_total, prev_total, analysis_month
    )
    fixed_variable = _build_fixed_variable(df_this, this_total)
    cat_deep, top_savable = _build_category_deep(df_this, df_prev, this_total)
    monthly_metrics = _build_monthly_metrics(
        df_this,
        cat_deep,
        fixed_variable,
        this_total,
        prev_total,
        year=year,
        month=month,
        monthly_budget=monthly_budget,
        monthly_income=monthly_income,
        salary_day=salary_day,
    )
    repeat_monthly = _build_repeat_monthly(df_this)
    weekly_trend = _build_weekly_trend(df_this, year, month)
    micro_monthly = _build_micro_monthly(df_this, this_total)
    late_night_monthly = _build_late_night_monthly(df_this, this_total)
    high_spending_monthly = _build_high_spending_monthly(df_this, upper_bound)
    saving_monthly = _build_saving_monthly(
        repeat_monthly["delivery"],  # type: ignore[arg-type]
        repeat_monthly["cafe"],  # type: ignore[arg-type]
        repeat_monthly["taxi"],  # type: ignore[arg-type]
        float(df_this[df_this["사용 금액"] < _MICRO_THRESHOLD]["사용 금액"].sum()),
        this_total,
    )
    cash_flow_volatility = _build_cash_flow_volatility(weekly_trend)
    spending_concentration = _build_spending_concentration(df_this)
    frictionless_and_density = _build_frictionless_and_density(df_this, this_total)
    installment_debt_pressure = _build_installment_debt_pressure(df_this, this_total)
    monthly_comparisons = _build_monthly_comparisons(
        df,
        year=year,
        month=month,
        analysis_month=analysis_month,
        prev_month=prev_month_str,
        current_total=this_total,
        current_count=int(len(df_this)),
    )

    return {
        "member_id": member_id,
        "analysis_month": analysis_month,
        "prev_month": prev_month_str,
        "source_path": str(source_path) if source_path is not None else None,
        "outlier_thresholds": {
            "q1": _round_float(q1),
            "q3": _round_float(q3),
            "iqr": _round_float(iqr),
            "upper_bound": _round_float(upper_bound),
        },
        "monthly_summary": monthly_summary,
        "monthly_comparisons": monthly_comparisons,
        "monthly_metrics": monthly_metrics,
        "fixed_variable": fixed_variable,
        "category_deep": cat_deep,
        "top_savable_categories": top_savable,
        "repeat_patterns": repeat_monthly,
        "weekly_trend": weekly_trend,
        "micro_spending": micro_monthly,
        "late_night_spending": late_night_monthly,
        "high_spending": high_spending_monthly,
        "saving_potential": saving_monthly,
        "cash_flow_volatility": cash_flow_volatility,
        "spending_concentration": spending_concentration,
        "frictionless_and_density": frictionless_and_density,
        "installment_debt_pressure": installment_debt_pressure,
    }
