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

_FIXED_PAYMENT_KEYWORD = "자동이체"
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


def _prev_ym(year: int, month: int) -> tuple[int, int, str]:
    """전월 연도·월·'YYYY-MM' 문자열을 반환한다."""
    prev_year = year - 1 if month == 1 else year
    prev_month = 12 if month == 1 else month - 1
    return prev_year, prev_month, f"{prev_year}-{prev_month:02d}"


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


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def build_monthly_consumption_analysis_from_frames(
    all_frame: pd.DataFrame,
    *,
    member_id: int = 1,
    analysis_month: str = "2024-04",
    source_path: str | Path | None = None,
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
        "fixed_variable": fixed_variable,
        "category_deep": cat_deep,
        "top_savable_categories": top_savable,
        "repeat_patterns": repeat_monthly,
        "weekly_trend": weekly_trend,
        "micro_spending": micro_monthly,
        "late_night_spending": late_night_monthly,
        "high_spending": high_spending_monthly,
        "saving_potential": saving_monthly,
    }
