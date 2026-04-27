from __future__ import annotations

from datetime import date, timedelta
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

    all_cats = sorted(set(this_cat.index) | set(prev_cat.index))
    rows: list[JsonValue] = []
    for cat in all_cats:
        ta = float(this_cat.get(cat, 0))
        pa = float(prev_cat.get(cat, 0))
        diff = ta - pa
        ratio = _safe_rate(ta, this_total) * 100
        diff_rate = _safe_rate(diff, pa) * 100 if pa != 0 else 0.0
        rows.append(
            {
                "category": str(cat),
                "total_amount": _to_amount(ta),
                "ratio_percent": _round_float(ratio),
                "transaction_count": int(this_cat_cnt.get(cat, 0)),
                "prev_week_amount": _to_amount(pa),
                "diff_amount": _to_amount(diff),
                "diff_rate_percent": _round_float(diff_rate),
            }
        )
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
    waste_detection = _build_waste_detection(df_this, this_total, upper_bound)
    saving_potential = _build_saving_potential(
        category_summary,
        repeat_patterns["delivery"],  # type: ignore[arg-type]
        repeat_patterns["cafe"],  # type: ignore[arg-type]
    )

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
        "repeat_patterns": repeat_patterns,
        "weekday_pattern": weekday_pattern,
        "waste_detection": waste_detection,
        "saving_potential": saving_potential,
    }
