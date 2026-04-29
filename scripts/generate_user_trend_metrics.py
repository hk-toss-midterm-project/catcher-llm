"""현재 사용자/거래 CSV에서 전체 소비 동향 지표와 matplotlib 차트를 생성한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from catcher_llm.analysis.transaction_schema import normalize_transaction_frame
from catcher_llm.analysis.user_monthly_analysis import (
    build_monthly_consumption_analysis_from_frames,
)

DEFAULT_USERS_PATH = Path("data/raw/csv/users_v3.csv")
DEFAULT_TRANSACTIONS_PATH = Path("data/raw/csv/transactions_v3.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/user_trend_metrics")

_APPROVED_STATUS = "APPROVED"
_CANCELLED_STATUS = "CANCELLED"
_SOURCE_MODULE = "catcher_llm.analysis.user_monthly_analysis"
_LATE_NIGHT_START_HOUR = 22
_PERCENT_DIGITS = 4

_TRANSACTION_REQUIRED_COLUMNS = {
    "멤버 id",
    "사용 금액",
    "사용 시간",
    "결제 내역",
    "업종 카테고리",
    "결제 방식 (온/오프라인)",
}
_USER_REQUIRED_COLUMNS = {
    "id",
    "annual_income",
    "target_max_spending_amount",
}


@dataclass(frozen=True)
class TrendOutputPaths:
    """생성된 지표 CSV 경로와 차트 PNG 경로를 묶어서 반환한다."""

    csv_paths: dict[str, Path]
    chart_paths: dict[str, Path]


def _safe_rate(numerator: float, denominator: float) -> float:
    """분모가 0인 비율 계산을 0으로 안전하게 처리한다."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round_float(value: float | int, digits: int = _PERCENT_DIGITS) -> float:
    """CSV로 읽기 쉬운 실수 지표가 되도록 소수 자릿수를 제한한다."""
    return round(float(value), digits)


def _to_amount(value: float | int) -> int:
    """원화 금액 지표를 정수 단위로 변환한다."""
    return int(round(float(value)))


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], label: str) -> None:
    """입력 DataFrame이 지표 생성에 필요한 컬럼을 모두 갖췄는지 검증한다."""
    missing_columns = sorted(required_columns.difference(str(column) for column in frame.columns))
    if missing_columns:
        raise ValueError(f"{label} 데이터에 필요한 컬럼이 없습니다: {missing_columns}")


def _age_group(age: int | float) -> str:
    """사용자 나이를 소비 동향 비교용 연령대로 변환한다."""
    age_value = int(age)
    if age_value < 30:
        return "20대 이하"
    if age_value < 40:
        return "30대"
    if age_value < 50:
        return "40대"
    if age_value < 60:
        return "50대"
    return "60대 이상"


def _income_group(annual_income: int | float) -> str:
    """연 소득을 전체 동향 비교용 소득 구간으로 변환한다."""
    income = float(annual_income)
    if income < 30_000_000:
        return "3천만원 미만"
    if income < 50_000_000:
        return "3천~5천만원"
    if income < 70_000_000:
        return "5천~7천만원"
    return "7천만원 이상"


def _normalize_users_frame(users_frame: pd.DataFrame) -> pd.DataFrame:
    """사용자 CSV를 지표 집계에 필요한 숫자형·세그먼트 컬럼으로 정리한다."""
    _validate_columns(users_frame, _USER_REQUIRED_COLUMNS, "사용자")
    users = users_frame.copy()
    users["id"] = pd.to_numeric(users["id"]).astype("int64")
    users["annual_income"] = pd.to_numeric(users["annual_income"]).fillna(0)
    users["target_max_spending_amount"] = pd.to_numeric(
        users["target_max_spending_amount"],
    ).fillna(0)
    users["monthly_income"] = users["annual_income"] / 12

    if "age" in users.columns:
        users["age"] = pd.to_numeric(users["age"]).fillna(0)
        users["age_group"] = users["age"].apply(_age_group)
    else:
        users["age_group"] = "미상"

    if "annual_income" in users.columns:
        users["income_group"] = users["annual_income"].apply(_income_group)
    else:
        users["income_group"] = "미상"

    for optional_column in ["gender", "region"]:
        if optional_column not in users.columns:
            users[optional_column] = "미상"
    if "personal_score" in users.columns:
        users["personal_score"] = pd.to_numeric(users["personal_score"]).fillna(0)
    else:
        users["personal_score"] = 0
    return users


def _normalize_transactions_frame(transactions_frame: pd.DataFrame) -> pd.DataFrame:
    """v3 거래 CSV를 기존 분석 모듈이 쓰는 한글 표준 컬럼과 파생 기간 컬럼으로 정리한다."""
    transactions = normalize_transaction_frame(transactions_frame)
    _validate_columns(transactions, _TRANSACTION_REQUIRED_COLUMNS, "거래")

    transactions = transactions.copy()
    transactions["멤버 id"] = pd.to_numeric(transactions["멤버 id"]).astype("int64")
    transactions["사용 금액"] = pd.to_numeric(transactions["사용 금액"]).fillna(0)
    transactions["사용 시간"] = pd.to_datetime(transactions["사용 시간"])
    transactions["date"] = transactions["사용 시간"].dt.date
    transactions["month"] = transactions["사용 시간"].dt.to_period("M").astype(str)
    transactions["week_start"] = (
        transactions["사용 시간"] - pd.to_timedelta(transactions["사용 시간"].dt.weekday, unit="D")
    ).dt.date
    transactions["week_end"] = (
        transactions["사용 시간"]
        + pd.to_timedelta(6 - transactions["사용 시간"].dt.weekday, unit="D")
    ).dt.date
    transactions["hour"] = transactions["사용 시간"].dt.hour
    transactions["weekday"] = transactions["사용 시간"].dt.dayofweek

    if "status" not in transactions.columns:
        transactions["status"] = _APPROVED_STATUS
    transactions["status"] = transactions["status"].astype("string").fillna(_APPROVED_STATUS)

    if "is_installment" in transactions.columns and "할부 여부" not in transactions.columns:
        transactions["할부 여부"] = transactions["is_installment"].map(
            lambda value: "Y" if bool(value) else "N",
        )
    if "할부 여부" not in transactions.columns:
        transactions["할부 여부"] = "N"

    if "installment_months" in transactions.columns and "할부 개월" not in transactions.columns:
        transactions["할부 개월"] = pd.to_numeric(transactions["installment_months"]).fillna(0)
    if "할부 개월" not in transactions.columns:
        transactions["할부 개월"] = 0

    if "is_overseas" not in transactions.columns:
        transactions["is_overseas"] = False
    transactions["is_overseas"] = transactions["is_overseas"].fillna(False).astype(bool)

    if "merchant_name" in transactions.columns:
        transactions["가맹점명"] = transactions["merchant_name"].astype("string").fillna("")
    else:
        transactions["가맹점명"] = transactions["결제 내역"].astype("string").fillna("")
    return transactions


def _trend_label(change_rate_percent: float | None) -> str:
    """증감률을 LLM이 읽기 쉬운 상승·하락·유지 라벨로 변환한다."""
    if change_rate_percent is None:
        return "비교불가"
    if change_rate_percent > 5:
        return "상승"
    if change_rate_percent < -5:
        return "하락"
    return "유지"


def _series_percent_change(series: pd.Series) -> pd.Series:
    """월별 정렬이 끝난 수치 Series의 전 기간 대비 증감률을 계산한다."""
    previous = series.shift(1)
    return ((series - previous) / previous * 100).where(previous != 0)


def _top_category_rows(approved_transactions: pd.DataFrame) -> pd.DataFrame:
    """월별 최상위 카테고리와 금액·비중을 계산한다."""
    category_amounts = (
        approved_transactions.groupby(["month", "업종 카테고리"], dropna=False)["사용 금액"]
        .sum()
        .reset_index(name="category_amount")
    )
    if category_amounts.empty:
        return pd.DataFrame(
            columns=["month", "top_category", "top_category_amount", "top_category_ratio_percent"],
        )

    month_totals = approved_transactions.groupby("month")["사용 금액"].sum()
    category_amounts = category_amounts.sort_values(
        ["month", "category_amount", "업종 카테고리"],
        ascending=[True, False, True],
    )
    top_categories = category_amounts.drop_duplicates("month").copy()
    top_categories["top_category_ratio_percent"] = top_categories.apply(
        lambda row: _round_float(
            _safe_rate(
                float(row["category_amount"]),
                float(month_totals.get(cast(str, row["month"]), 0)),
            )
            * 100,
        ),
        axis=1,
    )
    return top_categories.rename(
        columns={
            "업종 카테고리": "top_category",
            "category_amount": "top_category_amount",
        },
    )[["month", "top_category", "top_category_amount", "top_category_ratio_percent"]]


def _build_monthly_trends(
    *,
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    approved_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """문서의 월간 지표 축에 맞춰 전체 사용자 월별 소비 동향을 집계한다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    monthly = (
        approved_transactions.groupby("month")
        .agg(
            active_user_count=("멤버 id", "nunique"),
            approved_transaction_count=("사용 금액", "size"),
            total_amount=("사용 금액", "sum"),
            online_amount=(
                "사용 금액",
                lambda values: float(
                    values[
                        approved_transactions.loc[
                            values.index,
                            "결제 방식 (온/오프라인)",
                        ]
                        == "온라인"
                    ].sum(),
                ),
            ),
            installment_amount=(
                "사용 금액",
                lambda values: float(
                    values[approved_transactions.loc[values.index, "할부 여부"] == "Y"].sum(),
                ),
            ),
            overseas_amount=(
                "사용 금액",
                lambda values: float(
                    values[approved_transactions.loc[values.index, "is_overseas"]].sum(),
                ),
            ),
            late_night_amount=(
                "사용 금액",
                lambda values: float(
                    values[
                        approved_transactions.loc[values.index, "hour"] >= _LATE_NIGHT_START_HOUR
                    ].sum(),
                ),
            ),
        )
        .reset_index()
        .sort_values("month")
    )

    cancelled_counts = (
        transactions[transactions["status"] == _CANCELLED_STATUS]
        .groupby("month")
        .size()
        .rename("cancelled_transaction_count")
    )
    monthly = monthly.merge(cancelled_counts, on="month", how="left")
    monthly["cancelled_transaction_count"] = (
        monthly["cancelled_transaction_count"].fillna(0).astype("int64")
    )

    user_month_totals = (
        approved_transactions.groupby(["month", "멤버 id"])["사용 금액"]
        .sum()
        .reset_index(name="user_monthly_amount")
    )
    median_user_amounts = (
        user_month_totals.groupby("month")["user_monthly_amount"]
        .median()
        .rename("median_user_monthly_amount")
    )
    monthly = monthly.merge(median_user_amounts, on="month", how="left")

    active_user_budgets = (
        user_month_totals.merge(
            users[["id", "target_max_spending_amount"]],
            left_on="멤버 id",
            right_on="id",
            how="left",
        )
        .groupby("month")["target_max_spending_amount"]
        .sum()
        .rename("target_budget_total")
    )
    monthly = monthly.merge(active_user_budgets, on="month", how="left")
    monthly = monthly.merge(_top_category_rows(approved_transactions), on="month", how="left")

    monthly["average_amount_per_active_user"] = monthly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["active_user_count"])),
        ),
        axis=1,
    )
    monthly["average_transaction_amount"] = monthly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["approved_transaction_count"])),
        ),
        axis=1,
    )
    monthly["budget_usage_rate_percent"] = monthly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["target_budget_total"])) * 100,
        ),
        axis=1,
    )
    for source_column, target_column in [
        ("online_amount", "online_ratio_percent"),
        ("installment_amount", "installment_ratio_percent"),
        ("overseas_amount", "overseas_ratio_percent"),
        ("late_night_amount", "late_night_ratio_percent"),
    ]:
        monthly[target_column] = monthly.apply(
            lambda row, column=source_column: _round_float(
                _safe_rate(float(row[column]), float(row["total_amount"])) * 100,
            ),
            axis=1,
        )
    monthly["prev_month_total_amount"] = monthly["total_amount"].shift(1)
    monthly["prev_month_change_rate_percent"] = _series_percent_change(monthly["total_amount"])

    amount_columns = [
        "total_amount",
        "online_amount",
        "installment_amount",
        "overseas_amount",
        "late_night_amount",
        "target_budget_total",
        "top_category_amount",
    ]
    for column in amount_columns:
        if column in monthly.columns:
            monthly[column] = monthly[column].fillna(0).round().astype("int64")
    return monthly


def _build_daily_trends(approved_transactions: pd.DataFrame, total_user_count: int) -> pd.DataFrame:
    """문서의 일일 지표 축에 맞춰 전체 사용자 날짜별 소비 동향을 집계한다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    daily = (
        approved_transactions.groupby("date")
        .agg(
            active_user_count=("멤버 id", "nunique"),
            transaction_count=("사용 금액", "size"),
            total_amount=("사용 금액", "sum"),
            online_amount=(
                "사용 금액",
                lambda values: float(
                    values[
                        approved_transactions.loc[
                            values.index,
                            "결제 방식 (온/오프라인)",
                        ]
                        == "온라인"
                    ].sum(),
                ),
            ),
            late_night_amount=(
                "사용 금액",
                lambda values: float(
                    values[
                        approved_transactions.loc[values.index, "hour"] >= _LATE_NIGHT_START_HOUR
                    ].sum(),
                ),
            ),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["date"] = daily["date"].astype(str)
    daily["no_spending_user_count"] = total_user_count - daily["active_user_count"]
    daily["average_amount_per_active_user"] = daily.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["active_user_count"])),
        ),
        axis=1,
    )
    daily["average_transaction_amount"] = daily.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["transaction_count"])),
        ),
        axis=1,
    )
    daily["online_ratio_percent"] = daily.apply(
        lambda row: _round_float(
            _safe_rate(float(row["online_amount"]), float(row["total_amount"])) * 100,
        ),
        axis=1,
    )
    daily["late_night_ratio_percent"] = daily.apply(
        lambda row: _round_float(
            _safe_rate(float(row["late_night_amount"]), float(row["total_amount"])) * 100,
        ),
        axis=1,
    )
    return daily


def _build_weekly_trends(approved_transactions: pd.DataFrame) -> pd.DataFrame:
    """문서의 주간 지표 축에 맞춰 전체 사용자 주별 소비 동향을 집계한다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    weekly = (
        approved_transactions.groupby(["week_start", "week_end"])
        .agg(
            active_user_count=("멤버 id", "nunique"),
            transaction_count=("사용 금액", "size"),
            total_amount=("사용 금액", "sum"),
            weekday_amount=(
                "사용 금액",
                lambda values: float(
                    values[approved_transactions.loc[values.index, "weekday"] < 5].sum(),
                ),
            ),
            weekend_amount=(
                "사용 금액",
                lambda values: float(
                    values[approved_transactions.loc[values.index, "weekday"] >= 5].sum(),
                ),
            ),
        )
        .reset_index()
        .sort_values("week_start")
    )
    daily_by_week = (
        approved_transactions.groupby(["week_start", "date"])["사용 금액"]
        .sum()
        .reset_index(name="daily_amount")
    )
    volatility = (
        daily_by_week.groupby("week_start")["daily_amount"]
        .std(ddof=0)
        .fillna(0)
        .rename("weekly_spending_volatility")
    )
    weekly = weekly.merge(volatility, on="week_start", how="left")
    weekly["week_start"] = weekly["week_start"].astype(str)
    weekly["week_end"] = weekly["week_end"].astype(str)
    weekly["average_daily_amount"] = weekly["total_amount"].apply(
        lambda value: _round_float(_safe_rate(float(value), 7.0)),
    )
    weekly["weekday_spending_ratio_percent"] = weekly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["weekday_amount"]), float(row["total_amount"])) * 100,
        ),
        axis=1,
    )
    weekly["weekend_spending_ratio_percent"] = weekly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["weekend_amount"]), float(row["total_amount"])) * 100,
        ),
        axis=1,
    )
    weekly["weekend_overspending_index"] = weekly.apply(
        lambda row: _round_float(
            _safe_rate(float(row["weekend_amount"]) / 2, float(row["weekday_amount"]) / 5),
        ),
        axis=1,
    )
    weekly["prev_week_change_rate_percent"] = _series_percent_change(weekly["total_amount"])
    return weekly


def _build_category_monthly_trends(approved_transactions: pd.DataFrame) -> pd.DataFrame:
    """월별 카테고리 금액·비중·전월 대비 변화를 계산한다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    category = (
        approved_transactions.groupby(["month", "업종 카테고리"], dropna=False)
        .agg(
            total_amount=("사용 금액", "sum"),
            transaction_count=("사용 금액", "size"),
            active_user_count=("멤버 id", "nunique"),
        )
        .reset_index()
        .rename(columns={"업종 카테고리": "category"})
        .sort_values(["category", "month"])
    )
    month_totals = approved_transactions.groupby("month")["사용 금액"].sum().rename("month_total")
    category = category.merge(month_totals, on="month", how="left")
    category["monthly_ratio_percent"] = category.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["month_total"])) * 100,
        ),
        axis=1,
    )
    category["prev_month_category_amount"] = category.groupby("category")["total_amount"].shift(1)
    category["prev_month_change_rate_percent"] = category.groupby("category")[
        "total_amount"
    ].transform(_series_percent_change)
    category = category.sort_values(["month", "total_amount"], ascending=[True, False])
    return category


def _build_segment_monthly_trends(
    *,
    users: pd.DataFrame,
    approved_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """사용자 속성별 월간 소비 차이를 비교할 수 있는 세그먼트 지표를 만든다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    enriched = approved_transactions.merge(
        users[
            [
                "id",
                "age_group",
                "gender",
                "income_group",
                "target_max_spending_amount",
                "personal_score",
            ]
        ],
        left_on="멤버 id",
        right_on="id",
        how="left",
    )
    segment_spending = (
        enriched.groupby(["month", "age_group", "gender"], dropna=False)
        .agg(
            total_amount=("사용 금액", "sum"),
            active_user_count=("멤버 id", "nunique"),
            transaction_count=("사용 금액", "size"),
        )
        .reset_index()
    )
    segment_users = enriched.drop_duplicates(["month", "age_group", "gender", "멤버 id"])
    segment_user_metrics = (
        segment_users.groupby(["month", "age_group", "gender"], dropna=False)
        .agg(
            average_personal_score=("personal_score", "mean"),
            target_budget_total=("target_max_spending_amount", "sum"),
        )
        .reset_index()
    )
    segment = segment_spending.merge(
        segment_user_metrics,
        on=["month", "age_group", "gender"],
        how="left",
    ).sort_values(["month", "total_amount"], ascending=[True, False])
    segment["average_amount_per_active_user"] = segment.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["active_user_count"])),
        ),
        axis=1,
    )
    segment["budget_usage_rate_percent"] = segment.apply(
        lambda row: _round_float(
            _safe_rate(float(row["total_amount"]), float(row["target_budget_total"])) * 100,
        ),
        axis=1,
    )
    segment["average_personal_score"] = segment["average_personal_score"].fillna(0).round(4)
    return segment


def _first_dict_item(rows: object) -> dict[str, object]:
    """분석 결과의 리스트형 섹션에서 첫 번째 dict 항목을 안전하게 꺼낸다."""
    if not isinstance(rows, list) or not rows:
        return {}
    first = rows[0]
    if isinstance(first, dict):
        return first
    return {}


def _nested_number(row: dict[str, object], key: str) -> float:
    """dict 결과에서 수치 키를 float으로 안전하게 읽는다."""
    value = row.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _build_user_monthly_metrics(
    *,
    users: pd.DataFrame,
    approved_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """기존 월간 분석 모듈을 사용자-월마다 호출해 개인별 핵심 지표를 만든다."""
    if approved_transactions.empty:
        return pd.DataFrame()

    user_lookup = users.set_index("id", drop=False)
    active_user_months = (
        approved_transactions[["멤버 id", "month"]]
        .drop_duplicates()
        .sort_values(["멤버 id", "month"])
    )
    rows: list[dict[str, object]] = []
    for _, active_row in active_user_months.iterrows():
        user_id = int(active_row["멤버 id"])
        month = str(active_row["month"])
        user_row = user_lookup.loc[user_id] if user_id in user_lookup.index else None
        monthly_budget = (
            float(cast(pd.Series, user_row)["target_max_spending_amount"])
            if user_row is not None
            else None
        )
        monthly_income = (
            float(cast(pd.Series, user_row)["monthly_income"]) if user_row is not None else None
        )
        result = build_monthly_consumption_analysis_from_frames(
            approved_transactions,
            member_id=user_id,
            analysis_month=month,
            monthly_budget=monthly_budget,
            monthly_income=monthly_income,
        )

        monthly_summary = cast(dict[str, object], result["monthly_summary"])
        monthly_metrics = cast(dict[str, object], result["monthly_metrics"])
        top_category = _first_dict_item(result.get("category_deep"))
        frictionless = cast(dict[str, object], result["frictionless_and_density"])
        frictionless_spending = cast(dict[str, object], frictionless["frictionless_spending"])
        installment = cast(dict[str, object], result["installment_debt_pressure"])

        rows.append(
            {
                "user_id": user_id,
                "month": month,
                "monthly_total_amount": _to_amount(
                    _nested_number(monthly_metrics, "monthly_total_amount")
                ),
                "transaction_count": int(_nested_number(monthly_summary, "transaction_count")),
                "monthly_budget_usage_rate_percent": monthly_metrics.get(
                    "monthly_budget_usage_rate_percent",
                ),
                "previous_month_change_rate_percent": monthly_metrics.get(
                    "previous_month_change_rate_percent",
                ),
                "fixed_cost_amount": _to_amount(
                    _nested_number(monthly_metrics, "fixed_cost_amount")
                ),
                "variable_cost_amount": _to_amount(
                    _nested_number(monthly_metrics, "variable_cost_amount"),
                ),
                "subscription_total": _to_amount(
                    _nested_number(monthly_metrics, "subscription_total")
                ),
                "top_category": str(top_category.get("category", "")),
                "top_category_amount": _to_amount(_nested_number(top_category, "total_amount")),
                "top_category_ratio_percent": top_category.get("ratio_percent", 0.0),
                "frictionless_spending_ratio_percent": frictionless_spending.get(
                    "ratio_percent",
                    0.0,
                ),
                "installment_amount": _to_amount(
                    _nested_number(installment, "total_installment_amount"),
                ),
                "installment_ratio_percent": installment.get("installment_ratio_percent", 0.0),
                "source_module": _SOURCE_MODULE,
            },
        )
    return pd.DataFrame(rows)


def _build_llm_trend_metrics(monthly_trends: pd.DataFrame) -> pd.DataFrame:
    """월별 동향을 LLM이 읽기 쉬운 long-form 지표 CSV 구조로 변환한다."""
    rows: list[dict[str, object]] = []
    metric_specs: list[tuple[str, str, str, str]] = [
        ("total_amount", "monthly_total_amount", "KRW", "월간 승인 소비 총액"),
        (
            "average_amount_per_active_user",
            "average_amount_per_active_user",
            "KRW",
            "활성 사용자 1인당 월간 소비",
        ),
        ("active_user_count", "active_user_count", "users", "해당 월 소비가 있는 사용자 수"),
        (
            "approved_transaction_count",
            "approved_transaction_count",
            "transactions",
            "승인 거래 건수",
        ),
        ("online_ratio_percent", "online_ratio_percent", "percent", "온라인 결제 금액 비중"),
        (
            "installment_ratio_percent",
            "installment_ratio_percent",
            "percent",
            "할부 결제 금액 비중",
        ),
        (
            "late_night_ratio_percent",
            "late_night_ratio_percent",
            "percent",
            "22시 이후 야간 소비 금액 비중",
        ),
        (
            "budget_usage_rate_percent",
            "budget_usage_rate_percent",
            "percent",
            "활성 사용자 목표 지출 한도 대비 사용률",
        ),
    ]
    for _, month_row in monthly_trends.iterrows():
        change_value = month_row.get("prev_month_change_rate_percent")
        change_rate = None if pd.isna(change_value) else float(change_value)
        compare_period = None
        prev_month_value = month_row.get("prev_month_total_amount")
        if not pd.isna(prev_month_value):
            compare_period = "previous_month"

        for source_column, metric_name, unit, context in metric_specs:
            value = month_row.get(source_column)
            rows.append(
                {
                    "period_type": "month",
                    "period": str(month_row["month"]),
                    "metric_name": metric_name,
                    "metric_value": None if pd.isna(value) else float(value),
                    "metric_unit": unit,
                    "compare_period": compare_period,
                    "change_rate_percent": change_rate
                    if metric_name == "monthly_total_amount"
                    else None,
                    "trend_label": _trend_label(change_rate)
                    if metric_name == "monthly_total_amount"
                    else "참고",
                    "context": context,
                },
            )

        top_category = month_row.get("top_category")
        if isinstance(top_category, str) and top_category:
            rows.append(
                {
                    "period_type": "month",
                    "period": str(month_row["month"]),
                    "metric_name": "top_category_amount",
                    "metric_value": float(month_row.get("top_category_amount", 0)),
                    "metric_unit": "KRW",
                    "compare_period": None,
                    "change_rate_percent": None,
                    "trend_label": "참고",
                    "context": f"월간 최대 소비 카테고리: {top_category}",
                },
            )
    return pd.DataFrame(rows)


def build_trend_metric_tables(
    *,
    users_frame: pd.DataFrame,
    transactions_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """현재 사용자·거래 데이터에서 LLM/사람 확인용 동향 지표 테이블 묶음을 생성한다."""
    users = _normalize_users_frame(users_frame)
    transactions = _normalize_transactions_frame(transactions_frame)
    approved_transactions = transactions[transactions["status"] == _APPROVED_STATUS].copy()

    monthly_trends = _build_monthly_trends(
        users=users,
        transactions=transactions,
        approved_transactions=approved_transactions,
    )
    daily_trends = _build_daily_trends(approved_transactions, total_user_count=len(users))
    weekly_trends = _build_weekly_trends(approved_transactions)
    category_monthly_trends = _build_category_monthly_trends(approved_transactions)
    segment_monthly_trends = _build_segment_monthly_trends(
        users=users,
        approved_transactions=approved_transactions,
    )
    user_monthly_metrics = _build_user_monthly_metrics(
        users=users,
        approved_transactions=approved_transactions,
    )
    llm_trend_metrics = _build_llm_trend_metrics(monthly_trends)

    return {
        "llm_trend_metrics": llm_trend_metrics,
        "monthly_trends": monthly_trends,
        "daily_trends": daily_trends,
        "weekly_trends": weekly_trends,
        "category_monthly_trends": category_monthly_trends,
        "segment_monthly_trends": segment_monthly_trends,
        "user_monthly_metrics": user_monthly_metrics,
    }


def load_input_frames(
    *,
    users_path: Path = DEFAULT_USERS_PATH,
    transactions_path: Path = DEFAULT_TRANSACTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """기본 CSV 파일을 UTF-8 BOM 가능성을 고려해 DataFrame으로 읽는다."""
    users_frame = pd.read_csv(users_path, encoding="utf-8-sig")
    transactions_frame = pd.read_csv(transactions_path, encoding="utf-8-sig")
    return users_frame, transactions_frame


def _configure_matplotlib() -> object:
    """서버 환경에서 PNG 저장이 가능하도록 matplotlib 백엔드와 한글 폰트를 설정한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
        if font_name in available_font_names:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save_monthly_total_chart(monthly_trends: pd.DataFrame, chart_path: Path) -> None:
    """월간 총액과 사용자 1인당 소비 추이를 선 그래프로 저장한다."""
    if monthly_trends.empty:
        return

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        monthly_trends["month"],
        monthly_trends["total_amount"],
        marker="o",
        label="월간 총 소비",
    )
    axis.plot(
        monthly_trends["month"],
        monthly_trends["average_amount_per_active_user"],
        marker="o",
        label="활성 사용자 1인당 소비",
    )
    axis.set_title("월간 소비 총액과 1인당 소비 추이")
    axis.set_xlabel("월")
    axis.set_ylabel("금액(원)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def _save_category_monthly_chart(category_monthly_trends: pd.DataFrame, chart_path: Path) -> None:
    """월별 주요 카테고리 소비 금액을 누적 막대 그래프로 저장한다."""
    if category_monthly_trends.empty:
        return

    top_categories = (
        category_monthly_trends.groupby("category")["total_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    chart_frame = category_monthly_trends[
        category_monthly_trends["category"].isin(top_categories)
    ].copy()
    pivot = chart_frame.pivot_table(
        index="month",
        columns="category",
        values="total_amount",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title("월별 주요 카테고리 소비 구성")
    axis.set_xlabel("월")
    axis.set_ylabel("금액(원)")
    axis.legend(title="카테고리", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def _save_payment_behavior_chart(monthly_trends: pd.DataFrame, chart_path: Path) -> None:
    """온라인·할부·야간 소비 비중을 월별 선 그래프로 저장한다."""
    if monthly_trends.empty:
        return

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 5))
    for column, label in [
        ("online_ratio_percent", "온라인"),
        ("installment_ratio_percent", "할부"),
        ("late_night_ratio_percent", "야간"),
    ]:
        axis.plot(monthly_trends["month"], monthly_trends[column], marker="o", label=label)
    axis.set_title("월별 결제 행동 비중")
    axis.set_xlabel("월")
    axis.set_ylabel("비중(%)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def _save_weekly_total_chart(weekly_trends: pd.DataFrame, chart_path: Path) -> None:
    """주간 소비 총액과 주말 소비 비중을 함께 확인하는 그래프를 저장한다."""
    if weekly_trends.empty:
        return

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        weekly_trends["week_start"],
        weekly_trends["total_amount"],
        marker="o",
        label="주간 총 소비",
    )
    axis.set_title("주간 소비 총액 추이")
    axis.set_xlabel("주 시작일")
    axis.set_ylabel("금액(원)")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def _save_segment_chart(segment_monthly_trends: pd.DataFrame, chart_path: Path) -> None:
    """연령대별 월간 활성 사용자 1인당 소비 추이를 저장한다."""
    if segment_monthly_trends.empty:
        return

    pivot = segment_monthly_trends.pivot_table(
        index="month",
        columns="age_group",
        values="average_amount_per_active_user",
        aggfunc="mean",
        fill_value=0,
    ).sort_index()

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(marker="o", ax=axis)
    axis.set_title("연령대별 1인당 월간 소비 추이")
    axis.set_xlabel("월")
    axis.set_ylabel("금액(원)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="연령대", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)


def _save_charts(tables: dict[str, pd.DataFrame], charts_dir: Path) -> dict[str, Path]:
    """생성된 지표 테이블을 사람이 확인하기 쉬운 PNG 차트 묶음으로 저장한다."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = {
        "monthly_total_amount": charts_dir / "monthly_total_amount.png",
        "category_monthly_amount": charts_dir / "category_monthly_amount.png",
        "payment_behavior_ratios": charts_dir / "payment_behavior_ratios.png",
        "weekly_total_amount": charts_dir / "weekly_total_amount.png",
        "segment_monthly_amount": charts_dir / "segment_monthly_amount.png",
    }
    _save_monthly_total_chart(tables["monthly_trends"], chart_paths["monthly_total_amount"])
    _save_category_monthly_chart(
        tables["category_monthly_trends"],
        chart_paths["category_monthly_amount"],
    )
    _save_payment_behavior_chart(
        tables["monthly_trends"],
        chart_paths["payment_behavior_ratios"],
    )
    _save_weekly_total_chart(tables["weekly_trends"], chart_paths["weekly_total_amount"])
    _save_segment_chart(tables["segment_monthly_trends"], chart_paths["segment_monthly_amount"])
    return chart_paths


def save_trend_outputs(
    *,
    tables: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TrendOutputPaths:
    """지표 테이블 CSV와 matplotlib 차트를 지정한 출력 디렉터리에 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: dict[str, Path] = {}
    for table_name, table in tables.items():
        csv_path = output_dir / f"{table_name}.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        csv_paths[table_name] = csv_path

    chart_paths = _save_charts(tables, output_dir / "charts")
    return TrendOutputPaths(csv_paths=csv_paths, chart_paths=chart_paths)


def build_and_save_trend_metrics(
    *,
    users_path: Path = DEFAULT_USERS_PATH,
    transactions_path: Path = DEFAULT_TRANSACTIONS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TrendOutputPaths:
    """기본 CSV를 읽고 전체 동향 지표 CSV와 차트를 한 번에 생성한다."""
    users_frame, transactions_frame = load_input_frames(
        users_path=users_path,
        transactions_path=transactions_path,
    )
    tables = build_trend_metric_tables(
        users_frame=users_frame,
        transactions_frame=transactions_frame,
    )
    return save_trend_outputs(tables=tables, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    """CLI에서 입력 CSV 경로와 출력 디렉터리 옵션을 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS_PATH)
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI 진입점으로 동향 지표 생성 결과 경로를 출력한다."""
    args = _parse_args()
    result = build_and_save_trend_metrics(
        users_path=args.users,
        transactions_path=args.transactions,
        output_dir=args.output_dir,
    )
    print(f"Saved metric CSV files to: {args.output_dir}")
    print(f"Saved chart PNG files to: {args.output_dir / 'charts'}")
    print(f"LLM trend metrics: {result.csv_paths['llm_trend_metrics']}")


if __name__ == "__main__":
    main()
