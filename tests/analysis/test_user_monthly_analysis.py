"""월간 소비 분석 함수(build_monthly_consumption_analysis_from_frames)에 대한 pytest 테스트.

노트북 data_index_month.ipynb의 핵심 시나리오를 검증한다:
- 월간 총 지출 요약 (전월 대비 증감 계산)
- 고정비 vs 변동비 분석
- 카테고리 심층 분석 (유형 분류, 절약 가능성 TOP)
- 반복 소비 누적 (배달/카페/편의점/택시)
- 주차별 소비 추이
- 소액 다빈도 누적 분석
- 야간 소비 월간 분석
- 이상 지출 월간 누적 집계
- 절약 가능 금액 추정
"""

from __future__ import annotations

import pandas as pd
import pytest

from catcher_llm.analysis.user_monthly_analysis import (
    build_monthly_consumption_analysis_from_frames,
)

# ---------------------------------------------------------------------------
# 픽스처: 최소 재현 데이터
# ---------------------------------------------------------------------------

_MEMBER_ID = 1
_ANALYSIS_MONTH = "2024-04"
_PREV_MONTH = "2024-03"


def _make_row(
    member_id: int,
    used_at: str,
    amount: int,
    merchant: str,
    category: str,
    payment_channel: str = "카드",
) -> dict:
    """테스트용 거래 행을 생성한다."""
    return {
        "멤버 id": member_id,
        "id": 0,
        "사용 금액": amount,
        "사용 시간": used_at,
        "결제 내역": merchant,
        "업종 카테고리": category,
        "결제 방식 (온/오프라인)": payment_channel,
    }


@pytest.fixture()
def base_frame() -> pd.DataFrame:
    """당월(2024-04) + 전월(2024-03) + 과거 거래가 포함된 최소 DataFrame을 반환한다."""
    rows = [
        # 과거 (IQR 기준 산출용, 2024-02)
        _make_row(1, "2024-02-01 10:00:00", 5_000, "편의점", "식비"),
        _make_row(1, "2024-02-05 12:00:00", 8_000, "스타벅스", "식비"),
        _make_row(1, "2024-02-10 14:00:00", 12_000, "맥도날드", "식비"),
        _make_row(1, "2024-02-15 09:00:00", 30_000, "배달의민족", "식비"),
        _make_row(1, "2024-02-20 18:00:00", 50_000, "쿠팡", "쇼핑"),
        # 전월 (2024-03)
        _make_row(1, "2024-03-05 10:00:00", 20_000, "배달의민족", "식비"),
        _make_row(1, "2024-03-10 11:00:00", 10_000, "스타벅스", "식비"),
        _make_row(1, "2024-03-15 12:00:00", 15_000, "교통카드", "교통"),
        _make_row(1, "2024-03-20 09:00:00", 65_000, "SKT통신비", "생활", "자동이체"),
        # 당월 (2024-04)
        # 1주차
        _make_row(1, "2024-04-01 10:00:00", 65_000, "SKT통신비", "생활", "자동이체"),
        _make_row(1, "2024-04-02 22:00:00", 25_000, "배달의민족", "식비"),
        _make_row(1, "2024-04-03 09:00:00", 5_500, "스타벅스", "식비"),
        _make_row(1, "2024-04-04 13:00:00", 3_000, "CU", "식비"),
        _make_row(1, "2024-04-05 08:00:00", 6_000, "스타벅스", "식비"),
        _make_row(1, "2024-04-06 15:00:00", 30_000, "쿠팡", "쇼핑"),
        _make_row(1, "2024-04-07 19:00:00", 20_000, "쿠팡이츠", "식비"),
        # 2주차
        _make_row(1, "2024-04-08 12:00:00", 8_000, "이디야", "식비"),
        _make_row(1, "2024-04-09 21:00:00", 22_000, "배달의민족", "식비"),
        _make_row(1, "2024-04-10 14:00:00", 15_000, "카카오택시", "교통"),
        _make_row(1, "2024-04-11 09:00:00", 7_000, "GS25", "식비"),
        _make_row(1, "2024-04-12 18:00:00", 40_000, "병원", "의료"),
        _make_row(1, "2024-04-13 10:00:00", 9_000, "스타벅스", "식비"),
        _make_row(1, "2024-04-14 20:00:00", 18_000, "쿠팡이츠", "식비"),
        # 3주차
        _make_row(1, "2024-04-15 11:00:00", 6_500, "스타벅스", "식비"),
        _make_row(1, "2024-04-17 22:00:00", 23_000, "배달의민족", "식비"),
        _make_row(1, "2024-04-19 15:00:00", 35_000, "쿠팡", "쇼핑"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _run(frame: pd.DataFrame) -> dict:
    """분석 함수를 기본 파라미터로 실행해 결과 딕셔너리를 반환한다."""
    return build_monthly_consumption_analysis_from_frames(
        frame,
        member_id=_MEMBER_ID,
        analysis_month=_ANALYSIS_MONTH,
    )


# ---------------------------------------------------------------------------
# 최상위 구조 검증
# ---------------------------------------------------------------------------


def test_result_has_required_top_level_keys(base_frame: pd.DataFrame) -> None:
    """반환 딕셔너리에 필수 최상위 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    expected_keys = {
        "member_id",
        "analysis_month",
        "prev_month",
        "outlier_thresholds",
        "monthly_summary",
        "fixed_variable",
        "category_deep",
        "top_savable_categories",
        "repeat_patterns",
        "weekly_trend",
        "micro_spending",
        "late_night_spending",
        "high_spending",
        "saving_potential",
        "cash_flow_volatility",
        "spending_concentration",
        "frictionless_and_density",
        "installment_debt_pressure",
    }
    assert expected_keys.issubset(result.keys())


def test_member_id_and_months_are_echoed(base_frame: pd.DataFrame) -> None:
    """반환 결과에 입력한 멤버 ID와 분석 월·전월이 올바르게 포함되는지 검증한다."""
    result = _run(base_frame)
    assert result["member_id"] == _MEMBER_ID
    assert result["analysis_month"] == _ANALYSIS_MONTH
    assert result["prev_month"] == _PREV_MONTH


def test_monthly_metrics_follow_period_metric_document(base_frame: pd.DataFrame) -> None:
    """문서의 월간 소비 분석 10개 핵심 지표와 특수 지표가 계산되는지 검증한다."""
    result = build_monthly_consumption_analysis_from_frames(
        base_frame,
        member_id=_MEMBER_ID,
        analysis_month=_ANALYSIS_MONTH,
        monthly_budget=400_000,
        monthly_income=1_000_000,
        salary_day=1,
    )

    metrics = result["monthly_metrics"]

    assert metrics["monthly_total_amount"] == 338_000
    assert metrics["monthly_budget_usage_rate_percent"] == pytest.approx(84.5, abs=0.001)
    assert metrics["monthly_remaining_budget"] == 62_000
    assert metrics["monthly_overspend_amount"] == 0
    assert metrics["monthly_income_usage_rate_percent"] == pytest.approx(33.8, abs=0.001)
    assert metrics["target_spending_to_income_rate_percent"] == pytest.approx(40.0, abs=0.001)
    assert metrics["estimated_saving_amount"] == 662_000
    assert metrics["estimated_saving_rate_percent"] == pytest.approx(66.2, abs=0.001)
    assert metrics["target_saving_amount"] == 600_000
    assert metrics["target_saving_rate_percent"] == pytest.approx(60.0, abs=0.001)
    assert metrics["previous_month_change_rate_percent"] == pytest.approx(207.2727, abs=0.001)
    assert metrics["fixed_cost_amount"] == 65_000
    assert metrics["fixed_cost_ratio_percent"] == pytest.approx(19.2308, abs=0.001)
    assert metrics["variable_cost_amount"] == 273_000
    assert metrics["subscription_total"] == 0
    assert metrics["fixed_cost_burden_rate_percent"] == pytest.approx(6.5, abs=0.001)
    assert metrics["spending_capacity"] == 880_000
    assert metrics["nonessential_spending_income_rate_percent"] == pytest.approx(21.8, abs=0.001)
    assert metrics["post_salary_spending_increase_rate_percent"] is not None
    assert metrics["month_end_pressure_index"] is not None


def test_monthly_comparisons_include_recent_three_month_average() -> None:
    """월간 분석이 전월과 최근 3개월 평균 비교를 함께 제공하는지 검증한다."""
    frame = pd.DataFrame(
        [
            _make_row(1, "2024-01-10 10:00:00", 50_000, "1월", "식비"),
            _make_row(1, "2024-02-10 10:00:00", 100_000, "2월", "식비"),
            _make_row(1, "2024-03-10 10:00:00", 150_000, "3월", "식비"),
            _make_row(1, "2024-04-10 10:00:00", 300_000, "4월", "식비"),
        ]
    )

    result = build_monthly_consumption_analysis_from_frames(
        frame,
        member_id=1,
        analysis_month="2024-04",
    )

    comparisons = result["monthly_comparisons"]
    previous_month = comparisons["previous_month"]
    recent_average = comparisons["recent_3month_average"]

    assert previous_month["reference_month"] == "2024-03"
    assert previous_month["reference_total"] == 150_000
    assert recent_average["reference_months"] == ["2024-03", "2024-02", "2024-01"]
    assert recent_average["reference_month_count"] == 3
    assert recent_average["average_total"] == pytest.approx(100_000.0, abs=0.001)
    assert recent_average["amount_diff"] == pytest.approx(200_000.0, abs=0.001)
    assert recent_average["amount_diff_rate_percent"] == pytest.approx(200.0, abs=0.001)


# ---------------------------------------------------------------------------
# [1] 월간 총 지출 요약
# ---------------------------------------------------------------------------


def test_monthly_summary_total_equals_sum_of_this_month(base_frame: pd.DataFrame) -> None:
    """이번 달 총액이 당월 거래 금액의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    this_month_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.to_period("M").astype(str) == _ANALYSIS_MONTH)
    ]
    expected_total = int(this_month_rows["사용 금액"].sum())
    assert result["monthly_summary"]["this_month_total"] == expected_total


def test_monthly_summary_prev_total_equals_sum_of_prev_month(base_frame: pd.DataFrame) -> None:
    """전월 총액이 전월 거래 금액의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    prev_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.to_period("M").astype(str) == _PREV_MONTH)
    ]
    expected_prev = int(prev_rows["사용 금액"].sum())
    assert result["monthly_summary"]["prev_month_total"] == expected_prev


# ---------------------------------------------------------------------------
# [2] 고정비 vs 변동비 분석
# ---------------------------------------------------------------------------


def test_fixed_variable_fixed_total_matches_auto_transfer(base_frame: pd.DataFrame) -> None:
    """고정비 총액이 '자동이체' 결제 방식 거래의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    this_month_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.to_period("M").astype(str) == _ANALYSIS_MONTH)
    ]
    auto_total = int(
        this_month_rows[this_month_rows["결제 방식 (온/오프라인)"] == "자동이체"]["사용 금액"].sum()
    )
    assert result["fixed_variable"]["fixed_total"] == auto_total


def test_fixed_variable_ratios_sum_to_100(base_frame: pd.DataFrame) -> None:
    """고정비 비중과 변동비 비중의 합이 100%에 근사하는지 검증한다."""
    result = _run(base_frame)
    total_ratio = (
        result["fixed_variable"]["fixed_ratio_percent"]
        + result["fixed_variable"]["variable_ratio_percent"]
    )
    assert abs(total_ratio - 100.0) < 0.01


def test_fixed_variable_fixed_items_contains_skt(base_frame: pd.DataFrame) -> None:
    """고정비 항목에 SKT통신비가 포함되는지 검증한다."""
    result = _run(base_frame)
    merchant_names = [item["merchant"] for item in result["fixed_variable"]["fixed_items"]]
    assert "SKT통신비" in merchant_names


# ---------------------------------------------------------------------------
# [3] 카테고리 심층 분석
# ---------------------------------------------------------------------------


def test_category_deep_sorted_by_total_amount(base_frame: pd.DataFrame) -> None:
    """카테고리 목록이 총액 내림차순으로 정렬되어 있는지 검증한다."""
    result = _run(base_frame)
    amounts = [row["total_amount"] for row in result["category_deep"]]
    assert amounts == sorted(amounts, reverse=True)


def test_category_deep_type_classification(base_frame: pd.DataFrame) -> None:
    """식비·쇼핑은 '낭비성', 교통·의료·생활은 '필수'로 분류되는지 검증한다."""
    result = _run(base_frame)
    cat_map = {row["category"]: row["type"] for row in result["category_deep"]}
    if "식비" in cat_map:
        assert cat_map["식비"] == "낭비성"
    if "쇼핑" in cat_map:
        assert cat_map["쇼핑"] == "낭비성"
    if "교통" in cat_map:
        assert cat_map["교통"] == "필수"
    if "의료" in cat_map:
        assert cat_map["의료"] == "필수"
    if "생활" in cat_map:
        assert cat_map["생활"] == "필수"


def test_top_savable_categories_are_waste_type(base_frame: pd.DataFrame) -> None:
    """절약 가능성 TOP 카테고리가 모두 '낭비성' 유형인지 검증한다."""
    result = _run(base_frame)
    for row in result["top_savable_categories"]:
        assert row["type"] == "낭비성"


# ---------------------------------------------------------------------------
# [4] 반복 소비 누적 분석
# ---------------------------------------------------------------------------


def test_repeat_patterns_delivery_count(base_frame: pd.DataFrame) -> None:
    """배달 횟수가 올바르게 집계되는지 검증한다.

    픽스처 당월 배달의민족: 04-02, 04-09, 04-17 = 3건.
    쿠팡이츠: 04-07, 04-14 = 2건. 합계 5건.
    """
    result = _run(base_frame)
    delivery = result["repeat_patterns"]["delivery"]
    assert delivery["count"] == 5


def test_repeat_patterns_top5_merchants_count(base_frame: pd.DataFrame) -> None:
    """TOP 5 가맹점 목록의 항목 수가 5개 이하인지 검증한다."""
    result = _run(base_frame)
    top5 = result["repeat_patterns"]["top5_merchants"]
    assert len(top5) <= 5


# ---------------------------------------------------------------------------
# [5] 주차별 소비 추이
# ---------------------------------------------------------------------------


def test_weekly_trend_has_breakdown(base_frame: pd.DataFrame) -> None:
    """주차별 소비 추이 목록이 비어 있지 않고 필수 키를 포함하는지 검증한다."""
    result = _run(base_frame)
    breakdown = result["weekly_trend"]["weekly_breakdown"]
    assert len(breakdown) >= 1
    assert all(
        {"week_num", "start_date", "end_date", "total_amount", "count"}.issubset(row.keys())
        for row in breakdown
    )


def test_weekly_trend_direction_is_valid(base_frame: pd.DataFrame) -> None:
    """소비 추이 방향이 유효한 값(increasing/decreasing/stable) 중 하나인지 검증한다."""
    result = _run(base_frame)
    assert result["weekly_trend"]["trend_direction"] in {"increasing", "decreasing", "stable"}


def test_weekly_trend_week1_total_matches_first_week_sum(base_frame: pd.DataFrame) -> None:
    """1주차 총액이 04-01~04-07 거래 금액의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    week1_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date >= pd.Timestamp("2024-04-01").date())
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date <= pd.Timestamp("2024-04-07").date())
    ]
    expected = int(week1_rows["사용 금액"].sum())
    week1 = next(r for r in result["weekly_trend"]["weekly_breakdown"] if r["week_num"] == 1)
    assert week1["total_amount"] == expected


# ---------------------------------------------------------------------------
# [6] 소액 다빈도 누적 분석
# ---------------------------------------------------------------------------


def test_micro_spending_count_and_total(base_frame: pd.DataFrame) -> None:
    """1만 원 미만 소액 결제 건수와 총액이 올바르게 집계되는지 검증한다."""
    result = _run(base_frame)
    this_month_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.to_period("M").astype(str) == _ANALYSIS_MONTH)
    ]
    micro_rows = this_month_rows[this_month_rows["사용 금액"] < 10_000]
    assert result["micro_spending"]["count"] == len(micro_rows)
    assert result["micro_spending"]["total_amount"] == int(micro_rows["사용 금액"].sum())


# ---------------------------------------------------------------------------
# [7] 야간 소비 월간 분석
# ---------------------------------------------------------------------------


def test_late_night_count(base_frame: pd.DataFrame) -> None:
    """21시 이후 소비 건수가 올바르게 집계되는지 검증한다.

    픽스처 당월 21시 이후: 04-02 22시(배달의민족), 04-09 21시(배달의민족), 04-17 22시(배달의민족) = 3건.
    """
    result = _run(base_frame)
    assert result["late_night_spending"]["count"] == 3


# ---------------------------------------------------------------------------
# [8] 이상 지출 월간 누적 집계
# ---------------------------------------------------------------------------


def test_high_spending_items_have_required_keys(base_frame: pd.DataFrame) -> None:
    """이상 지출 항목에 필수 키(used_at, merchant, amount, category)가 있는지 검증한다."""
    result = _run(base_frame)
    for item in result["high_spending"]["items"]:
        assert "used_at" in item
        assert "merchant" in item
        assert "amount" in item
        assert "category" in item


def test_high_spending_total_equals_sum_of_items(base_frame: pd.DataFrame) -> None:
    """이상 지출 총액이 items의 amount 합계와 일치하는지 검증한다."""
    result = _run(base_frame)
    items_total = sum(item["amount"] for item in result["high_spending"]["items"])
    assert result["high_spending"]["total_amount"] == items_total


# ---------------------------------------------------------------------------
# [9] 절약 가능 금액 추정
# ---------------------------------------------------------------------------


def test_saving_potential_has_required_keys(base_frame: pd.DataFrame) -> None:
    """절약 가능성 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    saving = result["saving_potential"]
    required = {
        "delivery_reduce_30pct",
        "cafe_every_other_day",
        "taxi_to_transit",
        "micro_reduce_20pct",
        "total_potential_saving",
        "next_month_recommended_target",
    }
    assert required.issubset(saving.keys())


def test_saving_potential_total_is_sum_of_components(base_frame: pd.DataFrame) -> None:
    """총 절약 가능액이 각 절약 항목의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    s = result["saving_potential"]
    expected = (
        s["delivery_reduce_30pct"]
        + s["cafe_every_other_day"]
        + s["taxi_to_transit"]
        + s["micro_reduce_20pct"]
    )
    assert s["total_potential_saving"] == expected


def test_saving_potential_next_month_target_is_90pct(base_frame: pd.DataFrame) -> None:
    """다음 달 권장 목표액이 이번 달 총액의 90%인지 검증한다."""
    result = _run(base_frame)
    this_total = result["monthly_summary"]["this_month_total"]
    expected_target = int(this_total * 0.90)
    assert result["saving_potential"]["next_month_recommended_target"] == expected_target


# ---------------------------------------------------------------------------
# [10] 현금 흐름 변동성 지수
# ---------------------------------------------------------------------------


def test_cash_flow_volatility_has_required_keys(base_frame: pd.DataFrame) -> None:
    """현금 흐름 변동성 지수 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    cfv = result["cash_flow_volatility"]
    required = {
        "mean_weekly",
        "std_weekly",
        "cv_index",
        "pace_status",
        "weekly_ratios",
    }
    assert required.issubset(cfv.keys())


# ---------------------------------------------------------------------------
# [11] 파레토 지출 쏠림 지수
# ---------------------------------------------------------------------------


def test_spending_concentration_has_required_keys(base_frame: pd.DataFrame) -> None:
    """파레토 지출 쏠림 지수 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    sc = result["spending_concentration"]
    required = {
        "total_variable_amount",
        "top_1_category",
        "top_1_amount",
        "top_1_ratio_percent",
        "top_2_category",
        "top_2_amount",
        "top_2_ratio_percent",
        "top_2_combined_ratio_percent",
        "concentration_status",
    }
    assert required.issubset(sc.keys())


# ---------------------------------------------------------------------------
# [12] 월간 지출 마찰력 및 밀도 분석
# ---------------------------------------------------------------------------


def test_frictionless_and_density_has_required_keys(base_frame: pd.DataFrame) -> None:
    """지출 마찰력 및 밀도 분석 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    fad = result["frictionless_and_density"]
    assert "frictionless_spending" in fad
    assert "transaction_density" in fad


# ---------------------------------------------------------------------------
# [13] 할부 부채 압박 지수
# ---------------------------------------------------------------------------


def test_installment_debt_pressure_has_required_keys(base_frame: pd.DataFrame) -> None:
    """할부 부채 압박 지수 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    idp = result["installment_debt_pressure"]
    required = {
        "total_installment_amount",
        "installment_count",
        "installment_ratio_percent",
        "avg_installment_months",
        "max_installment_months",
        "items",
    }
    assert required.issubset(idp.keys())


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------


def test_missing_required_column_raises_value_error(base_frame: pd.DataFrame) -> None:
    """필수 컬럼이 빠진 DataFrame을 전달하면 ValueError가 발생하는지 검증한다."""
    broken = base_frame.drop(columns=["결제 방식 (온/오프라인)"])
    with pytest.raises(ValueError, match="필요한 컬럼"):
        build_monthly_consumption_analysis_from_frames(
            broken,
            member_id=_MEMBER_ID,
            analysis_month=_ANALYSIS_MONTH,
        )


def test_unknown_member_id_raises_value_error(base_frame: pd.DataFrame) -> None:
    """존재하지 않는 멤버 ID를 전달하면 ValueError가 발생하는지 검증한다."""
    with pytest.raises(ValueError, match="멤버"):
        build_monthly_consumption_analysis_from_frames(
            base_frame,
            member_id=9999,
            analysis_month=_ANALYSIS_MONTH,
        )


def test_no_this_month_transactions_returns_zero_total(base_frame: pd.DataFrame) -> None:
    """당월 거래가 없으면 이번 달 총액이 0인지 검증한다."""
    frame_no_this = base_frame[
        pd.to_datetime(base_frame["사용 시간"]).dt.to_period("M").astype(str) != _ANALYSIS_MONTH
    ].copy()
    result = build_monthly_consumption_analysis_from_frames(
        frame_no_this,
        member_id=_MEMBER_ID,
        analysis_month=_ANALYSIS_MONTH,
    )
    assert result["monthly_summary"]["this_month_total"] == 0
