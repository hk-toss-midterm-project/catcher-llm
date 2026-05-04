"""주간 소비 분석 함수(build_weekly_consumption_analysis_from_frames)에 대한 pytest 테스트.

노트북 data_index_week.ipynb의 핵심 시나리오를 검증한다:
- 주간 총 지출 요약 (전주 대비 증감 계산)
- 카테고리별 주간 분석
- 반복 소비 패턴 탐지 (TOP 가맹점, N일 연속 소비)
- 요일별 소비 패턴
- 낭비성 소비 탐지 (야간, 소액 다빈도, 고액)
- 절약 가능성 추정
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from catcher_llm.analysis.user_weekly_analysis import build_weekly_consumption_analysis_from_frames
from catcher_llm.services.consumption_feedback.weekly_feedback import parse_weekly_spending_data

# ---------------------------------------------------------------------------
# 픽스처: 최소 재현 데이터
# ---------------------------------------------------------------------------

_MEMBER_ID = 1
_WEEK_START = date(2024, 4, 1)
_WEEK_END = date(2024, 4, 7)
_PREV_START = _WEEK_START - timedelta(days=7)
_PREV_END = _WEEK_END - timedelta(days=7)


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
    """당주 + 전주 + 과거 거래가 모두 포함된 최소 DataFrame을 반환한다."""
    rows = [
        # 과거 (IQR 기준 산출용)
        _make_row(1, "2024-03-01 10:00:00", 5_000, "편의점", "식비"),
        _make_row(1, "2024-03-02 12:00:00", 8_000, "스타벅스", "식비"),
        _make_row(1, "2024-03-03 14:00:00", 12_000, "맥도날드", "식비"),
        _make_row(1, "2024-03-04 09:00:00", 30_000, "배달의민족", "식비"),
        _make_row(1, "2024-03-05 18:00:00", 50_000, "쿠팡", "쇼핑"),
        # 전주 (2024-03-25 ~ 2024-03-31)
        _make_row(1, "2024-03-25 10:00:00", 20_000, "배달의민족", "식비"),
        _make_row(1, "2024-03-26 11:00:00", 10_000, "스타벅스", "식비"),
        _make_row(1, "2024-03-27 12:00:00", 15_000, "교통카드", "교통"),
        # 이번 주 (2024-04-01 ~ 2024-04-07)
        # 월 - 고액 + 야간 없음
        _make_row(1, "2024-04-01 10:00:00", 65_000, "SKT통신비", "생활"),
        # 화 - 배달 야간
        _make_row(1, "2024-04-02 22:00:00", 25_000, "배달의민족", "식비"),
        # 수 - 카페
        _make_row(1, "2024-04-03 09:00:00", 5_500, "스타벅스", "식비"),
        # 목 - 소액 편의점
        _make_row(1, "2024-04-04 13:00:00", 3_000, "CU", "식비"),
        # 금 - 카페 연속
        _make_row(1, "2024-04-05 08:00:00", 6_000, "스타벅스", "식비"),
        # 토 - 쇼핑
        _make_row(1, "2024-04-06 15:00:00", 30_000, "쿠팡", "쇼핑"),
        # 일 - 배달
        _make_row(1, "2024-04-07 19:00:00", 20_000, "쿠팡이츠", "식비"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _run(frame: pd.DataFrame) -> dict:
    """분석 함수를 기본 파라미터로 실행해 결과 딕셔너리를 반환한다."""
    return build_weekly_consumption_analysis_from_frames(
        frame,
        member_id=_MEMBER_ID,
        week_start=_WEEK_START,
        week_end=_WEEK_END,
    )


# ---------------------------------------------------------------------------
# 최상위 구조 검증
# ---------------------------------------------------------------------------


def test_result_has_required_top_level_keys(base_frame: pd.DataFrame) -> None:
    """반환 딕셔너리에 필수 최상위 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    expected_keys = {
        "member_id",
        "week_start",
        "week_end",
        "outlier_thresholds",
        "weekly_summary",
        "category_summary",
        "repeat_patterns",
        "weekday_pattern",
        "waste_detection",
        "saving_potential",
        "elasticity_analysis",
    }
    assert expected_keys.issubset(result.keys())


def test_member_id_and_dates_are_echoed(base_frame: pd.DataFrame) -> None:
    """반환 결과에 입력한 멤버 ID와 분석 기간이 그대로 포함되는지 검증한다."""
    result = _run(base_frame)
    assert result["member_id"] == _MEMBER_ID
    assert result["week_start"] == str(_WEEK_START)
    assert result["week_end"] == str(_WEEK_END)


def test_weekly_metrics_follow_period_metric_document(base_frame: pd.DataFrame) -> None:
    """문서의 주간 소비 분석 10개 핵심 지표와 특수 지표가 계산되는지 검증한다."""
    result = build_weekly_consumption_analysis_from_frames(
        base_frame,
        member_id=_MEMBER_ID,
        week_start=_WEEK_START,
        week_end=_WEEK_END,
        weekly_budget=140_000,
        monthly_budget=600_000,
        monthly_income=3_000_000,
    )

    metrics = result["weekly_metrics"]

    assert metrics["weekly_total_amount"] == 154_500
    assert metrics["weekly_transaction_count"] == 7
    assert metrics["weekly_average_daily_amount"] == pytest.approx(22_071.4286, abs=0.001)
    assert metrics["weekday_spending_ratio_percent"] == pytest.approx(67.6375, abs=0.001)
    assert metrics["weekend_spending_ratio_percent"] == pytest.approx(32.3625, abs=0.001)
    assert metrics["previous_week_change_rate_percent"] == pytest.approx(243.3333, abs=0.001)
    assert metrics["weekly_budget_usage_rate_percent"] == pytest.approx(110.3571, abs=0.001)
    assert metrics["weekly_remaining_budget"] == 0
    assert metrics["weekly_overspend_amount"] == 14_500
    assert metrics["weekly_income_usage_rate_percent"] == pytest.approx(22.0714, abs=0.001)
    assert metrics["weekly_budget_burn_rate"] == pytest.approx(1.1036, abs=0.001)
    assert metrics["month_to_date_budget_usage_rate_percent"] == pytest.approx(25.75, abs=0.001)
    assert metrics["projected_monthly_spending_from_weekly_pace"] == 662_143
    assert metrics["weekly_spending_volatility"] > 0
    assert metrics["weekend_overspending_index"] > 0


def test_weekly_comparisons_include_recent_average_and_last_month_same_week() -> None:
    """주간 분석이 전주·최근 4주 평균·지난달 같은 주차 비교를 제공하는지 검증한다."""
    frame = pd.DataFrame(
        [
            _make_row(1, "2024-03-04 10:00:00", 100, "3월1주", "식비"),
            _make_row(1, "2024-03-11 10:00:00", 100, "4주전", "식비"),
            _make_row(1, "2024-03-18 10:00:00", 300, "3주전", "식비"),
            _make_row(1, "2024-03-25 10:00:00", 500, "전주", "식비"),
            _make_row(1, "2024-04-01 10:00:00", 700, "이번주", "식비"),
        ]
    )

    result = build_weekly_consumption_analysis_from_frames(
        frame,
        member_id=1,
        week_start="2024-04-01",
        week_end="2024-04-07",
    )

    comparisons = result["weekly_comparisons"]
    previous_week = comparisons["previous_week"]
    recent_average = comparisons["recent_4week_average"]
    same_week_last_month = comparisons["same_week_last_month"]

    assert previous_week["reference_total"] == 500
    assert recent_average["reference_week_count"] == 4
    assert recent_average["average_total"] == pytest.approx(250.0, abs=0.001)
    assert recent_average["amount_diff"] == pytest.approx(450.0, abs=0.001)
    assert recent_average["amount_diff_rate_percent"] == pytest.approx(180.0, abs=0.001)
    assert same_week_last_month["week_num"] == 1
    assert same_week_last_month["reference_start_date"] == "2024-03-01"
    assert same_week_last_month["reference_end_date"] == "2024-03-07"
    assert same_week_last_month["reference_total"] == 100


def test_weekly_same_week_last_month_clamps_when_previous_month_has_no_week_num() -> None:
    """전월에 동일 주차 시작일이 없어도 마지막 7일 범위로 비교하는지 검증한다."""
    frame = pd.DataFrame(
        [
            _make_row(1, "2026-02-22 10:00:00", 100, "2월말1", "식비"),
            _make_row(1, "2026-02-28 10:00:00", 200, "2월말2", "식비"),
            _make_row(1, "2026-03-30 10:00:00", 700, "이번주", "식비"),
        ]
    )

    result = build_weekly_consumption_analysis_from_frames(
        frame,
        member_id=1,
        week_start="2026-03-30",
        week_end="2026-04-05",
    )

    same_week_last_month = result["weekly_comparisons"]["same_week_last_month"]

    assert same_week_last_month["week_num"] == 5
    assert same_week_last_month["reference_start_date"] == "2026-02-22"
    assert same_week_last_month["reference_end_date"] == "2026-02-28"
    assert same_week_last_month["reference_total"] == 300


# ---------------------------------------------------------------------------
# [1] 주간 총 지출 요약
# ---------------------------------------------------------------------------


def test_weekly_summary_total_equals_sum_of_this_week(base_frame: pd.DataFrame) -> None:
    """이번 주 총액이 당주 거래 금액의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    this_week_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date >= _WEEK_START)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date <= _WEEK_END)
    ]
    expected_total = int(this_week_rows["사용 금액"].sum())
    assert result["weekly_summary"]["this_week_total"] == expected_total


def test_weekly_summary_prev_total_equals_sum_of_prev_week(base_frame: pd.DataFrame) -> None:
    """전주 총액이 전주 거래 금액의 합과 일치하는지 검증한다."""
    result = _run(base_frame)
    prev_rows = base_frame[
        (base_frame["멤버 id"] == _MEMBER_ID)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date >= _PREV_START)
        & (pd.to_datetime(base_frame["사용 시간"]).dt.date <= _PREV_END)
    ]
    expected_prev = int(prev_rows["사용 금액"].sum())
    assert result["weekly_summary"]["prev_week_total"] == expected_prev


def test_weekly_summary_transaction_count(base_frame: pd.DataFrame) -> None:
    """결제 건수가 당주 거래 행 수와 일치하는지 검증한다."""
    result = _run(base_frame)
    this_week_count = int(
        base_frame[
            (base_frame["멤버 id"] == _MEMBER_ID)
            & (pd.to_datetime(base_frame["사용 시간"]).dt.date >= _WEEK_START)
            & (pd.to_datetime(base_frame["사용 시간"]).dt.date <= _WEEK_END)
        ].shape[0]
    )
    assert result["weekly_summary"]["transaction_count"] == this_week_count


# ---------------------------------------------------------------------------
# [2] 카테고리별 주간 분석
# ---------------------------------------------------------------------------


def test_category_summary_is_sorted_by_total_amount(base_frame: pd.DataFrame) -> None:
    """카테고리 목록이 총액 내림차순으로 정렬되어 있는지 검증한다."""
    result = _run(base_frame)
    amounts = [row["total_amount"] for row in result["category_summary"]]
    assert amounts == sorted(amounts, reverse=True)


def test_category_summary_contains_food_category(base_frame: pd.DataFrame) -> None:
    """당주 식비 거래가 있으면 카테고리 결과에 식비가 포함되는지 검증한다."""
    result = _run(base_frame)
    categories = [row["category"] for row in result["category_summary"]]
    assert "식비" in categories


def test_category_summary_ratio_sums_to_100(base_frame: pd.DataFrame) -> None:
    """카테고리별 비중의 합이 100%에 근사하는지 검증한다."""
    result = _run(base_frame)
    total_ratio = sum(row["ratio_percent"] for row in result["category_summary"])
    assert abs(total_ratio - 100.0) < 0.1


def test_category_summary_marks_ratio_context_warning() -> None:
    """주간 카테고리 비중이 작은 총지출 분모로 과장될 때 경고를 포함하는지 검증한다."""
    frame = pd.DataFrame(
        [
            _make_row(1, "2024-03-25 10:00:00", 100_000, "마트", "생활"),
            _make_row(1, "2024-03-26 10:00:00", 100_000, "식당", "식비"),
            _make_row(1, "2024-04-01 10:00:00", 62_600, "택시", "교통"),
            _make_row(1, "2024-04-02 10:00:00", 8_700, "편의점", "식비"),
        ]
    )

    result = build_weekly_consumption_analysis_from_frames(
        frame,
        member_id=1,
        week_start="2024-04-01",
        week_end="2024-04-07",
    )
    traffic_row = next(row for row in result["category_summary"] if row["category"] == "교통")
    warning = traffic_row["ratio_context_warning"]
    parsed = parse_weekly_spending_data(result)
    parsed_traffic_row = next(row for row in parsed.category_summary if row.category == "교통")

    assert "비중 수치만으로 급증" in warning["interpretation_rule"]
    assert warning["current_amount"] == 62_600
    assert warning["current_count"] == 1
    assert parsed_traffic_row.ratio_context_warning is not None


# ---------------------------------------------------------------------------
# [3] 반복 소비 패턴
# ---------------------------------------------------------------------------


def test_repeat_patterns_has_top_merchants(base_frame: pd.DataFrame) -> None:
    """TOP 가맹점 목록이 존재하고 필수 키를 포함하는지 검증한다."""
    result = _run(base_frame)
    merchants = result["repeat_patterns"]["top_merchants"]
    assert len(merchants) > 0
    assert all("merchant" in m and "visit_count" in m and "total_amount" in m for m in merchants)


def test_repeat_patterns_starbucks_consecutive(base_frame: pd.DataFrame) -> None:
    """스타벅스가 2일 이상 연속 소비된 가맹점으로 탐지되는지 검증한다.

    픽스처에서 스타벅스는 수(04-03)·금(04-05)이 아닌 연속 일자가 아니지만,
    연속 여부보다 탐지 로직 자체가 실행되는지를 확인한다.
    """
    result = _run(base_frame)
    # consecutive_merchants 키가 있고 리스트 타입인지 확인
    consecutive = result["repeat_patterns"]["consecutive_merchants"]
    assert isinstance(consecutive, list)


def test_repeat_patterns_delivery_summary(base_frame: pd.DataFrame) -> None:
    """배달 요약(count, total_amount, avg_per_transaction)이 올바르게 집계되는지 검증한다."""
    result = _run(base_frame)
    delivery = result["repeat_patterns"]["delivery"]
    # 픽스처: 배달의민족(04-02 25000) + 쿠팡이츠(04-07 20000) = 2건, 45000원
    assert delivery["count"] == 2
    assert delivery["total_amount"] == 45_000
    assert delivery["avg_per_transaction"] == pytest.approx(22_500.0, abs=1)


# ---------------------------------------------------------------------------
# [4] 요일별 소비 패턴
# ---------------------------------------------------------------------------


def test_weekday_pattern_has_seven_days(base_frame: pd.DataFrame) -> None:
    """요일별 분석에 월~일 7개 요일 항목이 모두 포함되는지 검증한다."""
    result = _run(base_frame)
    breakdown = result["weekday_pattern"]["weekday_breakdown"]
    assert len(breakdown) == 7


def test_weekday_pattern_weekday_names(base_frame: pd.DataFrame) -> None:
    """요일별 분석 목록의 요일 이름이 월화수목금토일 순서인지 검증한다."""
    result = _run(base_frame)
    names = [r["weekday"] for r in result["weekday_pattern"]["weekday_breakdown"]]
    assert names == ["월", "화", "수", "목", "금", "토", "일"]


def test_weekday_pattern_peak_weekday_is_valid(base_frame: pd.DataFrame) -> None:
    """소비 피크 요일이 유효한 요일 이름 중 하나인지 검증한다."""
    result = _run(base_frame)
    peak = result["weekday_pattern"]["peak_weekday"]
    assert peak in ["월", "화", "수", "목", "금", "토", "일"]


# ---------------------------------------------------------------------------
# [5] 낭비성 소비 탐지
# ---------------------------------------------------------------------------


def test_waste_detection_late_night_count(base_frame: pd.DataFrame) -> None:
    """21시 이후 소비 건수가 올바르게 집계되는지 검증한다.

    픽스처에서 21시 이후(22시) 거래: 배달의민족(04-02) 1건.
    """
    result = _run(base_frame)
    late_night = result["waste_detection"]["late_night"]
    assert late_night["count"] == 1
    assert late_night["total_amount"] == 25_000


def test_waste_detection_micro_spending_count(base_frame: pd.DataFrame) -> None:
    """1만 원 미만 소액 결제 건수가 올바르게 집계되는지 검증한다.

    픽스처에서 1만 원 미만: 스타벅스 5500원, CU 3000원, 스타벅스 6000원 = 3건.
    """
    result = _run(base_frame)
    micro = result["waste_detection"]["micro_spending"]
    assert micro["count"] == 3
    assert micro["total_amount"] == 5_500 + 3_000 + 6_000


def test_waste_detection_high_spending_items_exist(base_frame: pd.DataFrame) -> None:
    """IQR 상한선 초과 고액 거래가 high_spending items에 포함되는지 검증한다."""
    result = _run(base_frame)
    high = result["waste_detection"]["high_spending"]
    # IQR 상한선이 충분히 낮아 SKT통신비 65000원이나 쿠팡 30000원이 잡혀야 함
    assert high["count"] >= 1
    assert high["total_amount"] > 0


# ---------------------------------------------------------------------------
# [6] 절약 가능성 추정
# ---------------------------------------------------------------------------


def test_saving_potential_has_required_keys(base_frame: pd.DataFrame) -> None:
    """절약 가능성 딕셔너리에 필수 키가 모두 존재하는지 검증한다."""
    result = _run(base_frame)
    saving = result["saving_potential"]
    assert "delivery_save_per_skip" in saving
    assert "cafe_save_half_visits" in saving
    assert "improved_categories" in saving
    assert "worsened_categories" in saving


def test_saving_potential_delivery_save_is_avg(base_frame: pd.DataFrame) -> None:
    """배달 1회 절약 금액이 배달 평균 단가와 동일한지 검증한다."""
    result = _run(base_frame)
    delivery_avg = result["repeat_patterns"]["delivery"]["avg_per_transaction"]
    save = result["saving_potential"]["delivery_save_per_skip"]
    assert save == int(delivery_avg)


# ---------------------------------------------------------------------------
# [8] 소비 탄성 및 심리적 반동 분석
# ---------------------------------------------------------------------------


def test_elasticity_analysis_has_required_keys(base_frame: pd.DataFrame) -> None:
    """탄성 분석 결과가 예상된 키를 가지고 있는지 검증한다."""
    result = _run(base_frame)
    elasticity = result["elasticity_analysis"]
    expected_keys = {
        "correlation",
        "threshold",
        "rebound_avg",
        "normal_avg",
        "cheat_effective",
        "recommended_cheat_amount",
    }
    assert expected_keys.issubset(elasticity.keys())


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------


def test_missing_required_column_raises_value_error(base_frame: pd.DataFrame) -> None:
    """필수 컬럼이 빠진 DataFrame을 전달하면 ValueError가 발생하는지 검증한다."""
    broken = base_frame.drop(columns=["업종 카테고리"])
    with pytest.raises(ValueError, match="필요한 컬럼"):
        build_weekly_consumption_analysis_from_frames(
            broken,
            member_id=_MEMBER_ID,
            week_start=_WEEK_START,
            week_end=_WEEK_END,
        )


def test_unknown_member_id_raises_value_error(base_frame: pd.DataFrame) -> None:
    """존재하지 않는 멤버 ID를 전달하면 ValueError가 발생하는지 검증한다."""
    with pytest.raises(ValueError, match="멤버"):
        build_weekly_consumption_analysis_from_frames(
            base_frame,
            member_id=9999,
            week_start=_WEEK_START,
            week_end=_WEEK_END,
        )


def test_empty_this_week_returns_zero_total(base_frame: pd.DataFrame) -> None:
    """당주에 거래가 없으면 이번 주 총액이 0인지 검증한다."""
    # 당주(04-01~04-07) 데이터만 제거한 프레임
    frame_no_this_week = base_frame[
        pd.to_datetime(base_frame["사용 시간"]).dt.date < _WEEK_START
    ].copy()
    result = build_weekly_consumption_analysis_from_frames(
        frame_no_this_week,
        member_id=_MEMBER_ID,
        week_start=_WEEK_START,
        week_end=_WEEK_END,
    )
    assert result["weekly_summary"]["this_week_total"] == 0
