"""전체 사용자 소비 동향 지표 생성 스크립트의 핵심 계약을 검증한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_user_trend_metrics import build_trend_metric_tables, save_trend_outputs


def _make_users_frame() -> pd.DataFrame:
    """동향 지표 테스트에 필요한 최소 사용자 데이터를 만든다."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "name": "테스트유저1",
                "age": 29,
                "gender": "Female",
                "annual_income": 36_000_000,
                "region": "서울 서울-강남구",
                "personal_score": 70,
                "target_max_spending_amount": 200_000,
            },
            {
                "id": 2,
                "name": "테스트유저2",
                "age": 44,
                "gender": "Male",
                "annual_income": 48_000_000,
                "region": "부산 부산-해운대구",
                "personal_score": 62,
                "target_max_spending_amount": 300_000,
            },
        ]
    )


def _make_transactions_frame() -> pd.DataFrame:
    """월별·카테고리별 동향이 드러나는 최소 거래 데이터를 만든다."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "user_id": 1,
                "amount": 10_000,
                "transaction_time": "2026-01-05 08:00:00",
                "description": "카페 및 디저트 결제",
                "merchant_name": "스타벅스",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "식비",
                "payment_channel": "ONLINE",
            },
            {
                "id": 2,
                "user_id": 1,
                "amount": 20_000,
                "transaction_time": "2026-01-06 12:00:00",
                "description": "온라인 쇼핑 결제",
                "merchant_name": "쿠팡",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "쇼핑",
                "payment_channel": "OFFLINE",
            },
            {
                "id": 3,
                "user_id": 2,
                "amount": 15_000,
                "transaction_time": "2026-01-07 18:00:00",
                "description": "대중교통 결제",
                "merchant_name": "교통카드",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "교통",
                "payment_channel": "OFFLINE",
            },
            {
                "id": 4,
                "user_id": 1,
                "amount": 40_000,
                "transaction_time": "2026-02-05 21:30:00",
                "description": "외식 결제",
                "merchant_name": "식당",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "식비",
                "payment_channel": "ONLINE",
            },
            {
                "id": 5,
                "user_id": 1,
                "amount": 100_000,
                "transaction_time": "2026-02-10 10:00:00",
                "description": "보험료 정기 납부",
                "merchant_name": "보험사",
                "is_installment": True,
                "installment_months": 3,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "납부",
                "payment_channel": "OFFLINE",
            },
            {
                "id": 6,
                "user_id": 2,
                "amount": 30_000,
                "transaction_time": "2026-02-08 13:00:00",
                "description": "취소된 식비",
                "merchant_name": "식당",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "CANCELLED",
                "is_overseas": False,
                "category": "식비",
                "payment_channel": "ONLINE",
            },
            {
                "id": 7,
                "user_id": 2,
                "amount": 50_000,
                "transaction_time": "2026-03-01 15:00:00",
                "description": "외식 결제",
                "merchant_name": "식당",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": False,
                "category": "식비",
                "payment_channel": "ONLINE",
            },
            {
                "id": 8,
                "user_id": 2,
                "amount": 70_000,
                "transaction_time": "2026-03-02 22:30:00",
                "description": "해외 결제",
                "merchant_name": "해외가맹점",
                "is_installment": False,
                "installment_months": 0,
                "is_interest_free": False,
                "status": "APPROVED",
                "is_overseas": True,
                "category": "해외",
                "payment_channel": "ONLINE",
            },
        ]
    )


def test_build_trend_metric_tables_returns_monthly_and_llm_metrics() -> None:
    """월별 총액·전월 대비 증감률과 LLM용 long-form 지표가 생성되는지 검증한다."""
    tables = build_trend_metric_tables(
        users_frame=_make_users_frame(),
        transactions_frame=_make_transactions_frame(),
    )

    monthly_trends = tables["monthly_trends"]
    category_trends = tables["category_monthly_trends"]
    llm_metrics = tables["llm_trend_metrics"]

    february = monthly_trends[monthly_trends["month"] == "2026-02"].iloc[0]
    assert int(february["total_amount"]) == 140_000
    assert float(february["prev_month_change_rate_percent"]) == pytest.approx(
        211.1111,
        abs=0.001,
    )
    assert int(february["cancelled_transaction_count"]) == 1
    assert "식비" in set(category_trends["category"])
    assert "monthly_total_amount" in set(llm_metrics["metric_name"])


def test_build_trend_metric_tables_reuses_user_monthly_analysis_metrics() -> None:
    """사용자 월별 지표에 기존 월간 분석 모듈의 핵심 결과가 포함되는지 검증한다."""
    tables = build_trend_metric_tables(
        users_frame=_make_users_frame(),
        transactions_frame=_make_transactions_frame(),
    )
    user_monthly_metrics = tables["user_monthly_metrics"]

    user1_february = user_monthly_metrics[
        (user_monthly_metrics["user_id"] == 1) & (user_monthly_metrics["month"] == "2026-02")
    ].iloc[0]

    assert int(user1_february["monthly_total_amount"]) == 140_000
    assert float(user1_february["monthly_budget_usage_rate_percent"]) == pytest.approx(70.0)
    assert int(user1_february["installment_amount"]) == 100_000
    assert user1_february["source_module"] == "catcher_llm.analysis.user_monthly_analysis"


def test_segment_monthly_trends_budget_counts_each_user_once() -> None:
    """세그먼트 예산 합계가 거래 건수만큼 중복 합산되지 않는지 검증한다."""
    tables = build_trend_metric_tables(
        users_frame=_make_users_frame(),
        transactions_frame=_make_transactions_frame(),
    )
    segment_monthly_trends = tables["segment_monthly_trends"]

    female_under_30_january = segment_monthly_trends[
        (segment_monthly_trends["month"] == "2026-01")
        & (segment_monthly_trends["age_group"] == "20대 이하")
        & (segment_monthly_trends["gender"] == "Female")
    ].iloc[0]

    assert int(female_under_30_january["target_budget_total"]) == 200_000
    assert float(female_under_30_january["budget_usage_rate_percent"]) == pytest.approx(15.0)


def test_save_trend_outputs_writes_csv_and_png_files(tmp_path: Path) -> None:
    """지표 테이블 CSV와 matplotlib PNG 차트 파일이 출력되는지 검증한다."""
    tables = build_trend_metric_tables(
        users_frame=_make_users_frame(),
        transactions_frame=_make_transactions_frame(),
    )

    result = save_trend_outputs(tables=tables, output_dir=tmp_path)

    assert (tmp_path / "monthly_trends.csv").exists()
    assert (tmp_path / "llm_trend_metrics.csv").exists()
    assert (tmp_path / "charts" / "monthly_total_amount.png").exists()
    assert (tmp_path / "charts" / "category_monthly_amount.png").exists()
    assert result.csv_paths["monthly_trends"] == tmp_path / "monthly_trends.csv"
