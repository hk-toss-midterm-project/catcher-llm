"""LangChain 기반 사용자 동향 보고서 생성 스크립트의 동작을 검증한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from scripts.generate_user_trend_report import (
    build_metric_context,
    generate_user_trend_report_from_metrics,
    resolve_metrics_dir,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """테스트 지표 CSV를 UTF-8 BOM 포함 형식으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_metric_dir(root: Path, dirname: str = "period_2026-01-01_to_2026-02-28") -> Path:
    """보고서 생성 테스트에 필요한 최소 지표 파일 묶음을 만든다."""
    metrics_dir = root / dirname
    _write_csv(
        metrics_dir / "monthly_trends.csv",
        [
            {
                "month": "2026-01",
                "active_user_count": 2,
                "approved_transaction_count": 3,
                "total_amount": 100_000,
                "average_amount_per_active_user": 50_000,
                "budget_usage_rate_percent": 100.0,
                "online_ratio_percent": 40.0,
                "installment_ratio_percent": 10.0,
                "late_night_ratio_percent": 5.0,
                "top_category": "쇼핑",
                "top_category_amount": 60_000,
                "top_category_ratio_percent": 60.0,
                "prev_month_change_rate_percent": None,
            },
            {
                "month": "2026-02",
                "active_user_count": 2,
                "approved_transaction_count": 4,
                "total_amount": 150_000,
                "average_amount_per_active_user": 75_000,
                "budget_usage_rate_percent": 150.0,
                "online_ratio_percent": 50.0,
                "installment_ratio_percent": 20.0,
                "late_night_ratio_percent": 8.0,
                "top_category": "납부",
                "top_category_amount": 90_000,
                "top_category_ratio_percent": 60.0,
                "prev_month_change_rate_percent": 50.0,
            },
        ],
    )
    _write_csv(
        metrics_dir / "category_monthly_trends.csv",
        [
            {
                "month": "2026-01",
                "category": "쇼핑",
                "total_amount": 60_000,
                "transaction_count": 2,
                "active_user_count": 2,
                "monthly_ratio_percent": 60.0,
            },
            {
                "month": "2026-02",
                "category": "납부",
                "total_amount": 90_000,
                "transaction_count": 2,
                "active_user_count": 2,
                "monthly_ratio_percent": 60.0,
            },
        ],
    )
    _write_csv(
        metrics_dir / "segment_monthly_trends.csv",
        [
            {
                "month": "2026-02",
                "age_group": "30대",
                "gender": "Female",
                "total_amount": 150_000,
                "active_user_count": 2,
                "transaction_count": 4,
                "average_amount_per_active_user": 75_000,
                "budget_usage_rate_percent": 150.0,
            }
        ],
    )
    _write_csv(
        metrics_dir / "user_monthly_metrics.csv",
        [
            {
                "user_id": 1,
                "month": "2026-02",
                "monthly_total_amount": 120_000,
                "monthly_budget_usage_rate_percent": 240.0,
                "top_category": "납부",
                "installment_ratio_percent": 35.0,
                "frictionless_spending_ratio_percent": 70.0,
            },
            {
                "user_id": 2,
                "month": "2026-02",
                "monthly_total_amount": 30_000,
                "monthly_budget_usage_rate_percent": 60.0,
                "top_category": "식비",
                "installment_ratio_percent": 0.0,
                "frictionless_spending_ratio_percent": 20.0,
            },
        ],
    )
    _write_csv(
        metrics_dir / "weekly_trends.csv",
        [
            {
                "week_start": "2026-02-02",
                "week_end": "2026-02-08",
                "total_amount": 90_000,
                "weekend_spending_ratio_percent": 30.0,
            }
        ],
    )
    _write_csv(
        metrics_dir / "daily_trends.csv",
        [
            {
                "date": "2026-02-05",
                "total_amount": 80_000,
                "online_ratio_percent": 70.0,
            }
        ],
    )
    _write_csv(
        metrics_dir / "llm_trend_metrics.csv",
        [
            {
                "period_type": "month",
                "period": "2026-02",
                "metric_name": "monthly_total_amount",
                "metric_value": 150_000,
                "metric_unit": "KRW",
                "trend_label": "상승",
                "context": "월간 승인 소비 총액",
            }
        ],
    )
    (metrics_dir / "charts").mkdir(parents=True, exist_ok=True)
    (metrics_dir / "charts" / "monthly_total_amount.png").write_bytes(b"png")
    return metrics_dir


def test_resolve_metrics_dir_uses_latest_period_directory(tmp_path: Path) -> None:
    """명시 경로가 없을 때 최신 기간 지표 디렉터리를 선택하는지 검증한다."""
    _write_metric_dir(tmp_path, "period_2026-01-01_to_2026-01-31")
    latest_dir = _write_metric_dir(tmp_path, "period_2026-02-01_to_2026-02-28")

    assert resolve_metrics_dir(metrics_root=tmp_path) == latest_dir


def test_build_metric_context_contains_key_sections(tmp_path: Path) -> None:
    """지표 CSV 묶음이 LLM 입력용 핵심 섹션 문자열로 정리되는지 검증한다."""
    metrics_dir = _write_metric_dir(tmp_path)

    context = build_metric_context(metrics_dir)

    assert "월별 핵심 지표" in context
    assert "2026-02" in context
    assert "카테고리 합산" in context
    assert "목표 사용률 200% 이상" in context
    assert "주간 피크" in context


def test_generate_user_trend_report_uses_langchain_and_saves_markdown(tmp_path: Path) -> None:
    """Fake ChatModel로 LangChain 체인이 실행되고 보고서 Markdown이 저장되는지 검증한다."""
    metrics_dir = _write_metric_dir(tmp_path / "metrics")
    fake_llm = FakeListChatModel(responses=["# 사용자 소비 동향 보고서\n\nLLM 생성 보고서"])

    result = generate_user_trend_report_from_metrics(
        metrics_dir=metrics_dir,
        output_root=tmp_path / "reports",
        llm=fake_llm,
    )

    assert result.output_path == tmp_path / "reports" / metrics_dir.name / "users_trend_report.md"
    assert result.output_path.exists()
    assert result.output_path.read_text(encoding="utf-8").startswith("# 사용자 소비 동향 보고서")
    assert result.metrics_dir == metrics_dir
