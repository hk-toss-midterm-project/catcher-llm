from __future__ import annotations

from datetime import date
from pathlib import Path

from catcher_llm.db.models import SessionModel
from catcher_llm.services.consumption_feedback.session_report import (
    build_daily_report_result_from_session,
    build_monthly_report_result_from_session,
    build_weekly_report_result_from_session,
)


def test_report_session_adapter_restores_daily_feedback_payload() -> None:
    """일일 리포트가 저장된 daily 세션 본문과 근거를 서비스 결과처럼 복원하는지 검증한다."""
    session_row = SessionModel(
        user_id=7,
        analysis_date="2026-04-01",
        period_type="daily",
        analysis_result="{}",
        feedback_message="오늘은 배달 소비가 평소보다 높았습니다.",
        feedback_reason=(
            '[{"evidence_type":"spending_metric","title":"배달 증가",'
            '"detail":"배달 지출이 증가했습니다."}]'
        ),
        todo_tomorrow="내일은 배달앱을 한 번 쉬세요.",
    )

    result = build_daily_report_result_from_session(session_row)

    assert result.member_id == 7
    assert result.analysis_date == "2026-04-01"
    assert result.feedback is not None
    assert result.feedback.scolding_message.startswith("오늘은 배달 소비")
    assert result.feedback.tomorrow_mission == "내일은 배달앱을 한 번 쉬세요."
    assert result.feedback.key_evidences[0].title == "배달 증가"
    assert result.daily_analysis is None


def test_report_session_adapter_restores_weekly_and_monthly_feedback_payloads() -> None:
    """주간·월간 리포트가 저장된 세션 본문과 미션을 각 기간 결과 모델로 복원하는지 검증한다."""
    weekly_session = SessionModel(
        user_id=8,
        analysis_date="2026-03-29",
        period_type="weekly",
        feedback_message="이번 주 카페 소비가 반복됐습니다.",
        feedback_reason="[]",
        todo_tomorrow="다음 주는 텀블러를 챙기세요.",
    )
    monthly_session = SessionModel(
        user_id=9,
        analysis_date="2026-04",
        period_type="monthly",
        feedback_message="이번 달 쇼핑 비중이 높았습니다.",
        feedback_reason="[]",
        todo_tomorrow="다음 달 쇼핑 예산을 먼저 잠그세요.",
    )

    weekly_result = build_weekly_report_result_from_session(
        weekly_session,
        week_end=date(2026, 4, 4),
    )
    monthly_result = build_monthly_report_result_from_session(monthly_session)

    assert weekly_result.week_start == "2026-03-29"
    assert weekly_result.week_end == "2026-04-04"
    assert weekly_result.feedback is not None
    assert weekly_result.feedback.next_week_mission == "다음 주는 텀블러를 챙기세요."

    assert monthly_result.analysis_month == "2026-04"
    assert monthly_result.feedback is not None
    assert monthly_result.feedback.next_month_mission == "다음 달 쇼핑 예산을 먼저 잠그세요."


def test_report_pages_load_cached_session_before_generating_feedback() -> None:
    """일·주·월 리포트 페이지가 피드백 페이지처럼 저장된 세션을 우선 재사용하는지 검증한다."""
    page_expectations = {
        "pages/04_daily_report.py": [
            "has_stored_feedback_payload",
            "load_daily_session_for_date",
            "build_daily_report_result_from_session",
            "일일 리포트 재생성",
        ],
        "pages/05_weekly_report.py": [
            "has_stored_feedback_payload",
            "load_weekly_session_for_date",
            "build_weekly_report_result_from_session",
            "주간 리포트 재생성",
        ],
        "pages/06_monthly_report.py": [
            "has_stored_feedback_payload",
            "load_monthly_session_for_date",
            "build_monthly_report_result_from_session",
            "월간 리포트 재생성",
        ],
    }

    for page_path, snippets in page_expectations.items():
        page_source = Path(page_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in page_source
