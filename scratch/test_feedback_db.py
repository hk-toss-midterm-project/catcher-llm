from __future__ import annotations

from sqlalchemy import select

from catcher_llm.db.models import SessionModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.consumption_feedback.monthly_feedback import generate_monthly_feedback
from catcher_llm.services.consumption_feedback.weekly_feedback import generate_weekly_feedback


def test_feedback_saving():
    """주간 및 월간 피드백이 session 테이블에 저장되는지 확인한다."""
    member_id = 1

    # 1. 주간 피드백 테스트
    week_start = "2024-04-01"
    week_end = "2024-04-07"
    print(f"\nTesting Weekly Feedback for {week_start}...")
    weekly_result = generate_weekly_feedback(
        member_id=member_id, week_start=week_start, week_end=week_end
    )
    if weekly_result.error:
        print(f"Weekly Error: {weekly_result.error}")
    assert weekly_result.error is None

    # 2. 월간 피드백 테스트
    analysis_month = "2024-04"
    print(f"Testing Monthly Feedback for {analysis_month}...")
    monthly_result = generate_monthly_feedback(member_id=member_id, analysis_month=analysis_month)
    assert monthly_result.error is None

    # 3. DB 확인
    with session_scope() as session:
        # 주간 세션 확인
        weekly_session = session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == week_start,
                SessionModel.period_type == "weekly",
            )
        )
        assert weekly_session is not None
        assert weekly_session.weekly_analysis_result is not None
        print(f"Successfully saved weekly session: {weekly_session.id}")

        # 월간 세션 확인
        monthly_session = session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == analysis_month,
                SessionModel.period_type == "monthly",
            )
        )
        assert monthly_session is not None
        assert monthly_session.monthly_analysis_result is not None
        print(f"Successfully saved monthly session: {monthly_session.id}")

        # 메모리 확인
        memories = session.scalars(
            select(UserMemoryModel).where(UserMemoryModel.user_id == member_id)
        ).all()
        print(f"Found {len(memories)} memory summaries for user {member_id}")
        for m in memories:
            print(f"- {m.period_type}: {m.summary[:50]}...")

        assert any(m.period_type == "weekly" for m in memories)
        assert any(m.period_type == "monthly" for m in memories)


if __name__ == "__main__":
    test_feedback_saving()
