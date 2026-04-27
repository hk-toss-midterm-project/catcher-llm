from __future__ import annotations

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from catcher_llm.schemas.consumption_feedback import (
    DailyFeedbackMemoryContext,
    DailyFeedbackResult,
    DailyFeedbackServiceResult,
    DailyFeedbackSessionContext,
    UserProfileContext,
)
from catcher_llm.ui.dev_navigation import get_dev_page_specs, get_repo_root


def test_get_repo_root_points_to_project_root() -> None:
    """개발 앱 네비게이션이 저장소 루트를 올바르게 기준점으로 삼는지 검증한다."""
    root = get_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "dev_pages").exists()


def test_get_dev_page_specs_registers_dev_pages_directory() -> None:
    """개발용 Streamlit 페이지들이 사이드바 표시 순서와 메타데이터로 등록되는지 검증한다."""
    specs = get_dev_page_specs()

    assert [spec.path.name for spec in specs] == [
        "01_chat.py",
        "02_retriever_probe.py",
        "03_document_rag.py",
        "04_daily_analysis.py",
        "05_consumption_interpretation.py",
        "06_daily_feedback.py",
        "07_weekly_analysis.py",
        "08_monthly_analysis.py",
    ]
    assert [spec.title for spec in specs] == [
        "Chat",
        "Retriever 테스트",
        "문서 RAG",
        "일일 소비 분석",
        "소비 해석 체인",
        "일일 피드백",
        "주간 소비 분석",
        "월간 소비 분석",
    ]
    assert [spec.icon for spec in specs] == [
        "💬",
        "🔎",
        "📄",
        "📊",
        "🧭",
        "📣",
        "🗓️",
        "📈",
    ]
    assert [spec.default for spec in specs] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(spec.path.is_file() for spec in specs)


def test_dev_app_renders_default_page_without_exception() -> None:
    """Streamlit 개발 앱의 기본 페이지가 import 예외 없이 렌더링되는지 검증한다."""
    app = AppTest.from_file("dev_app.py")

    app.run(timeout=10)

    assert len(app.exception) == 0


def test_daily_feedback_dev_page_renders_profile_and_memory_context() -> None:
    """일일 피드백 개발 페이지가 서비스의 사용자 프로필과 메모리 컨텍스트를 표시하는지 검증한다."""
    fake_result = DailyFeedbackServiceResult(
        member_id=1,
        analysis_date="2024-04-01",
        feedback=DailyFeedbackResult(
            summary_title="오늘은 생활비를 확인하세요",
            scolding_message="생활 카테고리 소비가 커졌습니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="내일 오전 장보기 목록을 확인합니다.",
        ),
        user_profile=UserProfileContext(
            user_id=1,
            name="김토스",
            job="개발자",
            saving_goal_text="비상금 300만원 만들기",
        ),
        memory_context=DailyFeedbackMemoryContext(
            user_id=1,
            memory_summary="최근 식비와 쇼핑 지출이 반복적으로 높다.",
            recent_sessions=[
                DailyFeedbackSessionContext(
                    analysis_date="2024-03-31",
                    todo_tomorrow="간식 결제를 줄인다.",
                )
            ],
        ),
        retrieval_queries=["비상금 300만원 만들기 목표 소비 절약 방법"],
    )

    with patch(
        "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
        return_value=fake_result,
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)
        app.button[0].click().run(timeout=10)

    assert len(app.exception) == 0
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)
    assert any(expander.label == "메모리/세션 컨텍스트 JSON" for expander in app.expander)
