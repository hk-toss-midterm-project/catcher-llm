from __future__ import annotations


def test_consumption_feedback_prompt_modules_group_related_builders() -> None:
    """소비 피드백 프롬프트 생성기가 분석·기간·피드백·메모리 모듈로 분리됐는지 검증한다."""
    from catcher_llm.prompts import consumption_feedback as legacy
    from catcher_llm.prompts.consumption_feedback import (
        analysis,
        build_consumption_pattern_prompt,
        build_daily_feedback_prompt,
        build_feedback_memory_rank_prompt,
        build_memory_summary_prompt,
        build_monthly_consumption_pattern_prompt,
        build_monthly_feedback_prompt,
        build_weekly_consumption_pattern_prompt,
        build_weekly_feedback_prompt,
        feedback,
        memory,
        monthly,
        weekly,
    )

    assert legacy.build_consumption_pattern_prompt is build_consumption_pattern_prompt
    assert build_consumption_pattern_prompt is analysis.build_consumption_pattern_prompt
    assert build_weekly_consumption_pattern_prompt is weekly.build_weekly_consumption_pattern_prompt
    assert (
        build_monthly_consumption_pattern_prompt is monthly.build_monthly_consumption_pattern_prompt
    )
    assert build_daily_feedback_prompt is feedback.build_daily_feedback_prompt
    assert build_weekly_feedback_prompt is feedback.build_weekly_feedback_prompt
    assert build_monthly_feedback_prompt is feedback.build_monthly_feedback_prompt
    assert build_memory_summary_prompt is memory.build_memory_summary_prompt
    assert build_feedback_memory_rank_prompt is memory.build_feedback_memory_rank_prompt
