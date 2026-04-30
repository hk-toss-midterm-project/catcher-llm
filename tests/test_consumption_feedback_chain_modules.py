from __future__ import annotations


def test_consumption_feedback_chain_modules_group_related_builders() -> None:
    """소비 피드백 체인 생성기가 분석·기간·피드백·메모리 모듈로 분리됐는지 검증한다."""
    from catcher_llm.chains import consumption_feedback as legacy
    from catcher_llm.chains.consumption_feedback import (
        analysis,
        build_consumption_pattern_chain,
        build_daily_feedback_chain,
        build_feedback_memory_rank_chain,
        build_memory_summary_chain,
        build_monthly_feedback_chain,
        build_monthly_spending_analysis_chain,
        build_spending_analysis_chain,
        build_weekly_feedback_chain,
        build_weekly_spending_analysis_chain,
        feedback,
        memory,
        monthly,
        weekly,
    )

    assert legacy.build_consumption_pattern_chain is build_consumption_pattern_chain
    assert build_consumption_pattern_chain is analysis.build_consumption_pattern_chain
    assert build_spending_analysis_chain is analysis.build_spending_analysis_chain
    assert build_weekly_spending_analysis_chain is weekly.build_weekly_spending_analysis_chain
    assert build_monthly_spending_analysis_chain is monthly.build_monthly_spending_analysis_chain
    assert build_daily_feedback_chain is feedback.build_daily_feedback_chain
    assert build_weekly_feedback_chain is feedback.build_weekly_feedback_chain
    assert build_monthly_feedback_chain is feedback.build_monthly_feedback_chain
    assert build_memory_summary_chain is memory.build_memory_summary_chain
    assert build_feedback_memory_rank_chain is memory.build_feedback_memory_rank_chain
