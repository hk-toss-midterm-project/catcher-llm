from __future__ import annotations


def test_consumption_feedback_schema_modules_group_related_models() -> None:
    """소비 피드백 스키마가 공통·일간·주간·월간·해석 결과 모듈로 분리됐는지 검증한다."""
    from catcher_llm.schemas import consumption_feedback as legacy
    from catcher_llm.schemas.consumption_feedback import (
        ActionAnalysisResult,
        DailyFeedbackServiceResult,
        JsonObject,
        MonthlyFeedbackServiceResult,
        RetrievedAdviceContext,
        UserSpendingData,
        WeeklyFeedbackServiceResult,
        analysis_outputs,
        base,
        daily,
        monthly,
        weekly,
    )

    assert legacy.JsonObject is JsonObject
    assert JsonObject is base.JsonObject
    assert RetrievedAdviceContext is base.RetrievedAdviceContext
    assert ActionAnalysisResult is analysis_outputs.ActionAnalysisResult
    assert UserSpendingData is daily.UserSpendingData
    assert DailyFeedbackServiceResult is daily.DailyFeedbackServiceResult
    assert WeeklyFeedbackServiceResult is weekly.WeeklyFeedbackServiceResult
    assert MonthlyFeedbackServiceResult is monthly.MonthlyFeedbackServiceResult
