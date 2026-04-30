from __future__ import annotations

from pydantic import BaseModel, Field

from catcher_llm.schemas.consumption_feedback.base import (
    ActionUrgency,
    SpendingFinding,
)


class PatternAnalysisResult(BaseModel):
    """소비 패턴 탐지 결과를 구조화한다."""

    repeated_consumption: list[SpendingFinding] = Field(default_factory=list)
    overspending_windows: list[SpendingFinding] = Field(default_factory=list)
    impulse_patterns: list[SpendingFinding] = Field(default_factory=list)
    contextual_patterns: list[SpendingFinding] = Field(default_factory=list)


class ProblemAnalysisResult(BaseModel):
    """문제 소비 식별 결과를 구조화한다."""

    money_leaks: list[SpendingFinding] = Field(default_factory=list)
    saving_blockers: list[SpendingFinding] = Field(default_factory=list)
    fixed_cost_issues: list[SpendingFinding] = Field(default_factory=list)
    variable_cost_issues: list[SpendingFinding] = Field(default_factory=list)
    short_term_problem_spending: list[SpendingFinding] = Field(default_factory=list)
    long_term_problem_spending: list[SpendingFinding] = Field(default_factory=list)


class CauseAnalysisResult(BaseModel):
    """소비 원인 해석 결과를 구조화한다."""

    habitual_causes: list[SpendingFinding] = Field(default_factory=list)
    reward_causes: list[SpendingFinding] = Field(default_factory=list)
    stress_causes: list[SpendingFinding] = Field(default_factory=list)
    convenience_causes: list[SpendingFinding] = Field(default_factory=list)
    small_accumulation_causes: list[SpendingFinding] = Field(default_factory=list)


class ActionMission(BaseModel):
    """행동 개선 포인트와 실행 미션을 표현한다."""

    action_type: str = Field(description="행동 포인트 유형")
    title: str = Field(description="실행 항목 제목")
    detail: str = Field(description="실행 방법 설명")
    target_json_path: str = Field(description="직접 연결되는 원본 또는 지표 JSON 경로")
    expected_effect: str = Field(description="기대 효과")
    urgency: ActionUrgency = Field(description="실행 우선순위 시점")


class GroupCompetitionMetric(BaseModel):
    """그룹 경쟁에 반영할 행동 지표를 표현한다."""

    metric_name: str = Field(description="행동 지표 이름")
    definition: str = Field(description="행동 지표 계산 규칙")
    target_value: str = Field(description="권장 목표값")
    reason: str = Field(description="이 지표를 추천하는 이유")


class ActionAnalysisResult(BaseModel):
    """행동 개선 포인트 도출 결과를 구조화한다."""

    immediate_cuts: list[ActionMission] = Field(default_factory=list)
    substitution_opportunities: list[ActionMission] = Field(default_factory=list)
    budget_control_areas: list[ActionMission] = Field(default_factory=list)
    next_week_missions: list[ActionMission] = Field(default_factory=list)
    group_competition_metrics: list[GroupCompetitionMetric] = Field(default_factory=list)


class CauseActionAnalysisResult(BaseModel):
    """원인 해석과 행동 개선 포인트를 한 번의 구조화 출력으로 묶는다."""

    cause_result: CauseAnalysisResult = Field(default_factory=CauseAnalysisResult)
    action_result: ActionAnalysisResult = Field(default_factory=ActionAnalysisResult)


class SpendingAnalysisResult(BaseModel):
    """일일 소비 해석의 패턴, 문제, 원인, 행동 결과 전체를 구조화한다."""

    pattern_result: PatternAnalysisResult = Field(default_factory=PatternAnalysisResult)
    problem_result: ProblemAnalysisResult = Field(default_factory=ProblemAnalysisResult)
    cause_result: CauseAnalysisResult = Field(default_factory=CauseAnalysisResult)
    action_result: ActionAnalysisResult = Field(default_factory=ActionAnalysisResult)


__all__ = [
    "PatternAnalysisResult",
    "ProblemAnalysisResult",
    "CauseAnalysisResult",
    "ActionMission",
    "GroupCompetitionMetric",
    "ActionAnalysisResult",
    "CauseActionAnalysisResult",
    "SpendingAnalysisResult",
]
