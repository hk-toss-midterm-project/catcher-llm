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


class InterventionTarget(BaseModel):
    """원인 해석에서 RAG 검색과 최종 피드백 판단으로 넘길 개입 타겟 후보를 표현한다."""

    target_type: str = Field(description="RAG 검색 타겟 유형. 구체적인 snake_case 값으로 작성한다.")
    title: str = Field(description="최종 행동 지시가 아닌 검색·검토용 개입 타겟 제목")
    linked_cause: str = Field(description="이 타겟과 연결된 원인 제목 또는 원인 요약")
    target_json_path: str = Field(
        description=(
            "직접 연결되는 원본 또는 지표 JSON 경로. 배열 인덱스만 단독으로 쓰지 말고 "
            "판단 필드까지 좁힌다."
        )
    )
    reason: str = Field(description="이 항목을 RAG 검색 타겟 후보로 넘기는 근거")
    query_hint: str | None = Field(default=None, description="RAG 검색에 사용할 짧은 질의 힌트")


class CauseAnalysisResult(BaseModel):
    """소비 원인 해석 결과를 구조화한다."""

    habitual_causes: list[SpendingFinding] = Field(default_factory=list)
    reward_causes: list[SpendingFinding] = Field(default_factory=list)
    stress_causes: list[SpendingFinding] = Field(default_factory=list)
    convenience_causes: list[SpendingFinding] = Field(default_factory=list)
    small_accumulation_causes: list[SpendingFinding] = Field(default_factory=list)
    one_off_high_spending_causes: list[SpendingFinding] = Field(
        default_factory=list,
        description=(
            "반복·습관이 아니라 단일 고액 결제 이벤트가 해당 기간 지출을 밀어 올린 원인 후보. "
            "고액 결제 항목의 실제 amount 경로를 근거로 작성한다."
        ),
    )
    fixed_cost_timing_causes: list[SpendingFinding] = Field(
        default_factory=list,
        description=(
            "관리비·통신비·공과금 등 고정비 납부 타이밍이 해당 기간 총액을 키운 원인 후보. "
            "불필요 지출로 단정하지 않고 납부 일정 점검 후보로만 작성한다."
        ),
    )
    period_concentration_causes: list[SpendingFinding] = Field(
        default_factory=list,
        description=(
            "특정 요일·기간 집중으로 총액이 커진 원인 후보. 요일·기간 집중 지표와 실제 금액 경로를 "
            "근거로 작성하고 심리 상태를 단정하지 않는다."
        ),
    )
    intervention_targets: list[InterventionTarget] = Field(
        default_factory=list,
        description=(
            "원인 해석에서 이어지는 RAG 검색 타겟 후보. 최종 행동이나 미션이 아니라 "
            "문서 검색과 최종 피드백 판단에 넘길 중간 산출물이다."
        ),
    )


class ActionMission(BaseModel):
    """RAG 검색과 최종 피드백 판단에 사용할 개선 후보와 개입 타겟을 표현한다."""

    action_type: str = Field(
        description=(
            "구체적인 snake_case 개입 후보 유형. generic immediate/substitution/mission 금지, "
            "가능하면 _candidate로 끝낸다."
        )
    )
    title: str = Field(description="검색과 검토에 사용할 후보 제목. 명령형 표현 금지.")
    detail: str = Field(
        description=(
            "후보가 겨냥하는 소비 행동과 검토 이유. 줄여보세요, 하세요, 재조정하세요, 계획하세요 "
            "같은 최종 행동 지시 문구는 쓰지 않는다."
        )
    )
    target_json_path: str = Field(
        description=(
            "직접 연결되는 원본 또는 지표 JSON 경로. 배열 인덱스만 단독으로 쓰지 말고 "
            "total_amount, diff_amount, amount, ratio_percent 같은 판단 필드까지 좁힌다."
        )
    )
    expected_effect: str = Field(
        description=(
            "최종 피드백에서 검토할 기대 효과. 확정 절약액 금지, 절약 가능처럼 단정하지 말고 "
            "가능성 검토나 원인 확인 수준으로 쓴다."
        )
    )
    urgency: ActionUrgency = Field(description="후보를 우선 검토할 시점")


class GroupCompetitionMetric(BaseModel):
    """그룹 경쟁 후보로 검토할 행동 지표를 표현한다."""

    metric_name: str = Field(description="그룹 경쟁 후보로 검토할 행동 기반 지표 이름")
    definition: str = Field(
        description="행동 기반 지표 계산 규칙. 총액·비율 같은 결과 지표만 제안하지 마라."
    )
    target_value: str = Field(description="후보 목표값")
    reason: str = Field(description="이 행동 기반 지표를 후보로 검토하는 이유")


class ActionAnalysisResult(BaseModel):
    """최종 미션이 아닌 개선 후보와 개입 타겟 도출 결과를 구조화한다."""

    immediate_cuts: list[ActionMission] = Field(
        default_factory=list,
        description="즉시 점검할 소비 개입 후보. 사용자가 바로 수행할 최종 절약 미션이 아니다.",
    )
    substitution_opportunities: list[ActionMission] = Field(
        default_factory=list,
        description="대체 가능성을 검토할 소비 후보. 특정 대체 행동을 근거 없이 확정하지 않는다.",
    )
    budget_control_areas: list[ActionMission] = Field(
        default_factory=list,
        description="예산 통제가 필요한 영역 후보. 재조정 지시가 아니라 검토 타겟으로 작성한다.",
    )
    next_week_missions: list[ActionMission] = Field(
        default_factory=list,
        description="다음 기간 최종 미션 후보. RAG와 메모리 확인 전에는 확정 미션으로 쓰지 않는다.",
    )
    group_competition_metrics: list[GroupCompetitionMetric] = Field(
        default_factory=list,
        description="그룹 경쟁에 사용할 행동 기반 지표 후보. 소비 총액 같은 결과 지표만 쓰지 않는다.",
    )


class CauseActionAnalysisResult(BaseModel):
    """원인 해석과 개선 후보·개입 타겟을 한 번의 구조화 출력으로 묶는다."""

    cause_result: CauseAnalysisResult = Field(default_factory=CauseAnalysisResult)
    action_result: ActionAnalysisResult = Field(default_factory=ActionAnalysisResult)


class SpendingAnalysisResult(BaseModel):
    """일일 소비 해석의 패턴, 문제, 원인, 개선 후보 결과 전체를 구조화한다."""

    pattern_result: PatternAnalysisResult = Field(default_factory=PatternAnalysisResult)
    problem_result: ProblemAnalysisResult = Field(default_factory=ProblemAnalysisResult)
    cause_result: CauseAnalysisResult = Field(default_factory=CauseAnalysisResult)
    action_result: ActionAnalysisResult = Field(default_factory=ActionAnalysisResult)


__all__ = [
    "PatternAnalysisResult",
    "ProblemAnalysisResult",
    "InterventionTarget",
    "CauseAnalysisResult",
    "ActionMission",
    "GroupCompetitionMetric",
    "ActionAnalysisResult",
    "CauseActionAnalysisResult",
    "SpendingAnalysisResult",
]
