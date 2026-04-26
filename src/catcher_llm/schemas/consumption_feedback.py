from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type MetricValue = bool | int | float | str
type CategoryDirection = Literal["increase", "decrease", "flat"]
type FindingConfidence = Literal["low", "medium", "high"]


class SourcePaths(BaseModel):
    """사용자 소비 JSON의 원천 데이터 경로를 표현한다."""

    past_csv: str | None = None
    today_csv: str | None = None
    past_source: str | None = None
    today_source: str | None = None


class OutlierThresholds(BaseModel):
    """이상 지출 판단에 사용한 사분위수 기준값을 표현한다."""

    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float


class CategoryRatioChange(BaseModel):
    """카테고리별 평소 대비 당일 소비 비중 변화를 표현한다."""

    category: str
    usual_ratio_percent: float
    today_ratio_percent: float
    diff_point: float


class StableMetrics(BaseModel):
    """클리핑 데이터 기반 안정 소비 지표를 표현한다."""

    past_daily_stable_average: float
    today_total: int
    increase_rate_percent: float
    category_ratio_changes: list[CategoryRatioChange]


class HighSpendingItem(BaseModel):
    """원본 데이터에서 탐지한 특이 고액 지출 항목을 표현한다."""

    used_at: str
    description: str
    amount: int
    category: str


class AnomalyDetection(BaseModel):
    """원본 데이터 기반 소비 규모와 이상 지출 탐지 결과를 표현한다."""

    past_daily_original_average: float
    spike_ratio: float
    is_spike: bool
    high_spending_threshold: float
    high_spending_items: list[HighSpendingItem]


class PreviousDayComparison(BaseModel):
    """전날 대비 소비 비교 결과를 표현한다."""

    yesterday_date: str
    yesterday_total: int
    today_total: int
    amount_diff: int
    amount_diff_rate_percent: float
    yesterday_count: int
    today_count: int
    count_diff: int
    yesterday_main_category: str | None
    today_main_category: str | None


class TimeSlotComparison(BaseModel):
    """시간대별 당일 소비와 평소 소비 차이를 표현한다."""

    time_slot: str
    today_amount: int
    usual_average_amount: float
    diff_amount: float


class TimeSlotAnalysis(BaseModel):
    """시간대별 소비 피크와 세부 비교 결과를 표현한다."""

    peak_slot: str | None
    time_slots: list[TimeSlotComparison]


class UserSpendingData(BaseModel):
    """일일 소비 분석 JSON 전체 구조를 검증 가능한 입력 모델로 표현한다."""

    member_id: int
    analysis_date: str
    source_paths: SourcePaths
    outlier_thresholds: OutlierThresholds
    stable_metrics: StableMetrics
    anomaly_detection: AnomalyDetection
    previous_day_comparison: PreviousDayComparison
    time_slot_analysis: TimeSlotAnalysis


class SpendingMetric(BaseModel):
    """JSON 데이터에서 바로 추출한 단일 소비 지표를 표현한다."""

    name: str
    value: MetricValue
    unit: str
    source_json_path: str
    description: str


class CategoryShiftIndicator(BaseModel):
    """카테고리 비중 변화 지표와 원본 JSON 경로를 함께 표현한다."""

    category: str
    usual_ratio_percent: float
    today_ratio_percent: float
    diff_point: float
    direction: CategoryDirection
    source_json_path: str


class MainCategoryShift(BaseModel):
    """전날 대비 주 소비 카테고리 변화 지표를 표현한다."""

    previous_category: str | None
    current_category: str | None
    source_json_path: str


class SpendingIndicatorPayload(BaseModel):
    """분석 체인에 직접 전달할 JSON 기반 소비 지표 묶음을 표현한다."""

    member_id: int
    analysis_date: str
    metrics: list[SpendingMetric]
    category_ratio_changes: list[CategoryShiftIndicator]
    largest_category_increase: CategoryShiftIndicator | None
    largest_category_decrease: CategoryShiftIndicator | None
    high_spending_items: list[HighSpendingItem]
    main_category_shift: MainCategoryShift
    time_slot_diffs: list[TimeSlotComparison]


class EvidenceItem(BaseModel):
    """JSON 지표 분석에서 참조한 근거 값을 표현한다."""

    json_path: str = Field(description="근거가 나온 원본 또는 지표 JSON 경로")
    supporting_value: str = Field(description="근거가 된 JSON 값")
    reason: str = Field(description="이 근거를 선택한 이유")


class SpendingFinding(BaseModel):
    """소비 분석의 단일 탐지 결과를 표현한다."""

    subcategory: str = Field(description="세부 분석 분류")
    title: str = Field(description="한 줄 요약")
    detail: str = Field(description="근거를 포함한 구체적인 설명")
    confidence: FindingConfidence = Field(description="판단 신뢰도")
    evidences: list[EvidenceItem] = Field(description="판단에 사용한 JSON 근거 목록")


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
    urgency: Literal["immediate", "this_week", "this_month"] = Field(
        description="실행 우선순위 시점"
    )


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
