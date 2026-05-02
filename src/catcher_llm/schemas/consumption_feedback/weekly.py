from __future__ import annotations

from pydantic import BaseModel, Field

from catcher_llm.schemas.consumption_feedback.base import (
    ActionUrgency,
    CategoryDirection,
    JsonObject,
    JsonValue,
    RetrievedAdviceContext,
    SpendingMetric,
    UserProfileContext,
    WeeklyFeedbackEvidenceType,
)
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackMemoryContext


class WeeklyOutlierThresholds(BaseModel):
    """주간 소비 분석에서 이상 지출 판단에 사용한 사분위수 기준값을 표현한다."""

    q1: float
    q3: float
    iqr: float
    upper_bound: float


class WeeklySummary(BaseModel):
    """주간 총 지출과 전주 대비 변화를 표현한다."""

    week_start: str
    week_end: str
    this_week_total: int
    prev_week_total: int
    amount_diff: int
    diff_rate_percent: float
    daily_average: float
    max_day_date: str | None
    max_day_amount: int
    min_day_date: str | None
    min_day_amount: int
    transaction_count: int


class WeeklyCategorySummary(BaseModel):
    """주간 카테고리별 지출과 전주 대비 변화를 표현한다."""

    category: str
    total_amount: int
    ratio_percent: float
    transaction_count: int
    prev_week_amount: int
    diff_amount: int
    diff_rate_percent: float


class WeeklyPeriodComparison(BaseModel):
    """주간 분석에서 하나의 기준 주간과 분석 주간을 비교한 결과를 표현한다."""

    label: str = ""
    reference_start_date: str = ""
    reference_end_date: str = ""
    current_start_date: str = ""
    current_end_date: str = ""
    reference_total: int = 0
    current_total: int = 0
    amount_diff: int = 0
    amount_diff_rate_percent: float = 0.0
    reference_count: int = 0
    current_count: int = 0
    count_diff: int = 0
    week_num: int | None = None


class WeeklyReferencePeriod(BaseModel):
    """최근 주간 평균 계산에 포함된 기준 주간의 소비를 표현한다."""

    start_date: str
    end_date: str
    total: int
    transaction_count: int


class WeeklyAverageComparison(BaseModel):
    """주간 분석에서 여러 기준 주간 평균과 분석 주간을 비교한 결과를 표현한다."""

    label: str = ""
    reference_periods: list[WeeklyReferencePeriod] = Field(default_factory=list)
    reference_week_count: int = 0
    average_total: float = 0.0
    current_total: int = 0
    amount_diff: float = 0.0
    amount_diff_rate_percent: float = 0.0
    average_count: float = 0.0
    current_count: int = 0
    count_diff: float = 0.0


class WeeklyComparisons(BaseModel):
    """주간 분석의 전주·최근 4주 평균·지난달 같은 주차 비교를 묶는다."""

    previous_week: WeeklyPeriodComparison = Field(default_factory=WeeklyPeriodComparison)
    recent_4week_average: WeeklyAverageComparison = Field(default_factory=WeeklyAverageComparison)
    same_week_last_month: WeeklyPeriodComparison = Field(default_factory=WeeklyPeriodComparison)


class WeeklyMerchantVisit(BaseModel):
    """주간 반복 가맹점 방문 횟수와 누적 금액을 표현한다."""

    merchant: str
    visit_count: int
    total_amount: int
    main_category: str | None = None


class WeeklyConsecutiveMerchant(BaseModel):
    """주간 연속 소비가 감지된 가맹점과 최장 연속 일수를 표현한다."""

    merchant: str
    max_consecutive_days: int


class WeeklyMerchantSummary(BaseModel):
    """주간 특정 가맹점군의 건수, 합계, 건당 평균을 표현한다."""

    count: int
    total_amount: int
    avg_per_transaction: float


class WeeklyRepeatPatterns(BaseModel):
    """주간 반복 소비 패턴 묶음을 표현한다."""

    top_merchants: list[WeeklyMerchantVisit] = Field(default_factory=list)
    consecutive_merchants: list[WeeklyConsecutiveMerchant] = Field(default_factory=list)
    delivery: WeeklyMerchantSummary
    cafe: WeeklyMerchantSummary
    convenience: WeeklyMerchantSummary
    taxi: WeeklyMerchantSummary


class WeeklyWeekdayBreakdown(BaseModel):
    """주간 요일별 지출액과 결제 건수를 표현한다."""

    weekday: str
    weekday_num: int
    total_amount: int
    transaction_count: int


class WeeklyWeekdayPattern(BaseModel):
    """주간 요일별 소비 패턴 요약을 표현한다."""

    weekday_breakdown: list[WeeklyWeekdayBreakdown] = Field(default_factory=list)
    peak_weekday: str | None
    weekday_average: float
    weekend_average: float
    weekday_vs_weekend_diff: float


class WeeklyCategoryAmount(BaseModel):
    """주간 문제 소비 탐지에서 카테고리별 금액을 표현한다."""

    category: str
    amount: int


class WeeklyLateNightSpending(BaseModel):
    """주간 야간 소비 탐지 결과를 표현한다."""

    total_amount: int
    count: int
    ratio_percent: float
    top_categories: list[WeeklyCategoryAmount] = Field(default_factory=list)


class WeeklyMicroSpending(BaseModel):
    """주간 소액 결제 탐지 결과를 표현한다."""

    threshold: int
    total_amount: int
    count: int
    top_categories: list[WeeklyCategoryAmount] = Field(default_factory=list)


class WeeklyHighSpendingItem(BaseModel):
    """주간 고액 결제 항목을 표현한다."""

    used_at: str
    merchant: str
    amount: int
    category: str


class WeeklyHighSpending(BaseModel):
    """주간 고액 결제 탐지 결과를 표현한다."""

    iqr_upper_bound: float
    total_amount: int
    count: int
    items: list[WeeklyHighSpendingItem] = Field(default_factory=list)


class WeeklyWasteDetection(BaseModel):
    """주간 야간·소액·고액 소비 탐지 결과를 표현한다."""

    late_night: WeeklyLateNightSpending
    micro_spending: WeeklyMicroSpending
    high_spending: WeeklyHighSpending


class WeeklyCategoryDiff(BaseModel):
    """전주 대비 개선 또는 악화된 카테고리와 증감액을 표현한다."""

    category: str
    diff_amount: int


class WeeklySavingPotential(BaseModel):
    """주간 절약 가능성 추정 결과를 표현한다."""

    delivery_save_per_skip: int
    cafe_save_half_visits: int
    improved_categories: list[WeeklyCategoryDiff] = Field(default_factory=list)
    worsened_categories: list[WeeklyCategoryDiff] = Field(default_factory=list)


class WeeklyElasticityAnalysis(BaseModel):
    """주간 소비 탄성 및 심리적 반동 분석 결과를 표현한다."""

    correlation: float | None = None
    threshold: int | None = None
    rebound_avg: int | None = None
    normal_avg: int | None = None
    cheat_effective: bool | None = None
    recommended_cheat_amount: int | None = None


class WeeklyRoutineIndicators(BaseModel):
    """문서 기준 주간 루틴 지표에 포함되는 반복 소비 목록을 표현한다."""

    top_merchants: list[JsonValue] = Field(default_factory=list)
    consecutive_merchants: list[JsonValue] = Field(default_factory=list)


class WeeklySpecialMetrics(BaseModel):
    """문서 기준 주간 특수 지표를 표현한다."""

    weekend_overspending_index: float = 0.0
    weekday_concentration_ratio_percent: float = 0.0
    routine_indicators: WeeklyRoutineIndicators = Field(default_factory=WeeklyRoutineIndicators)


class WeeklyMetrics(BaseModel):
    """문서 기준 주간 소비 지표 묶음을 표현한다."""

    weekly_total_amount: int = 0
    weekly_average_daily_amount: float = 0.0
    weekly_transaction_count: int = 0
    weekday_spending_ratio_percent: float = 0.0
    weekend_spending_ratio_percent: float = 0.0
    weekday_spending_pattern: list[JsonValue] = Field(default_factory=list)
    category_spending: list[JsonValue] = Field(default_factory=list)
    previous_week_change_rate_percent: float = 0.0
    weekly_spending_volatility: float = 0.0
    weekly_budget_usage_rate_percent: float | None = None
    weekly_remaining_budget: int | None = None
    weekly_overspend_amount: int | None = None
    weekly_income_usage_rate_percent: float | None = None
    weekly_budget_burn_rate: float | None = None
    month_to_date_budget_usage_rate_percent: float | None = None
    projected_monthly_spending_from_weekly_pace: int | None = None
    special_metrics: WeeklySpecialMetrics = Field(default_factory=WeeklySpecialMetrics)
    weekend_overspending_index: float = 0.0


class WeeklySpendingData(BaseModel):
    """주간 소비 분석 JSON 전체 구조를 검증 가능한 입력 모델로 표현한다."""

    member_id: int
    week_start: str
    week_end: str
    source_path: str | None = None
    outlier_thresholds: WeeklyOutlierThresholds
    weekly_summary: WeeklySummary
    category_summary: list[WeeklyCategorySummary] = Field(default_factory=list)
    weekly_comparisons: WeeklyComparisons = Field(default_factory=WeeklyComparisons)
    repeat_patterns: WeeklyRepeatPatterns
    weekday_pattern: WeeklyWeekdayPattern
    waste_detection: WeeklyWasteDetection
    saving_potential: WeeklySavingPotential
    elasticity_analysis: WeeklyElasticityAnalysis = Field(default_factory=WeeklyElasticityAnalysis)
    weekly_metrics: WeeklyMetrics = Field(default_factory=WeeklyMetrics)


class WeeklyCategoryChangeIndicator(BaseModel):
    """주간 카테고리 증감 지표와 원본 JSON 경로를 함께 표현한다."""

    category: str
    total_amount: int
    prev_week_amount: int
    diff_amount: int
    diff_rate_percent: float
    direction: CategoryDirection
    source_json_path: str


class WeeklySpendingIndicatorPayload(BaseModel):
    """주간 해석 체인에 직접 전달할 JSON 기반 소비 지표 묶음을 표현한다."""

    member_id: int
    week_start: str
    week_end: str
    metrics: list[SpendingMetric]
    category_changes: list[WeeklyCategoryChangeIndicator]
    largest_category_increase: WeeklyCategoryChangeIndicator | None
    largest_category_decrease: WeeklyCategoryChangeIndicator | None
    high_spending_items: list[WeeklyHighSpendingItem]
    top_merchants: list[WeeklyMerchantVisit]
    peak_weekday: str | None


class WeeklyFeedbackEvidence(BaseModel):
    """주간 소비 피드백 문장에 사용한 소비 JSON 또는 문서 근거를 표현한다."""

    evidence_type: WeeklyFeedbackEvidenceType
    title: str
    detail: str
    source_json_path: str | None = None
    source: str | None = None
    page_number: int | None = None


class WeeklyFeedbackAction(BaseModel):
    """주간 소비 피드백에서 사용자가 다음 주에 실행할 행동을 표현한다."""

    title: str
    detail: str
    target_json_path: str
    urgency: ActionUrgency
    related_source: str | None = None


class WeeklyFeedbackResult(BaseModel):
    """주간 분석 JSON과 RAG 근거를 바탕으로 생성한 최종 주간 소비 피드백을 표현한다."""

    summary_title: str
    feedback_message: str
    key_evidences: list[WeeklyFeedbackEvidence]
    action_items: list[WeeklyFeedbackAction]
    next_week_mission: str


class WeeklyFeedbackServiceResult(BaseModel):
    """주간 소비 피드백 서비스 실행 결과와 중간 산출물을 표현한다."""

    member_id: int
    week_start: str
    week_end: str
    feedback: WeeklyFeedbackResult | None = None
    weekly_analysis: WeeklySpendingData | None = None
    interpretation_result: JsonObject | None = None
    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedAdviceContext] = Field(default_factory=list)
    error: str | None = None


__all__ = [
    "WeeklyOutlierThresholds",
    "WeeklySummary",
    "WeeklyCategorySummary",
    "WeeklyPeriodComparison",
    "WeeklyReferencePeriod",
    "WeeklyAverageComparison",
    "WeeklyComparisons",
    "WeeklyMerchantVisit",
    "WeeklyConsecutiveMerchant",
    "WeeklyMerchantSummary",
    "WeeklyRepeatPatterns",
    "WeeklyWeekdayBreakdown",
    "WeeklyWeekdayPattern",
    "WeeklyCategoryAmount",
    "WeeklyLateNightSpending",
    "WeeklyMicroSpending",
    "WeeklyHighSpendingItem",
    "WeeklyHighSpending",
    "WeeklyWasteDetection",
    "WeeklyCategoryDiff",
    "WeeklySavingPotential",
    "WeeklyElasticityAnalysis",
    "WeeklyRoutineIndicators",
    "WeeklySpecialMetrics",
    "WeeklyMetrics",
    "WeeklySpendingData",
    "WeeklyCategoryChangeIndicator",
    "WeeklySpendingIndicatorPayload",
    "WeeklyFeedbackEvidence",
    "WeeklyFeedbackAction",
    "WeeklyFeedbackResult",
    "WeeklyFeedbackServiceResult",
]
