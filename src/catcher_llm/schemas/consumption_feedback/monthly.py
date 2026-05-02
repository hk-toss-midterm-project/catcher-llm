from __future__ import annotations

from pydantic import BaseModel, Field

from catcher_llm.schemas.consumption_feedback.base import (
    ActionUrgency,
    CategoryDirection,
    JsonObject,
    JsonValue,
    MonthlyFeedbackEvidenceType,
    RetrievedAdviceContext,
    SpendingMetric,
    UserProfileContext,
)
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackMemoryContext


class MonthlyOutlierThresholds(BaseModel):
    """월간 소비 분석에서 이상 지출 판단에 사용한 사분위수 기준값을 표현한다."""

    q1: float
    q3: float
    iqr: float
    upper_bound: float


class MonthlySummary(BaseModel):
    """월간 총 지출과 전월 대비 변화를 표현한다."""

    analysis_month: str
    this_month_total: int
    prev_month_total: int
    amount_diff: int
    diff_rate_percent: float
    daily_average: float
    active_days: int
    max_day_date: str | None
    max_day_amount: int
    min_day_date: str | None
    min_day_amount: int
    transaction_count: int


class MonthlyPeriodComparison(BaseModel):
    """월간 분석에서 기준월 하나와 분석월을 비교한 결과를 표현한다."""

    label: str = ""
    reference_month: str = ""
    current_month: str = ""
    reference_total: int = 0
    current_total: int = 0
    amount_diff: int = 0
    amount_diff_rate_percent: float = 0.0
    reference_count: int = 0
    current_count: int = 0
    count_diff: int = 0


class MonthlyReferenceMonth(BaseModel):
    """최근 월간 평균 계산에 포함된 기준월의 소비를 표현한다."""

    month: str
    total: int
    transaction_count: int


class MonthlyAverageComparison(BaseModel):
    """월간 분석에서 여러 기준월 평균과 분석월을 비교한 결과를 표현한다."""

    label: str = ""
    reference_months: list[str] = Field(default_factory=list)
    reference_month_details: list[MonthlyReferenceMonth] = Field(default_factory=list)
    reference_month_count: int = 0
    average_total: float = 0.0
    current_month: str = ""
    current_total: int = 0
    amount_diff: float = 0.0
    amount_diff_rate_percent: float = 0.0
    average_count: float = 0.0
    current_count: int = 0
    count_diff: float = 0.0


class MonthlyComparisons(BaseModel):
    """월간 분석의 전월·최근 3개월 평균 비교를 묶는다."""

    previous_month: MonthlyPeriodComparison = Field(default_factory=MonthlyPeriodComparison)
    recent_3month_average: MonthlyAverageComparison = Field(
        default_factory=MonthlyAverageComparison
    )


class MonthlyFixedItem(BaseModel):
    """월간 고정비 가맹점별 결제 건수와 누적 금액을 표현한다."""

    merchant: str
    count: int
    total_amount: int


class MonthlyFixedVariable(BaseModel):
    """월간 고정비와 변동비 비중을 표현한다."""

    fixed_total: int
    variable_total: int
    fixed_ratio_percent: float
    variable_ratio_percent: float
    fixed_items: list[MonthlyFixedItem] = Field(default_factory=list)


class MonthlyCategoryDeep(BaseModel):
    """월간 카테고리별 지출, 유형, 전월 대비 변화를 표현한다."""

    category: str
    type: str
    total_amount: int
    ratio_percent: float
    transaction_count: int
    prev_month_amount: int
    diff_amount: int
    diff_rate_percent: float


class MonthlyMerchantSummary(BaseModel):
    """월간 특정 가맹점군의 건수, 합계, 건당 평균을 표현한다."""

    count: int
    total_amount: int
    avg_per_transaction: float


class MonthlyMerchantVisit(BaseModel):
    """월간 반복 가맹점 방문 횟수와 누적 금액을 표현한다."""

    merchant: str
    visit_count: int
    total_amount: int


class MonthlyRepeatPatterns(BaseModel):
    """월간 반복 소비 패턴 묶음을 표현한다."""

    top5_merchants: list[MonthlyMerchantVisit] = Field(default_factory=list)
    delivery: MonthlyMerchantSummary
    cafe: MonthlyMerchantSummary
    convenience: MonthlyMerchantSummary
    taxi: MonthlyMerchantSummary


class MonthlyWeeklyBreakdown(BaseModel):
    """월간 주차별 지출액과 결제 건수를 표현한다."""

    week_num: int
    start_date: str
    end_date: str
    total_amount: int
    count: int


class MonthlyWeeklyTrend(BaseModel):
    """월간 주차별 소비 추이 요약을 표현한다."""

    weekly_breakdown: list[MonthlyWeeklyBreakdown] = Field(default_factory=list)
    trend_direction: str


class MonthlyCategoryCountAmount(BaseModel):
    """월간 문제 소비 탐지에서 카테고리별 금액과 건수를 표현한다."""

    category: str
    total_amount: int
    count: int


class MonthlyMicroSpending(BaseModel):
    """월간 소액 결제 탐지 결과를 표현한다."""

    threshold: int
    total_amount: int
    count: int
    ratio_percent: float
    top_categories: list[MonthlyCategoryCountAmount] = Field(default_factory=list)


class MonthlyLateNightSpending(BaseModel):
    """월간 야간 소비 탐지 결과를 표현한다."""

    late_night_hour: int
    total_amount: int
    count: int
    ratio_percent: float
    top_categories: list[MonthlyCategoryCountAmount] = Field(default_factory=list)


class MonthlyHighSpendingItem(BaseModel):
    """월간 고액 결제 항목을 표현한다."""

    used_at: str
    merchant: str
    amount: int
    category: str


class MonthlyHighSpending(BaseModel):
    """월간 고액 결제 탐지 결과를 표현한다."""

    iqr_upper_bound: float
    total_amount: int
    count: int
    items: list[MonthlyHighSpendingItem] = Field(default_factory=list)


class MonthlySavingPotential(BaseModel):
    """월간 절약 가능성 추정 결과를 표현한다."""

    delivery_reduce_30pct: int
    cafe_every_other_day: int
    taxi_to_transit: int
    micro_reduce_20pct: int
    total_potential_saving: int
    next_month_recommended_target: int


class MonthlyCashFlowVolatility(BaseModel):
    """월간 현금 흐름 변동성 지수(Cash Flow Volatility) 결과를 표현한다."""

    mean_weekly: int = 0
    std_weekly: int = 0
    cv_index: float = 0.0
    pace_status: str = ""
    weekly_ratios: list[JsonValue] = Field(default_factory=list)


class MonthlySpendingConcentration(BaseModel):
    """월간 파레토 지출 쏠림 지수(Spending Concentration Index) 결과를 표현한다."""

    total_variable_amount: int = 0
    top_1_category: str | None = None
    top_1_amount: int = 0
    top_1_ratio_percent: float = 0.0
    top_2_category: str | None = None
    top_2_amount: int = 0
    top_2_ratio_percent: float = 0.0
    top_2_combined_ratio_percent: float = 0.0
    concentration_status: str = ""


class MonthlyFrictionlessSpending(BaseModel):
    """월간 마찰력 없는 지출(온라인/간편결제) 지표를 표현한다."""

    total_amount: int = 0
    count: int = 0
    ratio_percent: float = 0.0


class MonthlyTransactionDensity(BaseModel):
    """월간 결제 밀도(일평균 결제 횟수, 건당 평균액) 지표를 표현한다."""

    avg_daily_count: float = 0.0
    avg_per_transaction: int = 0


class MonthlyFrictionlessAndDensity(BaseModel):
    """월간 지출 마찰력 및 밀도 분석 지표를 표현한다."""

    frictionless_spending: MonthlyFrictionlessSpending = Field(
        default_factory=MonthlyFrictionlessSpending
    )
    transaction_density: MonthlyTransactionDensity = Field(
        default_factory=MonthlyTransactionDensity
    )


class MonthlyInstallmentItem(BaseModel):
    """월간 할부 결제 항목을 표현한다."""

    used_at: str
    merchant: str
    amount: int
    installment_months: int


class MonthlyInstallmentDebtPressure(BaseModel):
    """월간 할부 부채 압박 지수(Installment Debt Pressure Index) 결과를 표현한다."""

    total_installment_amount: int = 0
    installment_count: int = 0
    installment_ratio_percent: float = 0.0
    avg_installment_months: float = 0.0
    max_installment_months: int = 0
    items: list[MonthlyInstallmentItem] = Field(default_factory=list)


class MonthlySpecialMetrics(BaseModel):
    """문서 기준 월간 특수 지표를 표현한다."""

    fixed_cost_burden_rate_percent: float | None = None
    spending_capacity: int | None = None
    subscription_leakage_rate_percent: float | None = None


class MonthlyMetrics(BaseModel):
    """문서 기준 월간 소비 지표 묶음을 표현한다."""

    monthly_total_amount: int = 0
    monthly_budget_usage_rate_percent: float | None = None
    previous_month_change_rate_percent: float = 0.0
    fixed_cost_amount: int = 0
    fixed_cost_ratio_percent: float = 0.0
    variable_cost_amount: int = 0
    category_monthly_spending_ratio: list[JsonValue] = Field(default_factory=list)
    subscription_total: int = 0
    post_salary_spending_increase_rate_percent: float | None = None
    month_end_pressure_index: float | None = None
    special_metrics: MonthlySpecialMetrics = Field(default_factory=MonthlySpecialMetrics)
    fixed_cost_burden_rate_percent: float | None = None


class MonthlySpendingData(BaseModel):
    """월간 소비 분석 JSON 전체 구조를 검증 가능한 입력 모델로 표현한다."""

    member_id: int
    analysis_month: str
    prev_month: str
    source_path: str | None = None
    outlier_thresholds: MonthlyOutlierThresholds
    monthly_summary: MonthlySummary
    monthly_comparisons: MonthlyComparisons = Field(default_factory=MonthlyComparisons)
    fixed_variable: MonthlyFixedVariable
    category_deep: list[MonthlyCategoryDeep] = Field(default_factory=list)
    top_savable_categories: list[MonthlyCategoryDeep] = Field(default_factory=list)
    repeat_patterns: MonthlyRepeatPatterns
    weekly_trend: MonthlyWeeklyTrend
    micro_spending: MonthlyMicroSpending
    late_night_spending: MonthlyLateNightSpending
    high_spending: MonthlyHighSpending
    saving_potential: MonthlySavingPotential
    cash_flow_volatility: MonthlyCashFlowVolatility = Field(
        default_factory=MonthlyCashFlowVolatility
    )
    spending_concentration: MonthlySpendingConcentration = Field(
        default_factory=MonthlySpendingConcentration
    )
    frictionless_and_density: MonthlyFrictionlessAndDensity = Field(
        default_factory=MonthlyFrictionlessAndDensity
    )
    installment_debt_pressure: MonthlyInstallmentDebtPressure = Field(
        default_factory=MonthlyInstallmentDebtPressure
    )
    monthly_metrics: MonthlyMetrics = Field(default_factory=MonthlyMetrics)


class MonthlyCategoryChangeIndicator(BaseModel):
    """월간 카테고리 증감 지표와 원본 JSON 경로를 함께 표현한다."""

    category: str
    category_type: str
    total_amount: int
    prev_month_amount: int
    diff_amount: int
    diff_rate_percent: float
    direction: CategoryDirection
    source_json_path: str


class MonthlySpendingIndicatorPayload(BaseModel):
    """월간 해석 체인에 직접 전달할 JSON 기반 소비 지표 묶음을 표현한다."""

    member_id: int
    analysis_month: str
    prev_month: str
    metrics: list[SpendingMetric]
    category_changes: list[MonthlyCategoryChangeIndicator]
    largest_category_increase: MonthlyCategoryChangeIndicator | None
    largest_category_decrease: MonthlyCategoryChangeIndicator | None
    high_spending_items: list[MonthlyHighSpendingItem]
    fixed_items: list[MonthlyFixedItem]
    top_merchants: list[MonthlyMerchantVisit]
    trend_direction: str


class MonthlyFeedbackEvidence(BaseModel):
    """월간 소비 피드백 문장에 사용한 소비 JSON 또는 문서 근거를 표현한다."""

    evidence_type: MonthlyFeedbackEvidenceType
    title: str
    detail: str
    source_json_path: str | None = None
    source: str | None = None
    page_number: int | None = None


class MonthlyFeedbackAction(BaseModel):
    """월간 소비 피드백에서 사용자가 다음 달에 실행할 행동을 표현한다."""

    title: str
    detail: str
    target_json_path: str
    urgency: ActionUrgency
    related_source: str | None = None


class MonthlyFeedbackResult(BaseModel):
    """월간 분석 JSON과 RAG 근거를 바탕으로 생성한 최종 월간 소비 피드백을 표현한다."""

    summary_title: str
    feedback_message: str
    key_evidences: list[MonthlyFeedbackEvidence]
    action_items: list[MonthlyFeedbackAction]
    next_month_mission: str


class MonthlyFeedbackServiceResult(BaseModel):
    """월간 소비 피드백 서비스 실행 결과와 중간 산출물을 표현한다."""

    member_id: int
    analysis_month: str
    feedback: MonthlyFeedbackResult | None = None
    monthly_analysis: MonthlySpendingData | None = None
    interpretation_result: JsonObject | None = None
    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedAdviceContext] = Field(default_factory=list)
    error: str | None = None


__all__ = [
    "MonthlyOutlierThresholds",
    "MonthlySummary",
    "MonthlyPeriodComparison",
    "MonthlyReferenceMonth",
    "MonthlyAverageComparison",
    "MonthlyComparisons",
    "MonthlyFixedItem",
    "MonthlyFixedVariable",
    "MonthlyCategoryDeep",
    "MonthlyMerchantSummary",
    "MonthlyMerchantVisit",
    "MonthlyRepeatPatterns",
    "MonthlyWeeklyBreakdown",
    "MonthlyWeeklyTrend",
    "MonthlyCategoryCountAmount",
    "MonthlyMicroSpending",
    "MonthlyLateNightSpending",
    "MonthlyHighSpendingItem",
    "MonthlyHighSpending",
    "MonthlySavingPotential",
    "MonthlyCashFlowVolatility",
    "MonthlySpendingConcentration",
    "MonthlyFrictionlessSpending",
    "MonthlyTransactionDensity",
    "MonthlyFrictionlessAndDensity",
    "MonthlyInstallmentItem",
    "MonthlyInstallmentDebtPressure",
    "MonthlySpecialMetrics",
    "MonthlyMetrics",
    "MonthlySpendingData",
    "MonthlyCategoryChangeIndicator",
    "MonthlySpendingIndicatorPayload",
    "MonthlyFeedbackEvidence",
    "MonthlyFeedbackAction",
    "MonthlyFeedbackResult",
    "MonthlyFeedbackServiceResult",
]
