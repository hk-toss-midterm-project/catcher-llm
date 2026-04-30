from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type MetricValue = bool | int | float | str | None
type CategoryDirection = Literal["increase", "decrease", "flat"]
type FindingConfidence = Literal["low", "medium", "high"]
type ActionUrgency = Literal["immediate", "this_week", "this_month"]
type DailyFeedbackEvidenceType = Literal["spending_metric", "interpretation", "document"]
type WeeklyFeedbackEvidenceType = Literal["spending_metric", "interpretation", "document"]
type MonthlyFeedbackEvidenceType = Literal["spending_metric", "interpretation", "document"]


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


class DailySingleDateComparison(BaseModel):
    """일일 분석에서 기준일 하나와 오늘 소비를 비교한 결과를 표현한다."""

    label: str = ""
    reference_date: str = ""
    reference_total: int = 0
    today_total: int = 0
    amount_diff: int = 0
    amount_diff_rate_percent: float = 0.0
    reference_count: int = 0
    today_count: int = 0
    count_diff: int = 0
    reference_main_category: str | None = None
    today_main_category: str | None = None


class DailyReferenceDay(BaseModel):
    """최근 같은 요일 평균 계산에 포함된 일자별 소비를 표현한다."""

    date: str
    total: int
    transaction_count: int


class DailyAverageComparison(BaseModel):
    """일일 분석에서 여러 기준일 평균과 오늘 소비를 비교한 결과를 표현한다."""

    label: str = ""
    reference_dates: list[str] = Field(default_factory=list)
    reference_days: list[DailyReferenceDay] = Field(default_factory=list)
    reference_day_count: int = 0
    average_total: float = 0.0
    today_total: int = 0
    amount_diff: float = 0.0
    amount_diff_rate_percent: float = 0.0
    average_count: float = 0.0
    today_count: int = 0
    count_diff: float = 0.0
    today_main_category: str | None = None


class DailyComparisons(BaseModel):
    """일일 분석의 어제·지난주 같은 요일·최근 4주 같은 요일 평균 비교를 묶는다."""

    previous_day: DailySingleDateComparison = Field(default_factory=DailySingleDateComparison)
    same_weekday_last_week: DailySingleDateComparison = Field(
        default_factory=DailySingleDateComparison
    )
    recent_4week_same_weekday_average: DailyAverageComparison = Field(
        default_factory=DailyAverageComparison
    )


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


class FrictionlessSpending(BaseModel):
    """온라인/간편결제 기반 마찰력 없는 지출 지표를 표현한다."""

    keywords: list[str] = Field(default_factory=list)
    transaction_count: int = 0
    total_amount: int = 0
    ratio_percent: float = 0.0


class TransactionDensity(BaseModel):
    """당일 결제 빈도와 건당 평균 결제 금액 지표를 표현한다."""

    transaction_count: int = 0
    average_amount_per_transaction: float = 0.0


class PaymentBehaviorAnalysis(BaseModel):
    """당일 지출 마찰력과 결제 밀도 분석 결과를 표현한다."""

    frictionless_spending: FrictionlessSpending = Field(default_factory=FrictionlessSpending)
    transaction_density: TransactionDensity = Field(default_factory=TransactionDensity)


class DailySpecialMetrics(BaseModel):
    """문서 기준 일일 특수 지표를 표현한다."""

    impulse_spending_score: float = 0.0
    daily_spending_risk: float = 0.0


class DailyMetrics(BaseModel):
    """문서 기준 일일 소비 지표 묶음을 표현한다."""

    daily_total_amount: int = 0
    daily_transaction_count: int = 0
    daily_average_transaction_amount: float = 0.0
    daily_max_transaction_amount: int = 0
    time_slot_amounts: list[JsonValue] = Field(default_factory=list)
    late_night_ratio_percent: float = 0.0
    category_spending: list[JsonValue] = Field(default_factory=list)
    daily_budget_usage_rate_percent: float | None = None
    no_spending_day: bool = False
    daily_anomaly_score: float = 0.0
    special_metrics: DailySpecialMetrics = Field(default_factory=DailySpecialMetrics)


class UserSpendingData(BaseModel):
    """일일 소비 분석 JSON 전체 구조를 검증 가능한 입력 모델로 표현한다."""

    member_id: int
    analysis_date: str
    source_paths: SourcePaths
    outlier_thresholds: OutlierThresholds
    stable_metrics: StableMetrics
    anomaly_detection: AnomalyDetection
    previous_day_comparison: PreviousDayComparison
    daily_comparisons: DailyComparisons = Field(default_factory=DailyComparisons)
    time_slot_analysis: TimeSlotAnalysis
    payment_behavior_analysis: PaymentBehaviorAnalysis = Field(
        default_factory=PaymentBehaviorAnalysis
    )
    daily_metrics: DailyMetrics = Field(default_factory=DailyMetrics)


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


class RetrievedAdviceContext(BaseModel):
    """피드백 생성을 위해 RAG에서 찾은 조언 문서 청크와 유용성 판단 결과를 표현한다."""

    query: str
    source: str
    content: str
    page_number: int | None = None
    document_kind: str | None = None
    usefulness_score: float | None = None
    usefulness_reason: str | None = None


class UserProfileContext(BaseModel):
    """최종 피드백 개인화에 사용할 사용자 프로필 정보를 표현한다."""

    user_id: int
    name: str | None = None
    age: int | None = None
    job: str | None = None
    gender: str | None = None
    income: str | None = None
    region: str | None = None
    card_grade: str | None = None
    persona: str | None = None
    saving_goal_text: str | None = None


class DailyFeedbackSessionContext(BaseModel):
    """최종 피드백에 참고할 과거 일일 피드백 세션 기록을 표현한다."""

    analysis_date: str
    analysis_result: str | None = None
    feedback_reason: str | None = None
    todo_tomorrow: str | None = None


class DailyFeedbackMemoryContext(BaseModel):
    """장기 사용자 메모리와 최근 세션 기록을 묶은 최종 피드백 맥락을 표현한다."""

    user_id: int
    period_type: str = "daily"
    memory_summary: str | None = None
    recent_sessions: list[DailyFeedbackSessionContext] = Field(default_factory=list)


class DailyFeedbackEvidence(BaseModel):
    """일일 소비 피드백 문장에 사용한 소비 JSON 또는 문서 근거를 표현한다."""

    evidence_type: DailyFeedbackEvidenceType
    title: str
    detail: str
    source_json_path: str | None = None
    source: str | None = None
    page_number: int | None = None


class DailyFeedbackAction(BaseModel):
    """일일 소비 피드백에서 사용자가 실행할 단기 행동을 표현한다."""

    title: str
    detail: str
    target_json_path: str
    urgency: ActionUrgency
    related_source: str | None = None


class DailyFeedbackResult(BaseModel):
    """분석 JSON과 RAG 근거를 바탕으로 생성한 최종 일일 소비 피드백을 표현한다."""

    summary_title: str
    scolding_message: str
    key_evidences: list[DailyFeedbackEvidence]
    action_items: list[DailyFeedbackAction]
    tomorrow_mission: str


class DailyFeedbackServiceResult(BaseModel):
    """일일 소비 피드백 서비스 실행 결과와 중간 산출물을 표현한다."""

    member_id: int
    analysis_date: str
    feedback: DailyFeedbackResult | None = None
    daily_analysis: UserSpendingData | None = None
    interpretation_result: JsonObject | None = None
    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedAdviceContext] = Field(default_factory=list)
    error: str | None = None


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
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedAdviceContext] = Field(default_factory=list)
    error: str | None = None


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
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedAdviceContext] = Field(default_factory=list)
    error: str | None = None
