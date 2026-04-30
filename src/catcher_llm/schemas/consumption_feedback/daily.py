from __future__ import annotations

from pydantic import BaseModel, Field

from catcher_llm.schemas.consumption_feedback.base import (
    ActionUrgency,
    CategoryDirection,
    DailyFeedbackEvidenceType,
    JsonObject,
    JsonValue,
    OutlierThresholds,
    RetrievedAdviceContext,
    SourcePaths,
    SpendingMetric,
    UserProfileContext,
)


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
    user_feedback_memory: str | None = None
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


__all__ = [
    "CategoryRatioChange",
    "StableMetrics",
    "HighSpendingItem",
    "AnomalyDetection",
    "PreviousDayComparison",
    "DailySingleDateComparison",
    "DailyReferenceDay",
    "DailyAverageComparison",
    "DailyComparisons",
    "TimeSlotComparison",
    "TimeSlotAnalysis",
    "FrictionlessSpending",
    "TransactionDensity",
    "PaymentBehaviorAnalysis",
    "DailySpecialMetrics",
    "DailyMetrics",
    "UserSpendingData",
    "CategoryShiftIndicator",
    "MainCategoryShift",
    "SpendingIndicatorPayload",
    "DailyFeedbackSessionContext",
    "DailyFeedbackMemoryContext",
    "DailyFeedbackEvidence",
    "DailyFeedbackAction",
    "DailyFeedbackResult",
    "DailyFeedbackServiceResult",
]
