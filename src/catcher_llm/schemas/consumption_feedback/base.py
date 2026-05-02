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


class SpendingMetric(BaseModel):
    """JSON 데이터에서 바로 추출한 단일 소비 지표를 표현한다."""

    name: str
    value: MetricValue
    unit: str
    source_json_path: str
    description: str


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
    annual_income: float | None = None
    monthly_income: float | None = None
    target_max_spending_amount: float | None = None


__all__ = [
    "JsonScalar",
    "JsonValue",
    "JsonObject",
    "MetricValue",
    "CategoryDirection",
    "FindingConfidence",
    "ActionUrgency",
    "DailyFeedbackEvidenceType",
    "WeeklyFeedbackEvidenceType",
    "MonthlyFeedbackEvidenceType",
    "SourcePaths",
    "OutlierThresholds",
    "SpendingMetric",
    "EvidenceItem",
    "SpendingFinding",
    "RetrievedAdviceContext",
    "UserProfileContext",
]
