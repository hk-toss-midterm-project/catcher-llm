from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sqlalchemy import select

from catcher_llm.chains.consumption_feedback import (
    build_memory_summary_chain,
    build_monthly_feedback_chain,
    build_monthly_spending_analysis_chain,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import SessionModel, UserFeedbackMemoryModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    CategoryDirection,
    CauseAnalysisResult,
    DailyFeedbackMemoryContext,
    DailyFeedbackSessionContext,
    InterventionTarget,
    JsonObject,
    JsonScalar,
    JsonValue,
    MonthlyCategoryChangeIndicator,
    MonthlyFeedbackResult,
    MonthlyFeedbackServiceResult,
    MonthlyFixedItem,
    MonthlyHighSpendingItem,
    MonthlyMerchantVisit,
    MonthlySpendingData,
    MonthlySpendingIndicatorPayload,
    RetrievedAdviceContext,
    SpendingMetric,
    UserProfileContext,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    _compact_json_dumps,
    _select_compact_fields,
    _serialize_compact_advice_contexts,
    _serialize_compact_interpretation_result,
    _serialize_compact_memory_context,
    _serialize_compact_user_profile,
    load_user_profile_context,
    retrieve_feedback_contexts,
    serialize_advice_contexts,
    serialize_context_object,
    serialize_interpretation_result,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_feedback_reason_summary,
    get_category_direction,
    make_spending_metric,
    truncate_context_text,
)
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.ratio_guard import (
    coerce_ratio_context_message,
    collect_ratio_context_warnings,
)
from catcher_llm.services.consumption_feedback.timing import (
    FeedbackTimingCallback,
    run_timed_feedback_step,
)
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.user_data_service import ensure_user_database

_MONTHLY_MEMORY_PERIOD_TYPE = "monthly"
_DEFAULT_MONTHLY_MEMORY_SESSION_LIMIT = 6

_DEFAULT_MONTHLY_RETRIEVAL_QUERY = "월간 소비 절약 실천 방법"
_PERCENT_TEXT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
_GENERIC_MONTHLY_MISSION_TERMS: tuple[str, ...] = (
    "각각",
    "목표를 세",
    "목표를 설정",
    "줄이는 목표",
    "줄이기 목표",
    "절감 목표",
    "감축 목표",
)
_MONTHLY_FEEDBACK_DOCUMENT_KINDS: tuple[DocumentKind, ...] = (
    DocumentKind.USER_REPORT,
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK,
    DocumentKind.KCA_REPORT,
)
_COMPACT_MONTHLY_METRIC_KEYS: tuple[str, ...] = (
    "monthly_total_amount",
    "monthly_budget_usage_rate_percent",
    "monthly_remaining_budget",
    "monthly_overspend_amount",
    "monthly_income_usage_rate_percent",
    "target_spending_to_income_rate_percent",
    "estimated_saving_amount",
    "estimated_saving_rate_percent",
    "target_saving_amount",
    "target_saving_rate_percent",
    "previous_month_change_rate_percent",
    "fixed_cost_amount",
    "fixed_cost_ratio_percent",
    "variable_cost_amount",
    "subscription_total",
    "post_salary_spending_increase_rate_percent",
    "month_end_pressure_index",
    "special_metrics",
    "fixed_cost_burden_rate_percent",
    "spending_capacity",
    "nonessential_spending_income_rate_percent",
)


def _to_json_value(value: object) -> JsonValue:
    """Pydantic 모델과 파이썬 객체를 JSON 직렬화 가능한 값으로 변환한다."""
    if isinstance(value, BaseModel):
        return cast(JsonValue, json.loads(value.model_dump_json()))
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return cast(JsonScalar, value)
    return str(value)


def _to_json_object(value: object) -> JsonObject:
    """임의 객체를 최상위 JSON 객체로 변환한다."""
    json_value = _to_json_value(value)
    if isinstance(json_value, dict):
        return json_value
    return {"value": json_value}


def parse_monthly_spending_data(payload: object) -> MonthlySpendingData:
    """이미 메모리에 있는 월간 소비 분석 JSON 객체를 검증된 입력 모델로 변환한다."""
    return MonthlySpendingData.model_validate(payload)


def _compact_monthly_metrics(monthly_data: MonthlySpendingData) -> JsonObject:
    """최종 피드백에 필요한 월간 예산·소득·위험 지표만 추출한다."""
    raw_metrics = _to_json_object(monthly_data.monthly_metrics)
    return _select_compact_fields(raw_metrics, _COMPACT_MONTHLY_METRIC_KEYS)


def _compact_monthly_analysis(monthly_data: MonthlySpendingData) -> JsonObject:
    """최종 피드백용 월간 분석 JSON에서 출처와 임계값 같은 저활용 필드를 제외한다."""
    return {
        "member_id": monthly_data.member_id,
        "analysis_month": monthly_data.analysis_month,
        "prev_month": monthly_data.prev_month,
        "monthly_summary": _to_json_object(monthly_data.monthly_summary),
        "monthly_comparisons": _to_json_object(monthly_data.monthly_comparisons),
        "fixed_variable": _to_json_object(monthly_data.fixed_variable),
        "category_deep": _to_json_value(monthly_data.category_deep),
        "top_savable_categories": _to_json_value(monthly_data.top_savable_categories),
        "repeat_patterns": _to_json_object(monthly_data.repeat_patterns),
        "weekly_trend": _to_json_object(monthly_data.weekly_trend),
        "micro_spending": _to_json_object(monthly_data.micro_spending),
        "late_night_spending": _to_json_object(monthly_data.late_night_spending),
        "high_spending": _to_json_object(monthly_data.high_spending),
        "saving_potential": _to_json_object(monthly_data.saving_potential),
        "cash_flow_volatility": _to_json_object(monthly_data.cash_flow_volatility),
        "spending_concentration": _to_json_object(monthly_data.spending_concentration),
        "frictionless_and_density": _to_json_object(monthly_data.frictionless_and_density),
        "installment_debt_pressure": _to_json_object(monthly_data.installment_debt_pressure),
        "monthly_metrics": _compact_monthly_metrics(monthly_data),
    }


def _build_monthly_category_change_indicators(
    monthly_data: MonthlySpendingData,
) -> list[MonthlyCategoryChangeIndicator]:
    """월간 카테고리 요약을 JSON 경로가 포함된 증감 지표 목록으로 변환한다."""
    indicators: list[MonthlyCategoryChangeIndicator] = []
    for index, category in enumerate(monthly_data.category_deep):
        direction: CategoryDirection = get_category_direction(float(category.diff_amount))
        indicators.append(
            MonthlyCategoryChangeIndicator(
                category=category.category,
                category_type=category.type,
                total_amount=category.total_amount,
                prev_month_amount=category.prev_month_amount,
                diff_amount=category.diff_amount,
                diff_rate_percent=category.diff_rate_percent,
                direction=direction,
                source_json_path=f"category_deep[{index}]",
            )
        )
    return indicators


def _get_largest_monthly_category_increase(
    changes: list[MonthlyCategoryChangeIndicator],
) -> MonthlyCategoryChangeIndicator | None:
    """전월 대비 지출이 가장 크게 증가한 월간 카테고리 지표를 찾는다."""
    increased_changes = [change for change in changes if change.diff_amount > 0]
    if not increased_changes:
        return None
    return max(increased_changes, key=lambda change: change.diff_amount)


def _get_largest_monthly_category_decrease(
    changes: list[MonthlyCategoryChangeIndicator],
) -> MonthlyCategoryChangeIndicator | None:
    """전월 대비 지출이 가장 크게 감소한 월간 카테고리 지표를 찾는다."""
    decreased_changes = [change for change in changes if change.diff_amount < 0]
    if not decreased_changes:
        return None
    return min(decreased_changes, key=lambda change: change.diff_amount)


def _build_monthly_core_metrics(monthly_data: MonthlySpendingData) -> list[SpendingMetric]:
    """월간 분석에 자주 쓰는 핵심 소비 지표를 원본 JSON에서 직접 추출한다."""
    monthly_summary = monthly_data.monthly_summary
    fixed_variable = monthly_data.fixed_variable
    repeat_patterns = monthly_data.repeat_patterns
    saving_potential = monthly_data.saving_potential
    monthly_metrics = monthly_data.monthly_metrics
    recent_average = monthly_data.monthly_comparisons.recent_3month_average

    return [
        make_spending_metric(
            "이번 달 총 지출액",
            monthly_summary.this_month_total,
            "KRW",
            "monthly_summary.this_month_total",
            "분석 월의 총 소비 금액",
        ),
        make_spending_metric(
            "전월 대비 지출 증감액",
            monthly_summary.amount_diff,
            "KRW",
            "monthly_summary.amount_diff",
            "전월 총 지출과 이번 달 총 지출의 차이",
        ),
        make_spending_metric(
            "전월 대비 지출 증감률",
            monthly_summary.diff_rate_percent,
            "percent",
            "monthly_summary.diff_rate_percent",
            "전월 총 지출 대비 이번 달 지출 증감률",
        ),
        make_spending_metric(
            "최근 3개월 평균 대비 지출 증감액",
            recent_average.amount_diff,
            "KRW",
            "monthly_comparisons.recent_3month_average.amount_diff",
            "최근 3개월 평균 지출과 분석 월 총 지출의 차이",
        ),
        make_spending_metric(
            "최근 3개월 평균 대비 지출 증감률",
            recent_average.amount_diff_rate_percent,
            "percent",
            "monthly_comparisons.recent_3month_average.amount_diff_rate_percent",
            "최근 3개월 평균 지출 대비 분석 월 지출 증감률",
        ),
        make_spending_metric(
            "월간 결제 건수",
            monthly_summary.transaction_count,
            "count",
            "monthly_summary.transaction_count",
            "분석 월 전체 결제 건수",
        ),
        make_spending_metric(
            "고정비 총액",
            fixed_variable.fixed_total,
            "KRW",
            "fixed_variable.fixed_total",
            "자동이체 등 고정비로 분류된 월간 결제 금액 합계",
        ),
        make_spending_metric(
            "고정비 비중",
            fixed_variable.fixed_ratio_percent,
            "percent",
            "fixed_variable.fixed_ratio_percent",
            "이번 달 총 지출 중 고정비 비중",
        ),
        make_spending_metric(
            "배달 결제 금액",
            repeat_patterns.delivery.total_amount,
            "KRW",
            "repeat_patterns.delivery.total_amount",
            "배달 키워드가 포함된 월간 결제 금액 합계",
        ),
        make_spending_metric(
            "카페 결제 건수",
            repeat_patterns.cafe.count,
            "count",
            "repeat_patterns.cafe.count",
            "카페 키워드가 포함된 월간 결제 건수",
        ),
        make_spending_metric(
            "주차별 소비 추이",
            monthly_data.weekly_trend.trend_direction,
            "trend",
            "weekly_trend.trend_direction",
            "월 초 대비 월 후반 지출 추이 방향",
        ),
        make_spending_metric(
            "소액 결제 건수",
            monthly_data.micro_spending.count,
            "count",
            "micro_spending.count",
            "소액 결제 기준 미만의 월간 결제 건수",
        ),
        make_spending_metric(
            "야간 소비 금액",
            monthly_data.late_night_spending.total_amount,
            "KRW",
            "late_night_spending.total_amount",
            "21시 이후 발생한 월간 결제 금액 합계",
        ),
        make_spending_metric(
            "고액 결제 건수",
            monthly_data.high_spending.count,
            "count",
            "high_spending.items",
            "IQR 상한선을 초과한 월간 결제 항목 수",
        ),
        make_spending_metric(
            "총 절약 가능액",
            saving_potential.total_potential_saving,
            "KRW",
            "saving_potential.total_potential_saving",
            "월간 분석에서 추정한 절약 가능 금액 합계",
        ),
        make_spending_metric(
            "다음 달 권장 목표액",
            saving_potential.next_month_recommended_target,
            "KRW",
            "saving_potential.next_month_recommended_target",
            "이번 달 총 지출의 90%로 계산한 다음 달 권장 목표액",
        ),
        make_spending_metric(
            "평일 vs 주말 소비 변동성(CV)",
            monthly_data.cash_flow_volatility.cv_index,
            "ratio",
            "cash_flow_volatility.cv_index",
            "주차별 소비의 변동 계수 (0에 가까울수록 일정한 페이스)",
        ),
        make_spending_metric(
            "월간 페이스 진단",
            monthly_data.cash_flow_volatility.pace_status,
            "status",
            "cash_flow_volatility.pace_status",
            "주차별 소비 편차 수준에 따른 안정/주의/위험 진단",
        ),
        make_spending_metric(
            "변동비 지출 쏠림 1위 카테고리",
            monthly_data.spending_concentration.top_1_category or "",
            "category",
            "spending_concentration.top_1_category",
            "필수 지출 제외 변동비 중 가장 많은 비중을 차지하는 카테고리",
        ),
        make_spending_metric(
            "변동비 지출 쏠림 진단",
            monthly_data.spending_concentration.concentration_status,
            "status",
            "spending_concentration.concentration_status",
            "파레토 분석 기반 지출 쏠림 정도 진단 (예: 극심한 쏠림, 분산 소비)",
        ),
        make_spending_metric(
            "간편결제 비중",
            monthly_data.frictionless_and_density.frictionless_spending.ratio_percent,
            "percent",
            "frictionless_and_density.frictionless_spending.ratio_percent",
            "온라인/간편결제 등 마찰력 없는 지출이 전체에서 차지하는 비율",
        ),
        make_spending_metric(
            "할부 결제 비중",
            monthly_data.installment_debt_pressure.installment_ratio_percent,
            "percent",
            "installment_debt_pressure.installment_ratio_percent",
            "이번 달 지출 중 할부 결제가 차지하는 비율",
        ),
        make_spending_metric(
            "월간 예산 대비 사용률",
            monthly_metrics.monthly_budget_usage_rate_percent,
            "percent",
            "monthly_metrics.monthly_budget_usage_rate_percent",
            "설정된 월간 예산 대비 분석 월 소비 금액 비율",
        ),
        make_spending_metric(
            "월간 잔여 예산",
            monthly_metrics.monthly_remaining_budget,
            "KRW",
            "monthly_metrics.monthly_remaining_budget",
            "월 목표 소비 한도에서 분석 월 소비를 제외하고 남은 금액",
        ),
        make_spending_metric(
            "월간 예산 초과액",
            monthly_metrics.monthly_overspend_amount,
            "KRW",
            "monthly_metrics.monthly_overspend_amount",
            "월 목표 소비 한도를 초과해 사용한 금액",
        ),
        make_spending_metric(
            "월소득 대비 총소비율",
            monthly_metrics.monthly_income_usage_rate_percent,
            "percent",
            "monthly_metrics.monthly_income_usage_rate_percent",
            "월소득 대비 분석 월 총 소비 금액 비율",
        ),
        make_spending_metric(
            "목표 소비 한도 소득 비중",
            monthly_metrics.target_spending_to_income_rate_percent,
            "percent",
            "monthly_metrics.target_spending_to_income_rate_percent",
            "월 목표 소비 한도가 월소득에서 차지하는 비율",
        ),
        make_spending_metric(
            "추정 저축액",
            monthly_metrics.estimated_saving_amount,
            "KRW",
            "monthly_metrics.estimated_saving_amount",
            "월소득에서 분석 월 총 소비를 제외한 추정 잔여 금액",
        ),
        make_spending_metric(
            "추정 저축률",
            monthly_metrics.estimated_saving_rate_percent,
            "percent",
            "monthly_metrics.estimated_saving_rate_percent",
            "월소득 대비 추정 저축액 비율",
        ),
        make_spending_metric(
            "목표 달성 시 저축액",
            monthly_metrics.target_saving_amount,
            "KRW",
            "monthly_metrics.target_saving_amount",
            "월 목표 소비 한도를 지켰을 때 월소득에서 남는 금액",
        ),
        make_spending_metric(
            "목표 달성 시 저축률",
            monthly_metrics.target_saving_rate_percent,
            "percent",
            "monthly_metrics.target_saving_rate_percent",
            "월소득 대비 목표 달성 시 저축액 비율",
        ),
        make_spending_metric(
            "구독료 합계",
            monthly_metrics.subscription_total,
            "KRW",
            "monthly_metrics.subscription_total",
            "구독성 결제로 분류된 월간 소비 금액 합계",
        ),
        make_spending_metric(
            "고정비 부담률",
            monthly_metrics.fixed_cost_burden_rate_percent,
            "percent",
            "monthly_metrics.fixed_cost_burden_rate_percent",
            "월 소득 대비 고정비 지출 비중",
        ),
        make_spending_metric(
            "소비 여력",
            monthly_metrics.spending_capacity,
            "KRW",
            "monthly_metrics.spending_capacity",
            "월소득에서 고정비와 필수 변동비를 제외한 소비 가능 금액",
        ),
        make_spending_metric(
            "비필수 소비 소득 비중",
            monthly_metrics.nonessential_spending_income_rate_percent,
            "percent",
            "monthly_metrics.nonessential_spending_income_rate_percent",
            "월소득 대비 비필수 카테고리 소비 금액 비율",
        ),
        make_spending_metric(
            "급여일 이후 소비 증가율",
            monthly_metrics.post_salary_spending_increase_rate_percent,
            "percent",
            "monthly_metrics.post_salary_spending_increase_rate_percent",
            "급여일 이후 일평균 소비가 급여일 이전보다 증가한 비율",
        ),
        make_spending_metric(
            "월말 소비 압박 지수",
            monthly_metrics.month_end_pressure_index,
            "ratio",
            "monthly_metrics.month_end_pressure_index",
            "월말 일평균 소비가 월말 이전 일평균 소비 대비 얼마나 커졌는지 나타내는 배율",
        ),
    ]


def extract_monthly_spending_indicators(
    monthly_data: MonthlySpendingData,
) -> MonthlySpendingIndicatorPayload:
    """월간 소비 분석 모델에서 해석 체인에 넣을 핵심 지표 묶음을 추출한다."""
    category_changes = _build_monthly_category_change_indicators(monthly_data)
    return MonthlySpendingIndicatorPayload(
        member_id=monthly_data.member_id,
        analysis_month=monthly_data.analysis_month,
        prev_month=monthly_data.prev_month,
        metrics=_build_monthly_core_metrics(monthly_data),
        category_changes=category_changes,
        largest_category_increase=_get_largest_monthly_category_increase(category_changes),
        largest_category_decrease=_get_largest_monthly_category_decrease(category_changes),
        high_spending_items=monthly_data.high_spending.items,
        fixed_items=monthly_data.fixed_variable.fixed_items,
        top_merchants=monthly_data.repeat_patterns.top5_merchants,
        trend_direction=monthly_data.weekly_trend.trend_direction,
    )


def make_monthly_spending_analysis_input(
    monthly_data: MonthlySpendingData,
    *,
    user_profile: object | None = None,
) -> dict[str, str]:
    """월간 해석 체인에 전달할 원본 JSON, 추출 지표 JSON, 사용자 프로필 JSON 입력을 만든다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    return {
        "raw_json": monthly_data.model_dump_json(),
        "indicator_json": indicators.model_dump_json(),
        "user_profile_json": serialize_context_object(user_profile or {}),
    }


def _append_unique_query(queries: list[str], query: str) -> None:
    """비어 있지 않고 아직 없는 RAG 검색 질의만 목록에 추가한다."""
    normalized_query = " ".join(query.split())
    if normalized_query and normalized_query not in queries:
        queries.append(normalized_query)


def _iter_action_missions(action_result: object) -> list[ActionMission]:
    """월간 해석 결과의 개선 후보 모델 또는 dict에서 RAG 검색 후보 목록을 추출한다."""
    if isinstance(action_result, ActionAnalysisResult):
        return [
            *action_result.immediate_cuts,
            *action_result.substitution_opportunities,
            *action_result.budget_control_areas,
            *action_result.next_week_missions,
        ]
    if not isinstance(action_result, dict):
        return []

    missions: list[ActionMission] = []
    for key in (
        "immediate_cuts",
        "substitution_opportunities",
        "budget_control_areas",
        "next_week_missions",
    ):
        raw_items = action_result.get(key)
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            try:
                missions.append(ActionMission.model_validate(raw_item))
            except ValueError:
                continue
    return missions


def _iter_intervention_targets(cause_result: object) -> list[InterventionTarget]:
    """월간 원인 해석 모델 또는 dict에서 RAG 검색용 개입 타겟 후보를 추출한다."""
    if isinstance(cause_result, CauseAnalysisResult):
        return list(cause_result.intervention_targets)
    if not isinstance(cause_result, dict):
        return []

    raw_items = cause_result.get("intervention_targets")
    if not isinstance(raw_items, list):
        return []

    targets: list[InterventionTarget] = []
    for raw_item in raw_items:
        try:
            targets.append(InterventionTarget.model_validate(raw_item))
        except ValueError:
            continue
    return targets


def _intervention_target_query_text(target: InterventionTarget) -> str:
    """월간 개입 타겟 후보에서 RAG 검색에 사용할 질의 문구를 선택한다."""
    return target.query_hint or target.title


def _append_monthly_high_spending_queries(
    queries: list[str],
    items: Sequence[MonthlyHighSpendingItem],
) -> None:
    """고액 결제 항목을 기반으로 월간 피드백용 RAG 검색 질의를 추가한다."""
    for item in sorted(
        items, key=lambda high_spending_item: high_spending_item.amount, reverse=True
    ):
        _append_unique_query(queries, f"{item.category} {item.merchant} 지출 줄이는 방법")


def _append_fixed_item_queries(
    queries: list[str],
    items: Sequence[MonthlyFixedItem],
) -> None:
    """고정비 항목을 기반으로 월간 피드백용 RAG 검색 질의를 추가한다."""
    for item in sorted(items, key=lambda fixed_item: fixed_item.total_amount, reverse=True)[:2]:
        _append_unique_query(queries, f"{item.merchant} 고정비 절약 방법")


def _append_top_merchant_queries(
    queries: list[str],
    merchants: Sequence[MonthlyMerchantVisit],
) -> None:
    """반복 가맹점 정보를 기반으로 월간 피드백용 RAG 검색 질의를 추가한다."""
    for merchant in sorted(merchants, key=lambda item: item.visit_count, reverse=True)[:2]:
        if merchant.visit_count < 2:
            continue
        _append_unique_query(queries, f"{merchant.merchant} 반복 소비 줄이는 방법")


def _append_monthly_report_category_candidates(
    categories: list[str],
    monthly_data: MonthlySpendingData,
    indicators: MonthlySpendingIndicatorPayload,
) -> None:
    """월간 USER_REPORT 비교 질의에 사용할 개인 소비 카테고리 후보를 우선순위대로 추가한다."""
    if indicators.largest_category_increase is not None:
        _append_unique_query(categories, indicators.largest_category_increase.category)
    if indicators.largest_category_decrease is not None:
        _append_unique_query(categories, indicators.largest_category_decrease.category)

    top_1_category = monthly_data.spending_concentration.top_1_category
    if top_1_category:
        _append_unique_query(categories, top_1_category)

    for category in sorted(
        monthly_data.top_savable_categories,
        key=lambda item: item.total_amount,
        reverse=True,
    ):
        _append_unique_query(categories, category.category)

    for item in sorted(
        indicators.high_spending_items,
        key=lambda high_spending_item: high_spending_item.amount,
        reverse=True,
    ):
        _append_unique_query(categories, item.category)


def _build_user_report_monthly_queries(
    monthly_data: MonthlySpendingData,
    *,
    max_queries: int,
) -> list[str]:
    """월간 USER_REPORT 카드 검색에 맞는 전체 사용자 비교 근거 질의를 생성한다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    analysis_month = monthly_data.analysis_month
    categories: list[str] = []
    _append_monthly_report_category_candidates(categories, monthly_data, indicators)

    queries: list[str] = []
    category_query_limit = max(1, max_queries - 2)
    for category in categories[:category_query_limit]:
        _append_unique_query(
            queries,
            f"{analysis_month} {category} 카테고리 전체 사용자 월간 소비 비중 전월 대비 변화",
        )

    _append_unique_query(
        queries,
        f"{analysis_month} 전체 사용자 결제 행동 온라인 할부 야간 마찰없는 결제 비중",
    )
    _append_unique_query(
        queries,
        f"{analysis_month} 전체 사용자 목표 사용률 초과 분포 예산 초과 위험군",
    )
    return queries[:max_queries]


def _build_monthly_trend_categories(
    monthly_data: MonthlySpendingData,
    indicators: MonthlySpendingIndicatorPayload,
    *,
    max_categories: int,
) -> list[str]:
    """월간 소비 동향 문서 질의에 넣을 개인 소비 카테고리 후보를 고른다."""
    categories: list[str] = []
    _append_monthly_report_category_candidates(categories, monthly_data, indicators)

    for category in sorted(
        monthly_data.category_deep,
        key=lambda category_item: category_item.total_amount,
        reverse=True,
    ):
        _append_unique_query(categories, category.category)

    return categories[:max_categories]


def _build_consumption_benchmark_monthly_queries(
    monthly_data: MonthlySpendingData,
    *,
    max_queries: int,
) -> list[str]:
    """CATCHER 카드 결제 소비 동향 보고서에 맞는 월간 벤치마크 질의를 생성한다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    categories = _build_monthly_trend_categories(
        monthly_data,
        indicators,
        max_categories=max(1, max_queries - 1),
    )

    queries: list[str] = []
    for category in categories:
        _append_unique_query(
            queries,
            f"{category} 카테고리 카드 결제 소비 동향 벤치마크",
        )

    _append_unique_query(
        queries,
        "월간 카드 결제 소비 동향 온라인 할부 야간 결제 패턴",
    )
    _append_unique_query(
        queries,
        "카테고리별 카드 결제 소비 패턴 비교",
    )
    return queries[:max_queries]


def _build_kca_monthly_trend_queries(
    monthly_data: MonthlySpendingData,
    *,
    max_queries: int,
) -> list[str]:
    """한국 소비자원 소비 동향 보고서에 맞는 월간 카테고리 동향 질의를 생성한다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    categories = _build_monthly_trend_categories(
        monthly_data,
        indicators,
        max_categories=max(1, max_queries - 1),
    )

    queries: list[str] = []
    for category in categories:
        _append_unique_query(
            queries,
            f"{category} 카테고리 한국 소비자원 소비 동향",
        )

    _append_unique_query(
        queries,
        "한국 소비자원 월간 소비 동향 카테고리 변화",
    )
    _append_unique_query(
        queries,
        "소비자원 결제 행태 소비 동향 카테고리별 변화",
    )
    return queries[:max_queries]


def build_monthly_feedback_retrieval_queries(
    monthly_data: MonthlySpendingData,
    *,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    max_queries: int = 4,
) -> list[str]:
    """월간 분석, 해석 결과, 사용자 프로필에서 최종 피드백용 RAG 검색 질의를 생성한다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    queries: list[str] = []

    if indicators.largest_category_increase is not None:
        category = indicators.largest_category_increase.category
        _append_unique_query(queries, f"{category} 월간 소비 절약 방법")

    _append_fixed_item_queries(queries, indicators.fixed_items)
    _append_monthly_high_spending_queries(queries, indicators.high_spending_items)

    cause_result = (interpretation_result or {}).get("cause_result")
    for target in _iter_intervention_targets(cause_result):
        _append_unique_query(queries, f"{_intervention_target_query_text(target)} 절약 방법")

    action_result = (interpretation_result or {}).get("action_result")
    for mission in _iter_action_missions(action_result):
        _append_unique_query(queries, f"{mission.title} 실천 방법")

    if user_profile is not None:
        if user_profile.saving_goal_text:
            _append_unique_query(
                queries, f"{user_profile.saving_goal_text} 목표 월간 소비 절약 방법"
            )
        if user_profile.job and indicators.largest_category_increase is not None:
            category = indicators.largest_category_increase.category
            _append_unique_query(queries, f"{user_profile.job} {category} 소비 줄이는 방법")
        if user_profile.persona:
            _append_unique_query(queries, f"{user_profile.persona} 월간 소비 습관 개선 방법")

    _append_top_merchant_queries(queries, indicators.top_merchants)

    # 파레토 지출 쏠림: 1위 변동비 카테고리 대상 쿼리
    top_1_cat = monthly_data.spending_concentration.top_1_category
    if top_1_cat:
        _append_unique_query(queries, f"{top_1_cat} 지출 줄이는 방법")

    _append_unique_query(queries, _DEFAULT_MONTHLY_RETRIEVAL_QUERY)
    return queries[:max_queries]


def build_monthly_feedback_retrieval_queries_by_document_kind(
    monthly_data: MonthlySpendingData,
    *,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    max_queries: int = 4,
) -> dict[DocumentKind, list[str]]:
    """월간 피드백 RAG 검색 질의를 문서 종류별 역할에 맞게 분리해 생성한다."""
    user_report_queries = _build_user_report_monthly_queries(
        monthly_data,
        max_queries=max_queries,
    )
    benchmark_queries = _build_consumption_benchmark_monthly_queries(
        monthly_data,
        max_queries=max_queries,
    )
    kca_queries = _build_kca_monthly_trend_queries(
        monthly_data,
        max_queries=max_queries,
    )
    return {
        DocumentKind.USER_REPORT: user_report_queries,
        DocumentKind.CATCHER_CONSUMPTION_BENCHMARK: benchmark_queries,
        DocumentKind.KCA_REPORT: kca_queries,
    }


def _flatten_monthly_retrieval_query_plan(
    query_plan: dict[DocumentKind, list[str]],
) -> list[str]:
    """문서 종류별 월간 RAG 질의 계획을 화면 표시와 결과 저장용 단일 목록으로 펼친다."""
    queries: list[str] = []
    for document_kind, document_queries in query_plan.items():
        for query in document_queries:
            _append_unique_query(queries, f"[{document_kind.value}] {query}")
    return queries


def _is_generic_percent_reduction_mission(mission: str) -> bool:
    """월간 미션이 구체 행동이 아닌 비율 감축 목표형 문장인지 판단한다."""
    normalized_mission = " ".join(mission.split())
    has_percent = _PERCENT_TEXT_PATTERN.search(normalized_mission) is not None
    if has_percent and any(term in normalized_mission for term in ("줄", "절감", "감축")):
        return True
    return any(term in normalized_mission for term in _GENERIC_MONTHLY_MISSION_TERMS)


def _select_monthly_mission_category(monthly_data: MonthlySpendingData) -> str:
    """구체 월간 미션에 사용할 우선 점검 카테고리를 선택한다."""
    indicators = extract_monthly_spending_indicators(monthly_data)
    if indicators.largest_category_increase is not None:
        return indicators.largest_category_increase.category

    if monthly_data.spending_concentration.top_1_category:
        return monthly_data.spending_concentration.top_1_category

    if monthly_data.top_savable_categories:
        return monthly_data.top_savable_categories[0].category

    if monthly_data.category_deep:
        largest_category = max(
            monthly_data.category_deep,
            key=lambda category: category.total_amount,
        )
        return largest_category.category

    return "변동비"


def _select_monthly_mission_high_spending_item(
    monthly_data: MonthlySpendingData,
    category: str,
) -> MonthlyHighSpendingItem | None:
    """우선 카테고리와 연결된 고액 결제 항목을 월간 미션 근거로 선택한다."""
    sorted_items = sorted(
        monthly_data.high_spending.items,
        key=lambda item: item.amount,
        reverse=True,
    )
    for item in sorted_items:
        if item.category == category:
            return item
    return sorted_items[0] if sorted_items else None


def _build_concrete_monthly_mission(monthly_data: MonthlySpendingData) -> str:
    """비율 감축 목표를 대신할 결제 전 확인 중심의 구체 월간 미션을 만든다."""
    category = _select_monthly_mission_category(monthly_data)
    high_spending_item = _select_monthly_mission_high_spending_item(monthly_data, category)
    if high_spending_item is not None:
        merchant = high_spending_item.merchant
        return (
            f"다음 달 첫째 주에는 {category} 결제 전에 {merchant} 같은 고액 결제를 "
            "하루 보류하고 필수 구매 목록에 있는지 확인합니다."
        )

    return (
        f"다음 달 첫째 주에는 {category} 결제 전 필수 구매 목록을 먼저 적고, "
        "목록 밖 결제는 하루 보류한 뒤 다시 확인합니다."
    )


def _sanitize_monthly_feedback_result(
    feedback: MonthlyFeedbackResult,
    monthly_data: MonthlySpendingData,
) -> MonthlyFeedbackResult:
    """최종 월간 피드백에서 목표형 미션을 실행 가능한 행동 미션으로 보정한다."""
    if _is_generic_percent_reduction_mission(feedback.next_month_mission):
        feedback.next_month_mission = _build_concrete_monthly_mission(monthly_data)
    return feedback


def make_monthly_feedback_input(
    *,
    monthly_data: MonthlySpendingData,
    interpretation_result: dict[str, object],
    advice_contexts: Sequence[RetrievedAdviceContext],
    user_profile: object | None = None,
    memory_context: object | None = None,
    compact_input: bool = True,
) -> dict[str, str]:
    """최종 월간 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    if not compact_input:
        return {
            "monthly_json": monthly_data.model_dump_json(),
            "interpretation_json": serialize_interpretation_result(interpretation_result),
            "retrieved_contexts": serialize_advice_contexts(advice_contexts),
            "user_profile_json": serialize_context_object(user_profile or {}),
            "memory_context_json": serialize_context_object(memory_context or {}),
        }

    return {
        "monthly_json": _compact_json_dumps(_compact_monthly_analysis(monthly_data)),
        "interpretation_json": _serialize_compact_interpretation_result(interpretation_result),
        "retrieved_contexts": _serialize_compact_advice_contexts(advice_contexts),
        "user_profile_json": _serialize_compact_user_profile(user_profile or {}),
        "memory_context_json": _serialize_compact_memory_context(memory_context or {}),
    }


def save_monthly_feedback_session(
    *,
    member_id: int,
    analysis_month: str,
    monthly_analysis: MonthlySpendingData,
    feedback: MonthlyFeedbackResult,
    settings: Settings | None = None,
) -> None:
    """최종 월간 피드백 실행 결과를 session 테이블에 해당 월 기준으로 저장하거나 갱신한다."""
    config = settings or get_settings()
    ensure_user_database(config)
    feedback_reason = json.dumps(
        [evidence.model_dump() for evidence in feedback.key_evidences],
        ensure_ascii=False,
    )

    with session_scope(config) as session:
        session_row = session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == analysis_month,
                SessionModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
            )
        )
        if session_row is None:
            session_row = SessionModel(
                user_id=member_id,
                analysis_date=analysis_month,
                period_type=_MONTHLY_MEMORY_PERIOD_TYPE,
            )
            session.add(session_row)

        session_row.analysis_result = monthly_analysis.model_dump_json()
        session_row.feedback_message = feedback.feedback_message
        session_row.feedback_reason = feedback_reason
        session_row.todo_tomorrow = feedback.next_month_mission


def load_monthly_session_for_date(
    *,
    member_id: int,
    analysis_month: str,
    settings: Settings | None = None,
) -> SessionModel | None:
    """session 테이블에서 특정 월의 monthly 세션을 조회한다. 없으면 None을 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        return session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == analysis_month,
                SessionModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
            )
        )


def load_monthly_feedback_memory_context(
    *,
    member_id: int,
    analysis_month: str | None = None,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_MONTHLY_MEMORY_SESSION_LIMIT,
) -> DailyFeedbackMemoryContext:
    """SQLite user_memories와 최근 월간 session에서 최종 피드백용 과거 맥락을 조회한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
            )
        )
        session_statement = select(SessionModel).where(
            SessionModel.user_id == member_id,
            SessionModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
        )
        if analysis_month is not None:
            session_statement = session_statement.where(SessionModel.analysis_date < analysis_month)
        recent_sessions = list(
            session.scalars(
                session_statement.order_by(
                    SessionModel.analysis_date.desc(),
                    SessionModel.id.desc(),
                ).limit(recent_session_limit)
            )
        )

    chronological_sessions = list(reversed(recent_sessions))

    with session_scope(config) as _fb_session:
        fb_rows = list(
            _fb_session.scalars(
                select(UserFeedbackMemoryModel)
                .where(
                    UserFeedbackMemoryModel.user_id == member_id,
                    UserFeedbackMemoryModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
                )
                .order_by(UserFeedbackMemoryModel.created_at.asc())
            )
        )
    feedback_memory_text: str | None = (
        "\n".join(r.reason for r in fb_rows) + "\n" if fb_rows else None
    )

    return DailyFeedbackMemoryContext(
        user_id=member_id,
        period_type=_MONTHLY_MEMORY_PERIOD_TYPE,
        memory_summary=memory.summary if memory is not None else None,
        user_feedback_memory=feedback_memory_text,
        recent_sessions=[
            DailyFeedbackSessionContext(
                analysis_date=item.analysis_date,
                analysis_result=item.analysis_result,
                feedback_reason=item.feedback_reason,
                todo_tomorrow=item.todo_tomorrow,
            )
            for item in chronological_sessions
        ],
    )


def _extract_monthly_total_summary(analysis_result: str | None) -> str:
    """저장된 월간 분석 JSON에서 해당 월 총 지출액을 짧은 문자열로 추출한다."""
    if analysis_result is None:
        return "-"
    try:
        raw_analysis = json.loads(analysis_result)
    except json.JSONDecodeError:
        return "-"

    if not isinstance(raw_analysis, dict):
        return "-"
    monthly_summary = raw_analysis.get("monthly_summary")
    if not isinstance(monthly_summary, dict):
        return "-"
    this_month_total = monthly_summary.get("this_month_total")
    if not isinstance(this_month_total, int | float):
        return "-"
    return f"{this_month_total:,.0f}원"


def _build_monthly_session_list_text(sessions: Sequence[SessionModel]) -> str:
    """월간 세션 목록을 LLM 요약 체인에 넣을 텍스트 목록으로 직렬화한다."""
    lines = []
    for session_row in sessions:
        monthly_total = _extract_monthly_total_summary(session_row.analysis_result)
        reason_summary = extract_feedback_reason_summary(session_row.feedback_reason)
        next_mission = truncate_context_text(session_row.todo_tomorrow)
        line = (
            f"- {session_row.analysis_date}: "
            f"월간지출 {monthly_total}; "
            f"핵심근거 {reason_summary}; "
            f"다음미션 {next_mission}"
        )
        lines.append(line)
    return "\n".join(lines)


def build_monthly_memory_summary_from_sessions(
    sessions: Sequence[SessionModel],
    settings: Settings | None = None,
) -> str:
    """최근 월간 피드백 세션 목록을 LLM으로 통합 요약한 문자열로 만든다."""
    if not sessions:
        return "아직 누적된 월간 피드백 세션이 없습니다."
    session_list = _build_monthly_session_list_text(sessions)
    chain = build_memory_summary_chain("월간", settings)
    try:
        return chain.invoke({"session_list": session_list})
    except Exception:
        return session_list


def refresh_monthly_user_memory(
    *,
    member_id: int,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_MONTHLY_MEMORY_SESSION_LIMIT,
) -> str:
    """최근 월간 session 기록을 바탕으로 user_memories의 monthly 요약을 생성하거나 갱신한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        recent_sessions = list(
            session.scalars(
                select(SessionModel)
                .where(
                    SessionModel.user_id == member_id,
                    SessionModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
                )
                .order_by(SessionModel.analysis_date.desc(), SessionModel.id.desc())
                .limit(recent_session_limit)
            )
        )
        summary = build_monthly_memory_summary_from_sessions(
            list(reversed(recent_sessions)), config
        )

        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
            )
        )
        if memory is None:
            memory = UserMemoryModel(
                user_id=member_id,
                period_type=_MONTHLY_MEMORY_PERIOD_TYPE,
                summary=summary,
            )
            session.add(memory)
        else:
            memory.summary = summary

    return summary


def _refresh_monthly_user_memory_if_possible(
    *,
    member_id: int,
    settings: Settings | None = None,
) -> None:
    """피드백 생성 성공 이후 월간 메모리 요약을 가능할 때만 갱신한다."""
    try:
        refresh_monthly_user_memory(member_id=member_id, settings=settings)
    except Exception:
        return


def _build_error_result(
    *,
    member_id: int,
    analysis_month: str,
    error: str,
    monthly_analysis: MonthlySpendingData | None = None,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    memory_context: DailyFeedbackMemoryContext | None = None,
    retrieval_queries: Sequence[str] | None = None,
    retrieved_contexts: Sequence[RetrievedAdviceContext] | None = None,
) -> MonthlyFeedbackServiceResult:
    """서비스 중간 실패를 호출자가 확인할 수 있는 결과 모델로 변환한다."""
    return MonthlyFeedbackServiceResult(
        member_id=member_id,
        analysis_month=analysis_month,
        monthly_analysis=monthly_analysis,
        interpretation_result=_to_json_object(interpretation_result or {}),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=list(retrieval_queries or []),
        retrieved_contexts=list(retrieved_contexts or []),
        error=error,
    )


def generate_monthly_feedback(
    *,
    member_id: int = 1,
    analysis_month: str = "2024-04",
    settings: Settings | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 6,
    max_queries: int = 4,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    interpretation_temperature: float = 0.0,
    feedback_temperature: float = 0.0,
    persona_key: str | None = None,
    compact_feedback_input: bool = True,
    timing_callback: FeedbackTimingCallback | None = None,
) -> MonthlyFeedbackServiceResult:
    """월간 소비 분석, 해석, RAG 검색, 최종 월간 피드백 생성을 한 번에 실행한다."""
    config = settings or get_settings()

    chat_model_error = config.chat_model_error
    if chat_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            analysis_month=analysis_month,
            error=chat_model_error,
        )

    embedding_model_error = config.embedding_model_error
    if embedding_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            analysis_month=analysis_month,
            error=embedding_model_error,
        )

    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    try:
        monthly_payload = run_timed_feedback_step(
            step_key="monthly_analysis",
            step_name="월간 소비 분석 JSON 생성",
            detail="SQLite transactions 조회와 pandas 지표 계산",
            timing_callback=timing_callback,
            operation=lambda: build_monthly_consumption_analysis_json(
                member_id=member_id,
                analysis_month=analysis_month,
                settings=config,
            ),
        )
        monthly_data = run_timed_feedback_step(
            step_key="parse_monthly_analysis",
            step_name="월간 분석 모델 검증",
            detail="분석 JSON을 MonthlySpendingData Pydantic 모델로 변환",
            timing_callback=timing_callback,
            operation=lambda: parse_monthly_spending_data(monthly_payload),
        )
        user_profile = run_timed_feedback_step(
            step_key="user_profile",
            step_name="사용자 프로필 조회",
            detail="SQLite users 테이블 조회",
            timing_callback=timing_callback,
            operation=lambda: load_user_profile_context(
                member_id=member_id,
                settings=config,
            ),
        )
        interpretation_chain = build_monthly_spending_analysis_chain(
            settings=config,
            temperature=interpretation_temperature,
        )
        interpretation_result = run_timed_feedback_step(
            step_key="interpretation_chain",
            step_name="소비 해석 체인 실행",
            detail="월간 지표 기반 구조화 LLM 호출",
            timing_callback=timing_callback,
            operation=lambda: interpretation_chain.invoke(
                make_monthly_spending_analysis_input(
                    monthly_data,
                    user_profile=user_profile,
                )
            ),
        )
        retrieval_query_plan = run_timed_feedback_step(
            step_key="retrieval_queries",
            step_name="RAG 검색 질의 생성",
            detail=f"document_kinds={len(_MONTHLY_FEEDBACK_DOCUMENT_KINDS)}, max_queries={max_queries}",
            timing_callback=timing_callback,
            operation=lambda: build_monthly_feedback_retrieval_queries_by_document_kind(
                monthly_data,
                interpretation_result=interpretation_result,
                user_profile=user_profile,
                max_queries=max_queries,
            ),
        )
        retrieval_queries = _flatten_monthly_retrieval_query_plan(retrieval_query_plan)
        queries_by_document_kind = (
            None if raw_data_dir is not None or source_files is not None else retrieval_query_plan
        )
        advice_contexts = run_timed_feedback_step(
            step_key="rag_retrieval",
            step_name="RAG 문서 검색",
            detail=(
                f"queries={sum(len(queries) for queries in retrieval_query_plan.values())}, "
                f"top_k={top_k}, "
                f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
            ),
            timing_callback=timing_callback,
            operation=lambda: retrieve_feedback_contexts(
                retrieval_queries,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                raw_data_dir=raw_data_dir,
                source_files=source_files,
                document_kinds=_MONTHLY_FEEDBACK_DOCUMENT_KINDS,
                queries_by_document_kind=queries_by_document_kind,
                settings=config,
            ),
        )
        if not advice_contexts:
            return _build_error_result(
                member_id=member_id,
                analysis_month=analysis_month,
                error="missing_documents",
                monthly_analysis=monthly_data,
                interpretation_result=interpretation_result,
                user_profile=user_profile,
                memory_context=memory_context,
                retrieval_queries=retrieval_queries,
            )

        memory_context = run_timed_feedback_step(
            step_key="memory_context",
            step_name="피드백 메모리 조회",
            detail="SQLite user_memories와 최근 monthly session 조회",
            timing_callback=timing_callback,
            operation=lambda: load_monthly_feedback_memory_context(
                member_id=member_id,
                analysis_month=analysis_month,
                settings=config,
            ),
        )
        feedback_chain = build_monthly_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
            persona_key=persona_key,
        )
        feedback = run_timed_feedback_step(
            step_key="feedback_chain",
            step_name="최종 피드백 체인 실행",
            detail="분석/해석/RAG/프로필/메모리 기반 구조화 LLM 호출",
            timing_callback=timing_callback,
            operation=lambda: feedback_chain.invoke(
                make_monthly_feedback_input(
                    monthly_data=monthly_data,
                    interpretation_result=interpretation_result,
                    advice_contexts=advice_contexts,
                    user_profile=user_profile,
                    memory_context=memory_context,
                    compact_input=compact_feedback_input,
                )
            ),
        )
        feedback_result = (
            feedback
            if isinstance(feedback, MonthlyFeedbackResult)
            else MonthlyFeedbackResult.model_validate(feedback)
        )
        feedback_result = _sanitize_monthly_feedback_result(feedback_result, monthly_data)
        feedback_result.feedback_message = coerce_ratio_context_message(
            message=feedback_result.feedback_message,
            warnings=collect_ratio_context_warnings(monthly_data.category_deep),
            period_label="이번 달",
            mission=feedback_result.next_month_mission,
        )
        run_timed_feedback_step(
            step_key="save_session",
            step_name="피드백 세션 저장",
            detail="SQLite session 테이블 저장 또는 갱신",
            timing_callback=timing_callback,
            operation=lambda: save_monthly_feedback_session(
                member_id=member_id,
                analysis_month=analysis_month,
                monthly_analysis=monthly_data,
                feedback=feedback_result,
                settings=config,
            ),
        )
        run_timed_feedback_step(
            step_key="refresh_memory",
            step_name="장기 메모리 요약 갱신",
            detail=f"최근 최대 {_DEFAULT_MONTHLY_MEMORY_SESSION_LIMIT}개 세션을 LLM으로 요약",
            timing_callback=timing_callback,
            operation=lambda: _refresh_monthly_user_memory_if_possible(
                member_id=member_id,
                settings=config,
            ),
        )
    except Exception as exc:
        return _build_error_result(
            member_id=member_id,
            analysis_month=analysis_month,
            error=str(exc),
            user_profile=user_profile,
            memory_context=memory_context,
        )

    return MonthlyFeedbackServiceResult(
        member_id=member_id,
        analysis_month=analysis_month,
        feedback=feedback_result,
        monthly_analysis=monthly_data,
        interpretation_result=_to_json_object(interpretation_result),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=retrieval_queries,
        retrieved_contexts=advice_contexts,
    )
