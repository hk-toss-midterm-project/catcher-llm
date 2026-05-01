from __future__ import annotations

import json
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
from catcher_llm.db.models import SessionModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    CategoryDirection,
    DailyFeedbackMemoryContext,
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
from catcher_llm.services.user_data_service import ensure_user_database

_MONTHLY_MEMORY_PERIOD_TYPE = "monthly"
_DEFAULT_MONTHLY_MEMORY_SESSION_LIMIT = 6

_DEFAULT_MONTHLY_RETRIEVAL_QUERY = "월간 소비 절약 실천 방법"


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
    """월간 해석 결과의 행동 개선 모델 또는 dict에서 실행 미션 목록을 추출한다."""
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


def make_monthly_feedback_input(
    *,
    monthly_data: MonthlySpendingData,
    interpretation_result: dict[str, object],
    advice_contexts: Sequence[RetrievedAdviceContext],
    user_profile: object | None = None,
    memory_context: object | None = None,
) -> dict[str, str]:
    """최종 월간 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    return {
        "monthly_json": monthly_data.model_dump_json(),
        "interpretation_json": serialize_interpretation_result(interpretation_result),
        "retrieved_contexts": serialize_advice_contexts(advice_contexts),
        "user_profile_json": serialize_context_object(user_profile or {}),
        "memory_context_json": serialize_context_object(memory_context or {}),
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
    settings: Settings | None = None,
) -> DailyFeedbackMemoryContext:
    """SQLite user_memories 테이블에서 월간 피드백용 장기 메모리 요약을 조회한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _MONTHLY_MEMORY_PERIOD_TYPE,
            )
        )

    return DailyFeedbackMemoryContext(
        user_id=member_id,
        period_type=_MONTHLY_MEMORY_PERIOD_TYPE,
        memory_summary=memory.summary if memory is not None else None,
        user_feedback_memory=memory.user_feedback_memory if memory is not None else None,
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
    top_k: int = 3,
    max_queries: int = 4,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    interpretation_temperature: float = 0.0,
    feedback_temperature: float = 0.0,
    persona_key: str | None = None,
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
        monthly_payload = build_monthly_consumption_analysis_json(
            member_id=member_id,
            analysis_month=analysis_month,
            settings=config,
        )
        monthly_data = parse_monthly_spending_data(monthly_payload)
        user_profile = load_user_profile_context(
            member_id=member_id,
            settings=config,
        )
        memory_context = load_monthly_feedback_memory_context(
            member_id=member_id,
            settings=config,
        )
        interpretation_chain = build_monthly_spending_analysis_chain(
            settings=config,
            temperature=interpretation_temperature,
        )
        interpretation_result = interpretation_chain.invoke(
            make_monthly_spending_analysis_input(
                monthly_data,
                user_profile=user_profile,
            )
        )
        retrieval_queries = build_monthly_feedback_retrieval_queries(
            monthly_data,
            interpretation_result=interpretation_result,
            user_profile=user_profile,
            max_queries=max_queries,
        )
        advice_contexts = retrieve_feedback_contexts(
            retrieval_queries,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            source_files=source_files,
            settings=config,
        )
        if not advice_contexts:
            return _build_error_result(
                member_id=member_id,
                analysis_month=analysis_month,
                error="missing_documents",
                monthly_analysis=monthly_data,
                interpretation_result=interpretation_result,
                user_profile=user_profile,
                retrieval_queries=retrieval_queries,
            )

        feedback_chain = build_monthly_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
            persona_key=persona_key,
        )
        feedback = feedback_chain.invoke(
            make_monthly_feedback_input(
                monthly_data=monthly_data,
                interpretation_result=interpretation_result,
                advice_contexts=advice_contexts,
                user_profile=user_profile,
                memory_context=memory_context,
            )
        )
        feedback_result = (
            feedback
            if isinstance(feedback, MonthlyFeedbackResult)
            else MonthlyFeedbackResult.model_validate(feedback)
        )
        save_monthly_feedback_session(
            member_id=member_id,
            analysis_month=analysis_month,
            monthly_analysis=monthly_data,
            feedback=feedback_result,
            settings=config,
        )
        _refresh_monthly_user_memory_if_possible(
            member_id=member_id,
            settings=config,
        )
    except Exception as exc:
        return _build_error_result(
            member_id=member_id,
            analysis_month=analysis_month,
            error=str(exc),
            user_profile=user_profile,
        )

    return MonthlyFeedbackServiceResult(
        member_id=member_id,
        analysis_month=analysis_month,
        feedback=feedback_result,
        monthly_analysis=monthly_data,
        interpretation_result=_to_json_object(interpretation_result),
        user_profile=user_profile,
        retrieval_queries=retrieval_queries,
        retrieved_contexts=advice_contexts,
    )
