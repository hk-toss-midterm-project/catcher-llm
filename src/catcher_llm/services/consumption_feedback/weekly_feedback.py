from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sqlalchemy import select

from catcher_llm.chains.consumption_feedback import (
    build_memory_summary_chain,
    build_weekly_feedback_chain,
    build_weekly_spending_analysis_chain,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import SessionModel, UserMemoryModel
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
    RetrievedAdviceContext,
    SpendingMetric,
    UserProfileContext,
    WeeklyCategoryChangeIndicator,
    WeeklyFeedbackResult,
    WeeklyFeedbackServiceResult,
    WeeklyHighSpendingItem,
    WeeklyMerchantVisit,
    WeeklySpendingData,
    WeeklySpendingIndicatorPayload,
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
from catcher_llm.services.consumption_feedback.weekly_analysis import (
    build_weekly_consumption_analysis_json,
)
from catcher_llm.services.user_data_service import ensure_user_database

_WEEKLY_MEMORY_PERIOD_TYPE = "weekly"
_DEFAULT_WEEKLY_MEMORY_SESSION_LIMIT = 8

_DEFAULT_WEEKLY_RETRIEVAL_QUERY = "주간 소비 절약 실천 방법"


def _parse_week_date(value: str | date) -> date:
    """문자열 또는 date 입력을 주간 피드백 기준일로 변환한다."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


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


def parse_weekly_spending_data(payload: object) -> WeeklySpendingData:
    """이미 메모리에 있는 주간 소비 분석 JSON 객체를 검증된 입력 모델로 변환한다."""
    return WeeklySpendingData.model_validate(payload)


def _build_weekly_category_change_indicators(
    weekly_data: WeeklySpendingData,
) -> list[WeeklyCategoryChangeIndicator]:
    """주간 카테고리 요약을 JSON 경로가 포함된 증감 지표 목록으로 변환한다."""
    indicators: list[WeeklyCategoryChangeIndicator] = []
    for index, category in enumerate(weekly_data.category_summary):
        direction: CategoryDirection = get_category_direction(float(category.diff_amount))
        indicators.append(
            WeeklyCategoryChangeIndicator(
                category=category.category,
                total_amount=category.total_amount,
                prev_week_amount=category.prev_week_amount,
                diff_amount=category.diff_amount,
                diff_rate_percent=category.diff_rate_percent,
                direction=direction,
                source_json_path=f"category_summary[{index}]",
            )
        )
    return indicators


def _get_largest_weekly_category_increase(
    changes: list[WeeklyCategoryChangeIndicator],
) -> WeeklyCategoryChangeIndicator | None:
    """전주 대비 지출이 가장 크게 증가한 주간 카테고리 지표를 찾는다."""
    increased_changes = [change for change in changes if change.diff_amount > 0]
    if not increased_changes:
        return None
    return max(increased_changes, key=lambda change: change.diff_amount)


def _get_largest_weekly_category_decrease(
    changes: list[WeeklyCategoryChangeIndicator],
) -> WeeklyCategoryChangeIndicator | None:
    """전주 대비 지출이 가장 크게 감소한 주간 카테고리 지표를 찾는다."""
    decreased_changes = [change for change in changes if change.diff_amount < 0]
    if not decreased_changes:
        return None
    return min(decreased_changes, key=lambda change: change.diff_amount)


def _build_weekly_core_metrics(weekly_data: WeeklySpendingData) -> list[SpendingMetric]:
    """주간 분석에 자주 쓰는 핵심 소비 지표를 원본 JSON에서 직접 추출한다."""
    weekly_summary = weekly_data.weekly_summary
    repeat_patterns = weekly_data.repeat_patterns
    waste_detection = weekly_data.waste_detection
    saving_potential = weekly_data.saving_potential
    weekly_metrics = weekly_data.weekly_metrics
    weekly_special_metrics = weekly_metrics.special_metrics
    weekly_comparisons = weekly_data.weekly_comparisons
    recent_average = weekly_comparisons.recent_4week_average
    same_week_last_month = weekly_comparisons.same_week_last_month

    return [
        make_spending_metric(
            "이번 주 총 지출액",
            weekly_summary.this_week_total,
            "KRW",
            "weekly_summary.this_week_total",
            "분석 주간의 총 소비 금액",
        ),
        make_spending_metric(
            "전주 대비 지출 증감액",
            weekly_summary.amount_diff,
            "KRW",
            "weekly_summary.amount_diff",
            "전주 총 지출과 이번 주 총 지출의 차이",
        ),
        make_spending_metric(
            "전주 대비 지출 증감률",
            weekly_summary.diff_rate_percent,
            "percent",
            "weekly_summary.diff_rate_percent",
            "전주 총 지출 대비 이번 주 지출 증감률",
        ),
        make_spending_metric(
            "최근 4주 평균 대비 지출 증감액",
            recent_average.amount_diff,
            "KRW",
            "weekly_comparisons.recent_4week_average.amount_diff",
            "최근 4주 주간 평균 지출과 분석 주간 총 지출의 차이",
        ),
        make_spending_metric(
            "최근 4주 평균 대비 지출 증감률",
            recent_average.amount_diff_rate_percent,
            "percent",
            "weekly_comparisons.recent_4week_average.amount_diff_rate_percent",
            "최근 4주 주간 평균 지출 대비 분석 주간 지출 증감률",
        ),
        make_spending_metric(
            "지난달 같은 주차 대비 지출 증감액",
            same_week_last_month.amount_diff,
            "KRW",
            "weekly_comparisons.same_week_last_month.amount_diff",
            "지난달 같은 주차 총 지출과 분석 주간 총 지출의 차이",
        ),
        make_spending_metric(
            "지난달 같은 주차 대비 지출 증감률",
            same_week_last_month.amount_diff_rate_percent,
            "percent",
            "weekly_comparisons.same_week_last_month.amount_diff_rate_percent",
            "지난달 같은 주차 총 지출 대비 분석 주간 지출 증감률",
        ),
        make_spending_metric(
            "주간 결제 건수",
            weekly_summary.transaction_count,
            "count",
            "weekly_summary.transaction_count",
            "분석 주간 전체 결제 건수",
        ),
        make_spending_metric(
            "주간 일평균 지출",
            weekly_summary.daily_average,
            "KRW",
            "weekly_summary.daily_average",
            "소비가 발생한 날 기준 일평균 지출액",
        ),
        make_spending_metric(
            "피크 요일",
            weekly_data.weekday_pattern.peak_weekday or "",
            "weekday",
            "weekday_pattern.peak_weekday",
            "분석 주간 지출액이 가장 큰 요일",
        ),
        make_spending_metric(
            "배달 결제 금액",
            repeat_patterns.delivery.total_amount,
            "KRW",
            "repeat_patterns.delivery.total_amount",
            "배달 키워드가 포함된 주간 결제 금액 합계",
        ),
        make_spending_metric(
            "카페 결제 건수",
            repeat_patterns.cafe.count,
            "count",
            "repeat_patterns.cafe.count",
            "카페 키워드가 포함된 주간 결제 건수",
        ),
        make_spending_metric(
            "야간 소비 금액",
            waste_detection.late_night.total_amount,
            "KRW",
            "waste_detection.late_night.total_amount",
            "21시 이후 발생한 주간 결제 금액 합계",
        ),
        make_spending_metric(
            "소액 결제 건수",
            waste_detection.micro_spending.count,
            "count",
            "waste_detection.micro_spending.count",
            "소액 결제 기준 미만의 주간 결제 건수",
        ),
        make_spending_metric(
            "고액 결제 건수",
            waste_detection.high_spending.count,
            "count",
            "waste_detection.high_spending.items",
            "IQR 상한선을 초과한 주간 결제 항목 수",
        ),
        make_spending_metric(
            "배달 1회 절약 가능액",
            saving_potential.delivery_save_per_skip,
            "KRW",
            "saving_potential.delivery_save_per_skip",
            "배달 주문을 한 번 줄였을 때 예상 절약 금액",
        ),
        make_spending_metric(
            "평일-주말 소비 상관관계",
            weekly_data.elasticity_analysis.correlation
            if weekly_data.elasticity_analysis.correlation is not None
            else 0.0,
            "ratio",
            "elasticity_analysis.correlation",
            "평일 지출과 주말 지출의 상관관계 (음수일수록 심리적 반동 위험 증가)",
        ),
        make_spending_metric(
            "평일 지출 임계점",
            weekly_data.elasticity_analysis.threshold or 0,
            "KRW",
            "elasticity_analysis.threshold",
            "이 금액 미만으로 안 쓰면 주말 소비가 급증하는 평일 일평균 지출 수준",
        ),
        make_spending_metric(
            "주중 소비 비중",
            weekly_metrics.weekday_spending_ratio_percent,
            "percent",
            "weekly_metrics.weekday_spending_ratio_percent",
            "분석 주간 총 소비 중 주중 소비가 차지하는 비중",
        ),
        make_spending_metric(
            "주말 소비 비중",
            weekly_metrics.weekend_spending_ratio_percent,
            "percent",
            "weekly_metrics.weekend_spending_ratio_percent",
            "분석 주간 총 소비 중 주말 소비가 차지하는 비중",
        ),
        make_spending_metric(
            "주간 소비 변동성",
            weekly_metrics.weekly_spending_volatility,
            "KRW",
            "weekly_metrics.weekly_spending_volatility",
            "월요일부터 일요일까지 일별 소비 금액의 표준편차",
        ),
        make_spending_metric(
            "주간 예산 소진율",
            weekly_metrics.weekly_budget_usage_rate_percent,
            "percent",
            "weekly_metrics.weekly_budget_usage_rate_percent",
            "설정된 주간 예산 대비 분석 주간 소비 금액 비율",
        ),
        make_spending_metric(
            "주말 과소비 지수",
            weekly_special_metrics.weekend_overspending_index,
            "ratio",
            "weekly_metrics.special_metrics.weekend_overspending_index",
            "주중 일평균 소비 대비 주말 일평균 소비 배율",
        ),
        make_spending_metric(
            "소비 요일 편중도",
            weekly_special_metrics.weekday_concentration_ratio_percent,
            "percent",
            "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
            "분석 주간 총 소비 중 최대 소비 요일이 차지하는 비중",
        ),
    ]


def extract_weekly_spending_indicators(
    weekly_data: WeeklySpendingData,
) -> WeeklySpendingIndicatorPayload:
    """주간 소비 분석 모델에서 해석 체인에 넣을 핵심 지표 묶음을 추출한다."""
    category_changes = _build_weekly_category_change_indicators(weekly_data)
    return WeeklySpendingIndicatorPayload(
        member_id=weekly_data.member_id,
        week_start=weekly_data.week_start,
        week_end=weekly_data.week_end,
        metrics=_build_weekly_core_metrics(weekly_data),
        category_changes=category_changes,
        largest_category_increase=_get_largest_weekly_category_increase(category_changes),
        largest_category_decrease=_get_largest_weekly_category_decrease(category_changes),
        high_spending_items=weekly_data.waste_detection.high_spending.items,
        top_merchants=weekly_data.repeat_patterns.top_merchants,
        peak_weekday=weekly_data.weekday_pattern.peak_weekday,
    )


def make_weekly_spending_analysis_input(
    weekly_data: WeeklySpendingData,
    *,
    user_profile: object | None = None,
) -> dict[str, str]:
    """주간 해석 체인에 전달할 원본 JSON, 추출 지표 JSON, 사용자 프로필 JSON 입력을 만든다."""
    indicators = extract_weekly_spending_indicators(weekly_data)
    return {
        "raw_json": weekly_data.model_dump_json(),
        "indicator_json": indicators.model_dump_json(),
        "user_profile_json": serialize_context_object(user_profile or {}),
    }


def _append_unique_query(queries: list[str], query: str) -> None:
    """비어 있지 않고 아직 없는 RAG 검색 질의만 목록에 추가한다."""
    normalized_query = " ".join(query.split())
    if normalized_query and normalized_query not in queries:
        queries.append(normalized_query)


def _iter_action_missions(action_result: object) -> list[ActionMission]:
    """주간 해석 결과의 개선 후보 모델 또는 dict에서 RAG 검색 후보 목록을 추출한다."""
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
    """주간 원인 해석 모델 또는 dict에서 RAG 검색용 개입 타겟 후보를 추출한다."""
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
    """주간 개입 타겟 후보에서 RAG 검색에 사용할 질의 문구를 선택한다."""
    return target.query_hint or target.title


def _append_weekly_high_spending_queries(
    queries: list[str],
    items: Sequence[WeeklyHighSpendingItem],
) -> None:
    """고액 결제 항목을 기반으로 주간 피드백용 RAG 검색 질의를 추가한다."""
    for item in sorted(
        items, key=lambda high_spending_item: high_spending_item.amount, reverse=True
    ):
        _append_unique_query(queries, f"{item.category} {item.merchant} 지출 줄이는 방법")


def _append_top_merchant_queries(
    queries: list[str],
    merchants: Sequence[WeeklyMerchantVisit],
) -> None:
    """반복 가맹점 정보를 기반으로 주간 피드백용 RAG 검색 질의를 추가한다."""
    for merchant in sorted(merchants, key=lambda item: item.visit_count, reverse=True)[:2]:
        if merchant.visit_count < 2:
            continue
        _append_unique_query(queries, f"{merchant.merchant} 반복 소비 줄이는 방법")


def build_weekly_feedback_retrieval_queries(
    weekly_data: WeeklySpendingData,
    *,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    max_queries: int = 4,
) -> list[str]:
    """주간 분석, 해석 결과, 사용자 프로필에서 최종 피드백용 RAG 검색 질의를 생성한다."""
    indicators = extract_weekly_spending_indicators(weekly_data)
    queries: list[str] = []

    if indicators.largest_category_increase is not None:
        category = indicators.largest_category_increase.category
        _append_unique_query(queries, f"{category} 주간 소비 절약 방법")

    _append_weekly_high_spending_queries(queries, indicators.high_spending_items)

    cause_result = (interpretation_result or {}).get("cause_result")
    for target in _iter_intervention_targets(cause_result):
        _append_unique_query(queries, f"{_intervention_target_query_text(target)} 절약 방법")

    action_result = (interpretation_result or {}).get("action_result")
    for mission in _iter_action_missions(action_result):
        _append_unique_query(queries, f"{mission.title} 실천 방법")

    if user_profile is not None:
        if user_profile.saving_goal_text:
            _append_unique_query(
                queries, f"{user_profile.saving_goal_text} 목표 주간 소비 절약 방법"
            )
        if user_profile.job and indicators.largest_category_increase is not None:
            category = indicators.largest_category_increase.category
            _append_unique_query(queries, f"{user_profile.job} {category} 소비 줄이는 방법")
        if user_profile.persona:
            _append_unique_query(queries, f"{user_profile.persona} 주간 소비 습관 개선 방법")

    _append_top_merchant_queries(queries, indicators.top_merchants)

    # 소비 탄성: 치팅 데이 패턴이 확인된 경우 관련 쿼리 추가
    if weekly_data.elasticity_analysis.cheat_effective:
        _append_unique_query(queries, "소비 탄성 관리 가심비 지출 전략")

    _append_unique_query(queries, _DEFAULT_WEEKLY_RETRIEVAL_QUERY)
    return queries[:max_queries]


def make_weekly_feedback_input(
    *,
    weekly_data: WeeklySpendingData,
    interpretation_result: dict[str, object],
    advice_contexts: Sequence[RetrievedAdviceContext],
    user_profile: object | None = None,
    memory_context: object | None = None,
) -> dict[str, str]:
    """최종 주간 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    return {
        "weekly_json": weekly_data.model_dump_json(),
        "interpretation_json": serialize_interpretation_result(interpretation_result),
        "retrieved_contexts": serialize_advice_contexts(advice_contexts),
        "user_profile_json": serialize_context_object(user_profile or {}),
        "memory_context_json": serialize_context_object(memory_context or {}),
    }


def save_weekly_feedback_session(
    *,
    member_id: int,
    week_start: date,
    weekly_analysis: WeeklySpendingData,
    feedback: WeeklyFeedbackResult,
    settings: Settings | None = None,
) -> None:
    """최종 주간 피드백 실행 결과를 session 테이블에 주 시작일 기준으로 저장하거나 갱신한다."""
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
                SessionModel.analysis_date == str(week_start),
                SessionModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
            )
        )
        if session_row is None:
            session_row = SessionModel(
                user_id=member_id,
                analysis_date=str(week_start),
                period_type=_WEEKLY_MEMORY_PERIOD_TYPE,
            )
            session.add(session_row)

        session_row.analysis_result = weekly_analysis.model_dump_json()
        session_row.feedback_message = feedback.feedback_message
        session_row.feedback_reason = feedback_reason
        session_row.todo_tomorrow = feedback.next_week_mission


def load_weekly_session_for_date(
    *,
    member_id: int,
    week_start: date,
    settings: Settings | None = None,
) -> SessionModel | None:
    """session 테이블에서 특정 주 시작일의 weekly 세션을 조회한다. 없으면 None을 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        return session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == str(week_start),
                SessionModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
            )
        )


def load_weekly_feedback_memory_context(
    *,
    member_id: int,
    week_start: date | str | None = None,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_WEEKLY_MEMORY_SESSION_LIMIT,
) -> DailyFeedbackMemoryContext:
    """SQLite user_memories와 최근 주간 session에서 최종 피드백용 과거 맥락을 조회한다."""
    config = settings or get_settings()
    ensure_user_database(config)
    normalized_week_start = str(_parse_week_date(week_start)) if week_start is not None else None

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
            )
        )
        session_statement = select(SessionModel).where(
            SessionModel.user_id == member_id,
            SessionModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
        )
        if normalized_week_start is not None:
            session_statement = session_statement.where(
                SessionModel.analysis_date < normalized_week_start
            )
        recent_sessions = list(
            session.scalars(
                session_statement.order_by(
                    SessionModel.analysis_date.desc(),
                    SessionModel.id.desc(),
                ).limit(recent_session_limit)
            )
        )

    chronological_sessions = list(reversed(recent_sessions))
    return DailyFeedbackMemoryContext(
        user_id=member_id,
        period_type=_WEEKLY_MEMORY_PERIOD_TYPE,
        memory_summary=memory.summary if memory is not None else None,
        user_feedback_memory=memory.user_feedback_memory if memory is not None else None,
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


def _extract_weekly_total_summary(analysis_result: str | None) -> str:
    """저장된 주간 분석 JSON에서 이번 주 총 지출액을 짧은 문자열로 추출한다."""
    if analysis_result is None:
        return "-"
    try:
        raw_analysis = json.loads(analysis_result)
    except json.JSONDecodeError:
        return "-"

    if not isinstance(raw_analysis, dict):
        return "-"
    weekly_summary = raw_analysis.get("weekly_summary")
    if not isinstance(weekly_summary, dict):
        return "-"
    this_week_total = weekly_summary.get("this_week_total")
    if not isinstance(this_week_total, int | float):
        return "-"
    return f"{this_week_total:,.0f}원"


def _build_weekly_session_list_text(sessions: Sequence[SessionModel]) -> str:
    """주간 세션 목록을 LLM 요약 체인에 넣을 텍스트 목록으로 직렬화한다."""
    lines = []
    for session_row in sessions:
        weekly_total = _extract_weekly_total_summary(session_row.analysis_result)
        reason_summary = extract_feedback_reason_summary(session_row.feedback_reason)
        next_mission = truncate_context_text(session_row.todo_tomorrow)
        line = (
            f"- {session_row.analysis_date}: "
            f"주간지출 {weekly_total}; "
            f"핵심근거 {reason_summary}; "
            f"다음미션 {next_mission}"
        )
        lines.append(line)
    return "\n".join(lines)


def build_weekly_memory_summary_from_sessions(
    sessions: Sequence[SessionModel],
    settings: Settings | None = None,
) -> str:
    """최근 주간 피드백 세션 목록을 LLM으로 통합 요약한 문자열로 만든다."""
    if not sessions:
        return "아직 누적된 주간 피드백 세션이 없습니다."
    session_list = _build_weekly_session_list_text(sessions)
    chain = build_memory_summary_chain("주간", settings)
    try:
        return chain.invoke({"session_list": session_list})
    except Exception:
        return session_list


def refresh_weekly_user_memory(
    *,
    member_id: int,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_WEEKLY_MEMORY_SESSION_LIMIT,
) -> str:
    """최근 주간 session 기록을 바탕으로 user_memories의 weekly 요약을 생성하거나 갱신한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        recent_sessions = list(
            session.scalars(
                select(SessionModel)
                .where(
                    SessionModel.user_id == member_id,
                    SessionModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
                )
                .order_by(SessionModel.analysis_date.desc(), SessionModel.id.desc())
                .limit(recent_session_limit)
            )
        )
        summary = build_weekly_memory_summary_from_sessions(list(reversed(recent_sessions)), config)

        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _WEEKLY_MEMORY_PERIOD_TYPE,
            )
        )
        if memory is None:
            memory = UserMemoryModel(
                user_id=member_id,
                period_type=_WEEKLY_MEMORY_PERIOD_TYPE,
                summary=summary,
            )
            session.add(memory)
        else:
            memory.summary = summary

    return summary


def _refresh_weekly_user_memory_if_possible(
    *,
    member_id: int,
    settings: Settings | None = None,
) -> None:
    """피드백 생성 성공 이후 주간 메모리 요약을 가능할 때만 갱신한다."""
    try:
        refresh_weekly_user_memory(member_id=member_id, settings=settings)
    except Exception:
        return


def _build_error_result(
    *,
    member_id: int,
    week_start: date,
    week_end: date,
    error: str,
    weekly_analysis: WeeklySpendingData | None = None,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    memory_context: DailyFeedbackMemoryContext | None = None,
    retrieval_queries: Sequence[str] | None = None,
    retrieved_contexts: Sequence[RetrievedAdviceContext] | None = None,
) -> WeeklyFeedbackServiceResult:
    """서비스 중간 실패를 호출자가 확인할 수 있는 결과 모델로 변환한다."""
    return WeeklyFeedbackServiceResult(
        member_id=member_id,
        week_start=str(week_start),
        week_end=str(week_end),
        weekly_analysis=weekly_analysis,
        interpretation_result=_to_json_object(interpretation_result or {}),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=list(retrieval_queries or []),
        retrieved_contexts=list(retrieved_contexts or []),
        error=error,
    )


def generate_weekly_feedback(
    *,
    member_id: int = 1,
    week_start: str | date = "2024-04-01",
    week_end: str | date = "2024-04-07",
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
) -> WeeklyFeedbackServiceResult:
    """주간 소비 분석, 해석, RAG 검색, 최종 주간 피드백 생성을 한 번에 실행한다."""
    config = settings or get_settings()
    start_day = _parse_week_date(week_start)
    end_day = _parse_week_date(week_end)

    chat_model_error = config.chat_model_error
    if chat_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            week_start=start_day,
            week_end=end_day,
            error=chat_model_error,
        )

    embedding_model_error = config.embedding_model_error
    if embedding_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            week_start=start_day,
            week_end=end_day,
            error=embedding_model_error,
        )

    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    try:
        weekly_payload = build_weekly_consumption_analysis_json(
            member_id=member_id,
            week_start=start_day,
            week_end=end_day,
            settings=config,
        )
        weekly_data = parse_weekly_spending_data(weekly_payload)
        user_profile = load_user_profile_context(
            member_id=member_id,
            settings=config,
        )
        memory_context = load_weekly_feedback_memory_context(
            member_id=member_id,
            week_start=start_day,
            settings=config,
        )
        interpretation_chain = build_weekly_spending_analysis_chain(
            settings=config,
            temperature=interpretation_temperature,
        )
        interpretation_result = interpretation_chain.invoke(
            make_weekly_spending_analysis_input(
                weekly_data,
                user_profile=user_profile,
            )
        )
        retrieval_queries = build_weekly_feedback_retrieval_queries(
            weekly_data,
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
                week_start=start_day,
                week_end=end_day,
                error="missing_documents",
                weekly_analysis=weekly_data,
                interpretation_result=interpretation_result,
                user_profile=user_profile,
                memory_context=memory_context,
                retrieval_queries=retrieval_queries,
            )

        feedback_chain = build_weekly_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
            persona_key=persona_key,
        )
        feedback = feedback_chain.invoke(
            make_weekly_feedback_input(
                weekly_data=weekly_data,
                interpretation_result=interpretation_result,
                advice_contexts=advice_contexts,
                user_profile=user_profile,
                memory_context=memory_context,
            )
        )
        feedback_result = (
            feedback
            if isinstance(feedback, WeeklyFeedbackResult)
            else WeeklyFeedbackResult.model_validate(feedback)
        )
        save_weekly_feedback_session(
            member_id=member_id,
            week_start=start_day,
            weekly_analysis=weekly_data,
            feedback=feedback_result,
            settings=config,
        )
        _refresh_weekly_user_memory_if_possible(
            member_id=member_id,
            settings=config,
        )
    except Exception as exc:
        return _build_error_result(
            member_id=member_id,
            week_start=start_day,
            week_end=end_day,
            error=str(exc),
            user_profile=user_profile,
            memory_context=memory_context,
        )

    return WeeklyFeedbackServiceResult(
        member_id=member_id,
        week_start=str(start_day),
        week_end=str(end_day),
        feedback=feedback_result,
        weekly_analysis=weekly_data,
        interpretation_result=_to_json_object(interpretation_result),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=retrieval_queries,
        retrieved_contexts=advice_contexts,
    )
