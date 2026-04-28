from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from catcher_llm.chains.consumption_feedback import (
    build_weekly_feedback_chain,
    build_weekly_spending_analysis_chain,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    CategoryDirection,
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
    get_category_direction,
    make_spending_metric,
)
from catcher_llm.services.consumption_feedback.weekly_analysis import (
    build_weekly_consumption_analysis_json,
)

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
        "raw_json": weekly_data.model_dump_json(indent=2),
        "indicator_json": indicators.model_dump_json(indent=2),
        "user_profile_json": serialize_context_object(user_profile or {}),
    }


def _append_unique_query(queries: list[str], query: str) -> None:
    """비어 있지 않고 아직 없는 RAG 검색 질의만 목록에 추가한다."""
    normalized_query = " ".join(query.split())
    if normalized_query and normalized_query not in queries:
        queries.append(normalized_query)


def _iter_action_missions(action_result: object) -> list[ActionMission]:
    """주간 해석 결과의 행동 개선 모델 또는 dict에서 실행 미션 목록을 추출한다."""
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
) -> dict[str, str]:
    """최종 주간 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    return {
        "weekly_json": weekly_data.model_dump_json(indent=2),
        "interpretation_json": serialize_interpretation_result(interpretation_result),
        "retrieved_contexts": serialize_advice_contexts(advice_contexts),
        "user_profile_json": serialize_context_object(user_profile or {}),
    }


def _build_error_result(
    *,
    member_id: int,
    week_start: date,
    week_end: date,
    error: str,
    weekly_analysis: WeeklySpendingData | None = None,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
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
                retrieval_queries=retrieval_queries,
            )

        feedback_chain = build_weekly_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
        )
        feedback = feedback_chain.invoke(
            make_weekly_feedback_input(
                weekly_data=weekly_data,
                interpretation_result=interpretation_result,
                advice_contexts=advice_contexts,
                user_profile=user_profile,
            )
        )
        feedback_result = (
            feedback
            if isinstance(feedback, WeeklyFeedbackResult)
            else WeeklyFeedbackResult.model_validate(feedback)
        )
    except Exception as exc:
        return _build_error_result(
            member_id=member_id,
            week_start=start_day,
            week_end=end_day,
            error=str(exc),
            user_profile=user_profile,
        )

    return WeeklyFeedbackServiceResult(
        member_id=member_id,
        week_start=str(start_day),
        week_end=str(end_day),
        feedback=feedback_result,
        weekly_analysis=weekly_data,
        interpretation_result=_to_json_object(interpretation_result),
        user_profile=user_profile,
        retrieval_queries=retrieval_queries,
        retrieved_contexts=advice_contexts,
    )
