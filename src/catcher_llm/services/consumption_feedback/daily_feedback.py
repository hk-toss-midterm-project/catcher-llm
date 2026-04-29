from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sqlalchemy import select

from catcher_llm.chains.consumption_feedback import (
    build_daily_feedback_chain,
    build_memory_summary_chain,
    build_spending_analysis_chain,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import SessionModel, UserMemoryModel, UserModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    DailyFeedbackMemoryContext,
    DailyFeedbackResult,
    DailyFeedbackServiceResult,
    DailyFeedbackSessionContext,
    JsonObject,
    JsonScalar,
    JsonValue,
    RetrievedAdviceContext,
    UserProfileContext,
    UserSpendingData,
)
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_feedback_reason_summary,
    extract_spending_indicators,
    make_spending_analysis_input,
    parse_user_spending_data,
    truncate_context_text,
)
from catcher_llm.services.rag.core import retrieve_context_records
from catcher_llm.services.user_data_service import ensure_user_database

_DEFAULT_RETRIEVAL_QUERY = "일일 소비 절약 실천 방법"
_DAILY_MEMORY_PERIOD_TYPE = "daily"
_DEFAULT_RECENT_SESSION_LIMIT = 7
_DEFAULT_MEMORY_SESSION_LIMIT = 14


def _parse_analysis_date(value: str | date) -> date:
    """문자열 또는 date 입력을 일일 피드백 기준일로 변환한다."""
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


def serialize_interpretation_result(interpretation_result: dict[str, object]) -> str:
    """소비 해석 체인 결과를 최종 피드백 프롬프트에 넣을 JSON 문자열로 변환한다."""
    return json.dumps(
        _to_json_object(interpretation_result),
        ensure_ascii=False,
        indent=2,
    )


def serialize_advice_contexts(contexts: Sequence[RetrievedAdviceContext]) -> str:
    """RAG 검색 문서 청크 목록을 최종 피드백 프롬프트용 JSON 문자열로 변환한다."""
    return json.dumps(
        [context.model_dump() for context in contexts],
        ensure_ascii=False,
        indent=2,
    )


def serialize_context_object(value: object) -> str:
    """사용자 프로필과 메모리 컨텍스트를 프롬프트 입력용 JSON 문자열로 변환한다."""
    return json.dumps(
        _to_json_object(value),
        ensure_ascii=False,
        indent=2,
    )


def load_user_profile_context(
    *,
    member_id: int,
    settings: Settings | None = None,
) -> UserProfileContext:
    """SQLite users 테이블에서 최종 피드백 개인화에 사용할 사용자 프로필을 조회한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        user = session.scalar(select(UserModel).where(UserModel.id == member_id))

    if user is None:
        raise ValueError(f"멤버 {member_id}번 사용자를 찾을 수 없습니다.")

    return UserProfileContext(
        user_id=user.id,
        name=user.name,
        age=user.age,
        job=user.job,
        gender=user.gender,
        income=user.income,
        region=user.region,
        card_grade=user.card_grade,
        persona=user.persona,
        saving_goal_text=user.saving_goal_text,
    )


def load_daily_feedback_memory_context(
    *,
    member_id: int,
    analysis_date: date,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_RECENT_SESSION_LIMIT,
) -> DailyFeedbackMemoryContext:
    """SQLite user_memories와 session 테이블에서 최종 피드백용 과거 맥락을 조회한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
            )
        )
        recent_sessions = list(
            session.scalars(
                select(SessionModel)
                .where(
                    SessionModel.user_id == member_id,
                    SessionModel.analysis_date < str(analysis_date),
                    SessionModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
                )
                .order_by(SessionModel.analysis_date.desc(), SessionModel.id.desc())
                .limit(recent_session_limit)
            )
        )

    chronological_sessions = list(reversed(recent_sessions))
    return DailyFeedbackMemoryContext(
        user_id=member_id,
        period_type=_DAILY_MEMORY_PERIOD_TYPE,
        memory_summary=memory.summary if memory is not None else None,
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


def save_daily_feedback_session(
    *,
    member_id: int,
    analysis_date: date,
    daily_analysis: UserSpendingData,
    feedback: DailyFeedbackResult,
    settings: Settings | None = None,
) -> None:
    """최종 일일 피드백 실행 결과를 session 테이블에 날짜 기준으로 저장하거나 갱신한다."""
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
                SessionModel.analysis_date == str(analysis_date),
                SessionModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
            )
        )
        if session_row is None:
            session_row = SessionModel(
                user_id=member_id,
                analysis_date=str(analysis_date),
                period_type=_DAILY_MEMORY_PERIOD_TYPE,
            )
            session.add(session_row)

        session_row.analysis_result = daily_analysis.model_dump_json()
        session_row.feedback_message = feedback.scolding_message
        session_row.feedback_reason = feedback_reason
        session_row.todo_tomorrow = feedback.tomorrow_mission


def _extract_daily_total_summary(analysis_result: str | None) -> str:
    """저장된 일일 분석 JSON에서 오늘 총 지출액을 짧은 문자열로 추출한다."""
    if analysis_result is None:
        return "-"
    try:
        raw_analysis = json.loads(analysis_result)
    except json.JSONDecodeError:
        return "-"

    if not isinstance(raw_analysis, dict):
        return "-"
    stable_metrics = raw_analysis.get("stable_metrics")
    if not isinstance(stable_metrics, dict):
        return "-"
    today_total = stable_metrics.get("today_total")
    if not isinstance(today_total, int | float):
        return "-"
    return f"{today_total:,.0f}원"


def load_all_daily_sessions(
    *,
    member_id: int,
    settings: Settings | None = None,
) -> list[SessionModel]:
    """session 테이블에서 해당 유저의 모든 daily 세션을 날짜 오름차순으로 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        return list(
            session.scalars(
                select(SessionModel)
                .where(
                    SessionModel.user_id == member_id,
                    SessionModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
                )
                .order_by(SessionModel.analysis_date.asc())
            )
        )


def _build_session_list_text(sessions: Sequence[SessionModel]) -> str:
    """세션 목록을 LLM 요약 체인에 넣을 수 있는 텍스트 목록으로 직렬화한다."""
    lines = []
    for session_row in sessions:
        daily_total = _extract_daily_total_summary(session_row.analysis_result)
        reason_summary = extract_feedback_reason_summary(session_row.feedback_reason)
        tomorrow_mission = truncate_context_text(session_row.todo_tomorrow)
        lines.append(
            f"- {session_row.analysis_date}: "
            f"지출 {daily_total}; "
            f"핵심근거 {reason_summary}; "
            f"다음미션 {tomorrow_mission}"
        )
    return "\n".join(lines)


def build_daily_memory_summary_from_sessions(
    sessions: Sequence[SessionModel],
    settings: Settings | None = None,
) -> str:
    """최근 일일 피드백 세션 목록을 LLM으로 통합 요약한 문자열로 만든다."""
    if not sessions:
        return "아직 누적된 일일 피드백 세션이 없습니다."
    session_list = _build_session_list_text(sessions)
    chain = build_memory_summary_chain("일일", settings)
    try:
        return chain.invoke({"session_list": session_list})
    except Exception:
        return session_list


def refresh_daily_user_memory(
    *,
    member_id: int,
    settings: Settings | None = None,
    recent_session_limit: int = _DEFAULT_MEMORY_SESSION_LIMIT,
) -> str:
    """최근 session 기록을 바탕으로 user_memories의 daily 요약을 생성하거나 갱신한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        recent_sessions = list(
            session.scalars(
                select(SessionModel)
                .where(
                    SessionModel.user_id == member_id,
                    SessionModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
                )
                .order_by(SessionModel.analysis_date.desc(), SessionModel.id.desc())
                .limit(recent_session_limit)
            )
        )
        summary = build_daily_memory_summary_from_sessions(list(reversed(recent_sessions)), config)

        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == member_id,
                UserMemoryModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
            )
        )
        if memory is None:
            memory = UserMemoryModel(
                user_id=member_id,
                period_type=_DAILY_MEMORY_PERIOD_TYPE,
                summary=summary,
            )
            session.add(memory)
        else:
            memory.summary = summary

    return summary


def _refresh_daily_user_memory_if_possible(
    *,
    member_id: int,
    settings: Settings | None = None,
) -> None:
    """피드백 생성 성공 이후 일일 메모리 요약을 가능할 때만 갱신한다."""
    try:
        refresh_daily_user_memory(member_id=member_id, settings=settings)
    except Exception:
        return


def _append_unique_query(queries: list[str], query: str) -> None:
    """비어 있지 않고 아직 없는 RAG 검색 질의만 목록에 추가한다."""
    normalized_query = " ".join(query.split())
    if normalized_query and normalized_query not in queries:
        queries.append(normalized_query)


def _iter_action_missions(action_result: object) -> list[ActionMission]:
    """해석 결과의 행동 개선 모델 또는 dict에서 실행 미션 목록을 추출한다."""
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


def build_feedback_retrieval_queries(
    user_data: UserSpendingData,
    *,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    max_queries: int = 4,
) -> list[str]:
    """소비 분석, 해석 결과, 사용자 프로필에서 최종 피드백용 RAG 검색 질의를 생성한다."""
    indicators = extract_spending_indicators(user_data)
    queries: list[str] = []

    if indicators.largest_category_increase is not None:
        category = indicators.largest_category_increase.category
        _append_unique_query(queries, f"{category} 소비 절약 방법")

    for item in sorted(
        indicators.high_spending_items,
        key=lambda high_spending_item: high_spending_item.amount,
        reverse=True,
    ):
        _append_unique_query(queries, f"{item.category} {item.description} 지출 줄이는 방법")

    current_category = indicators.main_category_shift.current_category
    if current_category:
        _append_unique_query(queries, f"{current_category} 과소비 줄이는 방법")

    action_result = (interpretation_result or {}).get("action_result")
    for mission in _iter_action_missions(action_result):
        _append_unique_query(queries, f"{mission.title} 실천 방법")

    if user_profile is not None:
        if user_profile.saving_goal_text:
            _append_unique_query(queries, f"{user_profile.saving_goal_text} 목표 소비 절약 방법")
        if user_profile.job and current_category:
            _append_unique_query(queries, f"{user_profile.job} {current_category} 소비 줄이는 방법")
        if user_profile.persona:
            _append_unique_query(queries, f"{user_profile.persona} 소비 습관 개선 방법")

    _append_unique_query(queries, _DEFAULT_RETRIEVAL_QUERY)
    return queries[:max_queries]


def retrieve_feedback_contexts(
    queries: Sequence[str],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 3,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
) -> list[RetrievedAdviceContext]:
    """RAG 검색 질의 목록을 실행해 최종 피드백에 사용할 문서 청크를 수집한다."""
    contexts: list[RetrievedAdviceContext] = []
    seen_contexts: set[tuple[str, int | None, str]] = set()
    for query in queries:
        records = retrieve_context_records(
            query,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            source_files=source_files,
            settings=settings,
        )
        for record in records:
            page_number = record.get("page_number")
            actual_page_number = page_number if isinstance(page_number, int) else None
            source = str(record["source"])
            content = str(record["content"])
            dedupe_key = (source, actual_page_number, content)
            if dedupe_key in seen_contexts:
                continue
            seen_contexts.add(dedupe_key)
            contexts.append(
                RetrievedAdviceContext(
                    query=query,
                    source=source,
                    content=content,
                    page_number=actual_page_number,
                )
            )
    return contexts


def make_daily_feedback_input(
    *,
    user_data: UserSpendingData,
    interpretation_result: dict[str, object],
    advice_contexts: Sequence[RetrievedAdviceContext],
    user_profile: object | None = None,
    memory_context: object | None = None,
) -> dict[str, str]:
    """최종 일일 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    return {
        "daily_json": user_data.model_dump_json(indent=2),
        "interpretation_json": serialize_interpretation_result(interpretation_result),
        "retrieved_contexts": serialize_advice_contexts(advice_contexts),
        "user_profile_json": serialize_context_object(user_profile or {}),
        "memory_context_json": serialize_context_object(memory_context or {}),
    }


def _build_error_result(
    *,
    member_id: int,
    analysis_date: date,
    error: str,
    daily_analysis: UserSpendingData | None = None,
    interpretation_result: dict[str, object] | None = None,
    user_profile: UserProfileContext | None = None,
    memory_context: DailyFeedbackMemoryContext | None = None,
    retrieval_queries: Sequence[str] | None = None,
    retrieved_contexts: Sequence[RetrievedAdviceContext] | None = None,
) -> DailyFeedbackServiceResult:
    """서비스 중간 실패를 호출자가 확인할 수 있는 결과 모델로 변환한다."""
    return DailyFeedbackServiceResult(
        member_id=member_id,
        analysis_date=str(analysis_date),
        daily_analysis=daily_analysis,
        interpretation_result=_to_json_object(interpretation_result or {}),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=list(retrieval_queries or []),
        retrieved_contexts=list(retrieved_contexts or []),
        error=error,
    )


def generate_daily_feedback(
    *,
    member_id: int = 1,
    analysis_date: str | date = "2024-04-01",
    previous_date: str | date = "2024-03-31",
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
) -> DailyFeedbackServiceResult:
    """일일 소비 분석, 해석, RAG 검색, 최종 잔소리 피드백 생성을 한 번에 실행한다."""
    config = settings or get_settings()
    analysis_day = _parse_analysis_date(analysis_date)
    previous_day = _parse_analysis_date(previous_date)

    chat_model_error = config.chat_model_error
    if chat_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            analysis_date=analysis_day,
            error=chat_model_error,
        )

    embedding_model_error = config.embedding_model_error
    if embedding_model_error is not None:
        return _build_error_result(
            member_id=member_id,
            analysis_date=analysis_day,
            error=embedding_model_error,
        )

    user_profile: UserProfileContext | None = None
    memory_context: DailyFeedbackMemoryContext | None = None
    try:
        daily_payload = build_daily_consumption_analysis_json(
            member_id=member_id,
            analysis_date=analysis_day,
            previous_date=previous_day,
            settings=config,
        )
        user_data = parse_user_spending_data(daily_payload)
        user_profile = load_user_profile_context(
            member_id=member_id,
            settings=config,
        )
        interpretation_chain = build_spending_analysis_chain(
            settings=config,
            temperature=interpretation_temperature,
        )
        interpretation_result = interpretation_chain.invoke(
            make_spending_analysis_input(
                user_data,
                user_profile=user_profile,
            )
        )
        retrieval_queries = build_feedback_retrieval_queries(
            user_data,
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
                analysis_date=analysis_day,
                error="missing_documents",
                daily_analysis=user_data,
                interpretation_result=interpretation_result,
                retrieval_queries=retrieval_queries,
            )

        memory_context = load_daily_feedback_memory_context(
            member_id=member_id,
            analysis_date=analysis_day,
            settings=config,
        )
        feedback_chain = build_daily_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
            persona_key=persona_key,
        )
        feedback = feedback_chain.invoke(
            make_daily_feedback_input(
                user_data=user_data,
                interpretation_result=interpretation_result,
                advice_contexts=advice_contexts,
                user_profile=user_profile,
                memory_context=memory_context,
            )
        )
        feedback_result = (
            feedback
            if isinstance(feedback, DailyFeedbackResult)
            else DailyFeedbackResult.model_validate(feedback)
        )
        save_daily_feedback_session(
            member_id=member_id,
            analysis_date=analysis_day,
            daily_analysis=user_data,
            feedback=feedback_result,
            settings=config,
        )
        _refresh_daily_user_memory_if_possible(
            member_id=member_id,
            settings=config,
        )
    except Exception as exc:
        return _build_error_result(
            member_id=member_id,
            analysis_date=analysis_day,
            error=str(exc),
        )

    return DailyFeedbackServiceResult(
        member_id=member_id,
        analysis_date=str(analysis_day),
        feedback=feedback_result,
        daily_analysis=user_data,
        interpretation_result=_to_json_object(interpretation_result),
        user_profile=user_profile,
        memory_context=memory_context,
        retrieval_queries=retrieval_queries,
        retrieved_contexts=advice_contexts,
    )
