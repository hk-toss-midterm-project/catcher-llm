from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from langchain_core.runnables import Runnable
from pydantic import BaseModel
from sqlalchemy import select

from catcher_llm.chains.consumption_feedback import (
    build_balanced_spending_analysis_chain,
    build_daily_feedback_chain,
    build_memory_summary_chain,
    build_spending_analysis_chain,
    build_unified_spending_analysis_chain,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import SessionModel, UserFeedbackMemoryModel, UserMemoryModel, UserModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    CauseAnalysisResult,
    DailyFeedbackMemoryContext,
    DailyFeedbackResult,
    DailyFeedbackServiceResult,
    DailyFeedbackSessionContext,
    InterventionTarget,
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
from catcher_llm.services.consumption_feedback.financial_context import (
    load_user_financial_context,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_feedback_reason_summary,
    extract_spending_indicators,
    make_spending_analysis_input,
    parse_user_spending_data,
    truncate_context_text,
)
from catcher_llm.services.consumption_feedback.ratio_guard import (
    coerce_ratio_context_message,
    collect_ratio_context_warnings,
)
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.rag.core import retrieve_context_records
from catcher_llm.services.user_data_service import ensure_user_database

_DEFAULT_RETRIEVAL_QUERY = "일일 소비 절약 실천 방법"
_DAILY_MEMORY_PERIOD_TYPE = "daily"
_DEFAULT_RECENT_SESSION_LIMIT = 7
_DEFAULT_MEMORY_SESSION_LIMIT = 14
_DEFAULT_USEFULNESS_THRESHOLD = 0.25
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_GENERIC_QUERY_TOKENS = {
    "방법",
    "소비",
    "지출",
    "절약",
    "실천",
    "줄이는",
    "줄이기",
    "관리",
    "개선",
    "목표",
    "근거",
    "피드백",
    "전략",
}
_FEEDBACK_DOCUMENT_KIND_DIRS: dict[DocumentKind, Path] = {
    DocumentKind.SAVING_TIPS: Path("pdf") / "saving_tips",
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK: Path("pdf") / "catcher_consumption_benchmark",
    DocumentKind.USER_REPORT: Path("markdown") / "users_report",
    DocumentKind.WELFARE: Path("pdf") / "welfare",
    DocumentKind.KCA_REPORT: Path("pdf") / "kca_report",
}
_DEFAULT_FEEDBACK_DOCUMENT_KINDS: tuple[DocumentKind, ...] = (
    DocumentKind.SAVING_TIPS,
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK,
    DocumentKind.USER_REPORT,
    DocumentKind.WELFARE,
    DocumentKind.KCA_REPORT,
)
_DAILY_FEEDBACK_DOCUMENT_KINDS: tuple[DocumentKind, ...] = (DocumentKind.WELFARE,)
_DOCUMENT_KIND_QUERY_SUFFIXES: dict[DocumentKind, str] = {
    DocumentKind.SAVING_TIPS: "절약 행동 실천 방법",
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK: (
        "Catcher 2018년 7~12월 한국인 300만 명 소비 벤치마크 비교 근거"
    ),
    DocumentKind.USER_REPORT: "사용자 소비 동향 비교 근거",
    DocumentKind.WELFARE: "복지 지원 정책 혜택 가능성",
    DocumentKind.KCA_REPORT: "소비자 주의사항 피해 예방 근거",
}
_MAX_COMPACT_FEEDBACK_CONTEXTS = 5
_MAX_COMPACT_CONTEXT_CONTENT_LENGTH = 500
_MAX_COMPACT_MEMORY_SESSIONS = 3
_MAX_COMPACT_MEMORY_TEXT_LENGTH = 240
_MAX_COMPACT_INTERPRETATION_ITEMS = 2
_COMPACT_DAILY_METRIC_KEYS: tuple[str, ...] = (
    "daily_total_amount",
    "daily_transaction_count",
    "daily_average_transaction_amount",
    "daily_max_transaction_amount",
    "late_night_ratio_percent",
    "daily_budget_usage_rate_percent",
    "daily_remaining_budget",
    "daily_overspend_amount",
    "daily_income_usage_rate_percent",
    "month_to_date_budget_usage_rate_percent",
    "projected_monthly_spending",
    "projected_monthly_budget_usage_rate_percent",
    "required_daily_budget_until_month_end",
    "no_spending_day",
    "daily_anomaly_score",
    "special_metrics",
)
_COMPACT_USER_PROFILE_KEYS: tuple[str, ...] = (
    "user_id",
    "job",
    "persona",
    "saving_goal_text",
    "income",
    "card_grade",
    "annual_income",
    "monthly_income",
    "target_max_spending_amount",
)
_COMPACT_MEMORY_KEYS: tuple[str, ...] = (
    "user_id",
    "period_type",
    "memory_summary",
    "user_feedback_memory",
)
_COMPACT_MEMORY_SESSION_KEYS: tuple[str, ...] = (
    "analysis_date",
    "feedback_reason",
    "todo_tomorrow",
)
type DailyFeedbackInterpretationMode = Literal["split", "balanced", "unified"]


@dataclass(frozen=True)
class DailyFeedbackTimingRecord:
    """일일 피드백 생성 단계별 실행 시간과 상태를 담는 개발용 기록이다."""

    step_key: str
    step_name: str
    elapsed_seconds: float
    status: Literal["success", "error"] = "success"
    detail: str | None = None
    error: str | None = None


DailyFeedbackTimingCallback = Callable[[DailyFeedbackTimingRecord], None]


def _emit_daily_feedback_timing(
    callback: DailyFeedbackTimingCallback | None,
    record: DailyFeedbackTimingRecord,
) -> None:
    """타이밍 수집 콜백을 호출하되 콜백 오류가 피드백 생성을 막지 않게 한다."""
    if callback is None:
        return
    try:
        callback(record)
    except Exception:
        return


def _run_timed_daily_feedback_step[T](
    *,
    step_key: str,
    step_name: str,
    operation: Callable[[], T],
    timing_callback: DailyFeedbackTimingCallback | None = None,
    detail: str | None = None,
) -> T:
    """일일 피드백 생성의 한 단계를 실행하고 성공/실패 소요 시간을 기록한다."""
    start = perf_counter()
    try:
        result = operation()
    except Exception as exc:
        _emit_daily_feedback_timing(
            timing_callback,
            DailyFeedbackTimingRecord(
                step_key=step_key,
                step_name=step_name,
                elapsed_seconds=perf_counter() - start,
                status="error",
                detail=detail,
                error=str(exc),
            ),
        )
        raise

    _emit_daily_feedback_timing(
        timing_callback,
        DailyFeedbackTimingRecord(
            step_key=step_key,
            step_name=step_name,
            elapsed_seconds=perf_counter() - start,
            status="success",
            detail=detail,
        ),
    )
    return result


def _build_daily_interpretation_chain(
    *,
    mode: DailyFeedbackInterpretationMode,
    settings: Settings,
    temperature: float,
) -> Runnable[dict[str, str], dict[str, object]]:
    """요청한 해석 모드에 맞는 일일 소비 해석 체인을 생성한다."""
    if mode == "split":
        return build_spending_analysis_chain(
            settings=settings,
            temperature=temperature,
        )
    if mode == "balanced":
        return build_balanced_spending_analysis_chain(
            settings=settings,
            temperature=temperature,
        )
    if mode == "unified":
        return build_unified_spending_analysis_chain(
            settings=settings,
            temperature=temperature,
        )

    msg = f"지원하지 않는 일일 소비 해석 모드입니다: {mode}"
    raise ValueError(msg)


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
    )


def serialize_advice_contexts(contexts: Sequence[RetrievedAdviceContext]) -> str:
    """RAG 검색 문서 청크 목록을 최종 피드백 프롬프트용 JSON 문자열로 변환한다."""
    return json.dumps(
        [context.model_dump() for context in contexts],
        ensure_ascii=False,
    )


def serialize_context_object(value: object) -> str:
    """사용자 프로필과 메모리 컨텍스트를 프롬프트 입력용 JSON 문자열로 변환한다."""
    return json.dumps(
        _to_json_object(value),
        ensure_ascii=False,
    )


def _has_json_content(value: JsonValue) -> bool:
    """프롬프트 입력에 남길 실제 값이 있는지 확인한다."""
    return value is not None and value != "" and value != [] and value != {}


def _compact_json_dumps(value: JsonObject | list[JsonObject]) -> str:
    """토큰 절약을 위해 불필요한 공백 없이 JSON 문자열을 만든다."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _select_compact_fields(payload: JsonObject, keys: Sequence[str]) -> JsonObject:
    """지정한 키 중 비어 있지 않은 값만 최종 피드백 입력으로 추린다."""
    return {key: payload[key] for key in keys if key in payload and _has_json_content(payload[key])}


def _truncate_json_text(value: JsonValue, *, max_length: int) -> JsonValue:
    """문자열 JSON 값은 한 줄 요약 길이로 제한하고 나머지는 그대로 둔다."""
    if isinstance(value, str):
        return truncate_context_text(value, max_length=max_length)
    return value


def _compact_daily_metrics(user_data: UserSpendingData) -> JsonObject:
    """최종 피드백에 필요한 일일 예산·소득·위험 지표만 추출한다."""
    raw_metrics = _to_json_object(user_data.daily_metrics)
    return _select_compact_fields(raw_metrics, _COMPACT_DAILY_METRIC_KEYS)


def _compact_daily_analysis(user_data: UserSpendingData) -> JsonObject:
    """최종 피드백용 일일 분석 JSON에서 출처와 임계값 같은 저활용 필드를 제외한다."""
    raw_anomaly = _to_json_object(user_data.anomaly_detection)
    compact_anomaly = _select_compact_fields(
        raw_anomaly,
        (
            "spike_ratio",
            "is_spike",
            "high_spending_threshold",
            "high_spending_items",
        ),
    )
    return {
        "member_id": user_data.member_id,
        "analysis_date": user_data.analysis_date,
        "stable_metrics": _to_json_object(user_data.stable_metrics),
        "anomaly_detection": compact_anomaly,
        "previous_day_comparison": _to_json_object(user_data.previous_day_comparison),
        "daily_comparisons": _to_json_object(user_data.daily_comparisons),
        "time_slot_analysis": _to_json_object(user_data.time_slot_analysis),
        "payment_behavior_analysis": _to_json_object(user_data.payment_behavior_analysis),
        "daily_metrics": _compact_daily_metrics(user_data),
    }


def _compact_list_value(value: JsonValue) -> JsonValue:
    """리스트 값은 상위 일부 항목만 남기고 긴 문자열은 짧게 자른다."""
    if not isinstance(value, list):
        return _compact_nested_value(value)
    compact_items = [
        _compact_nested_value(item) for item in value[:_MAX_COMPACT_INTERPRETATION_ITEMS]
    ]
    return [item for item in compact_items if _has_json_content(item)]


def _compact_nested_value(value: JsonValue) -> JsonValue:
    """해석 결과 JSON을 재귀적으로 줄여 최종 피드백 판단에 필요한 정보만 남긴다."""
    if isinstance(value, dict):
        compact: JsonObject = {}
        for key, item in value.items():
            compact_item = (
                _compact_list_value(item) if isinstance(item, list) else _compact_nested_value(item)
            )
            if _has_json_content(compact_item):
                compact[key] = compact_item
        return compact
    if isinstance(value, str):
        return truncate_context_text(value, max_length=_MAX_COMPACT_MEMORY_TEXT_LENGTH)
    return value


def _serialize_compact_interpretation_result(
    interpretation_result: dict[str, object],
) -> str:
    """소비 해석 결과를 상위 일부 근거와 후보 중심의 축약 JSON으로 직렬화한다."""
    raw_result = _to_json_object(interpretation_result)
    compact_result: JsonObject = {}
    for key in ("pattern_result", "problem_result", "cause_result", "action_result"):
        if key in raw_result:
            compact_result[key] = _compact_nested_value(raw_result[key])
    if not compact_result:
        compact_result = cast(JsonObject, _compact_nested_value(raw_result))
    return _compact_json_dumps(compact_result)


def _advice_context_rank_key(item: tuple[int, RetrievedAdviceContext]) -> tuple[float, int]:
    """RAG 근거 정렬에서 유용성 점수를 우선하고 같은 점수에서는 기존 순서를 유지한다."""
    index, context = item
    score = context.usefulness_score if context.usefulness_score is not None else -1.0
    return (-score, index)


def _serialize_compact_advice_contexts(
    contexts: Sequence[RetrievedAdviceContext],
) -> str:
    """RAG 문서 근거를 상위 일부와 짧은 본문만 남긴 JSON 문자열로 만든다."""
    compact_contexts: list[JsonObject] = []
    ranked_contexts = sorted(enumerate(contexts), key=_advice_context_rank_key)
    for _, context in ranked_contexts[:_MAX_COMPACT_FEEDBACK_CONTEXTS]:
        compact_context: JsonObject = {
            "query": context.query,
            "source": context.source,
            "content": truncate_context_text(
                context.content,
                max_length=_MAX_COMPACT_CONTEXT_CONTENT_LENGTH,
            ),
        }
        if context.page_number is not None:
            compact_context["page_number"] = context.page_number
        if context.document_kind:
            compact_context["document_kind"] = context.document_kind
        if context.usefulness_score is not None:
            compact_context["usefulness_score"] = context.usefulness_score
        compact_contexts.append(compact_context)
    return _compact_json_dumps(compact_contexts)


def _serialize_compact_user_profile(value: object) -> str:
    """사용자 프로필에서 최종 피드백 개인화에 필요한 필드만 직렬화한다."""
    raw_profile = _to_json_object(value)
    return _compact_json_dumps(_select_compact_fields(raw_profile, _COMPACT_USER_PROFILE_KEYS))


def _compact_memory_session(raw_session: JsonValue) -> JsonObject:
    """최근 세션에서 과거 분석 전체 JSON을 제외하고 근거·미션 요약만 남긴다."""
    if not isinstance(raw_session, dict):
        return {}
    session = cast(JsonObject, raw_session)
    compact_session: JsonObject = {}
    for key in _COMPACT_MEMORY_SESSION_KEYS:
        value = session.get(key)
        if value is None:
            continue
        compact_value = _truncate_json_text(value, max_length=_MAX_COMPACT_MEMORY_TEXT_LENGTH)
        if _has_json_content(compact_value):
            compact_session[key] = compact_value
    return compact_session


def _serialize_compact_memory_context(value: object) -> str:
    """최종 피드백용 메모리 컨텍스트를 장기 요약과 최근 미션 중심으로 줄인다."""
    raw_memory = _to_json_object(value)
    compact_memory = _select_compact_fields(raw_memory, _COMPACT_MEMORY_KEYS)
    for key in ("memory_summary", "user_feedback_memory"):
        if key in compact_memory:
            compact_memory[key] = _truncate_json_text(
                compact_memory[key],
                max_length=_MAX_COMPACT_MEMORY_TEXT_LENGTH,
            )

    raw_sessions = raw_memory.get("recent_sessions")
    if isinstance(raw_sessions, list):
        compact_sessions = [
            compact_session
            for raw_session in raw_sessions[-_MAX_COMPACT_MEMORY_SESSIONS:]
            if (compact_session := _compact_memory_session(raw_session))
        ]
        if compact_sessions:
            compact_memory["recent_sessions"] = compact_sessions
    return _compact_json_dumps(compact_memory)


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

    financial_context = load_user_financial_context(member_id=member_id, settings=config)

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
        annual_income=financial_context.annual_income,
        monthly_income=financial_context.monthly_income,
        target_max_spending_amount=financial_context.target_max_spending_amount,
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

    # user_feedback_memories 테이블에서 거부 이유 목록 로드
    with session_scope(config) as _fb_session:
        fb_rows = list(
            _fb_session.scalars(
                select(UserFeedbackMemoryModel)
                .where(
                    UserFeedbackMemoryModel.user_id == member_id,
                    UserFeedbackMemoryModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
                )
                .order_by(UserFeedbackMemoryModel.created_at.asc())
            )
        )
    feedback_memory_text: str | None = (
        "\n".join(r.reason for r in fb_rows) + "\n" if fb_rows else None
    )

    return DailyFeedbackMemoryContext(
        user_id=member_id,
        period_type=_DAILY_MEMORY_PERIOD_TYPE,
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


def load_daily_session_for_date(
    *,
    member_id: int,
    analysis_date: date,
    settings: Settings | None = None,
) -> SessionModel | None:
    """session 테이블에서 특정 날짜의 daily 세션을 조회한다. 없으면 None을 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        return session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == str(analysis_date),
                SessionModel.period_type == _DAILY_MEMORY_PERIOD_TYPE,
            )
        )


def _build_session_list_text(sessions: Sequence[SessionModel]) -> str:
    """세션 목록을 LLM 요약 체인에 넣을 수 있는 텍스트 목록으로 직렬화한다."""
    lines = []
    for session_row in sessions:
        daily_total = _extract_daily_total_summary(session_row.analysis_result)
        reason_summary = extract_feedback_reason_summary(session_row.feedback_reason)
        tomorrow_mission = truncate_context_text(session_row.todo_tomorrow)
        line = (
            f"- {session_row.analysis_date}: "
            f"지출 {daily_total}; "
            f"핵심근거 {reason_summary}; "
            f"다음미션 {tomorrow_mission}"
        )
        lines.append(line)
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


def _coerce_feedback_document_kind(document_kind: DocumentKind | str) -> DocumentKind:
    """문서 종류 입력을 피드백 RAG에서 사용할 DocumentKind 값으로 정규화한다."""
    if isinstance(document_kind, DocumentKind):
        return document_kind
    return DocumentKind(document_kind)


def _normalize_feedback_document_kinds(
    document_kinds: Sequence[DocumentKind | str] | None,
) -> list[DocumentKind]:
    """피드백 RAG 검색에 사용할 문서 종류 목록을 중복 없이 정규화한다."""
    raw_kinds: Sequence[DocumentKind | str] = (
        _DEFAULT_FEEDBACK_DOCUMENT_KINDS if document_kinds is None else document_kinds
    )
    normalized_kinds: list[DocumentKind] = []
    for raw_kind in raw_kinds:
        document_kind = _coerce_feedback_document_kind(raw_kind)
        if document_kind not in _FEEDBACK_DOCUMENT_KIND_DIRS:
            continue
        if document_kind not in normalized_kinds:
            normalized_kinds.append(document_kind)
    return normalized_kinds


def _build_feedback_query_texts(queries: Sequence[str]) -> list[str]:
    """사용자 기반 RAG 검색 질의를 정리하되 서로 다른 검색 의도를 합치지 않는다."""
    normalized_queries = [" ".join(query.split()) for query in queries if query.strip()]
    if not normalized_queries:
        return [_DEFAULT_RETRIEVAL_QUERY]
    return list(dict.fromkeys(normalized_queries))


def _build_document_kind_query(document_kind: DocumentKind, query_summary: str) -> str:
    """문서 종류의 역할을 반영한 피드백 RAG 검색 질의를 만든다."""
    suffix = _DOCUMENT_KIND_QUERY_SUFFIXES[document_kind]
    return f"[{document_kind.value}] {query_summary} {suffix}"


def _get_document_kind_raw_data_dir(
    settings: Settings,
    document_kind: DocumentKind,
) -> Path:
    """문서 종류별로 분리된 원본 디렉터리 경로를 반환한다."""
    return settings.raw_data_dir / _FEEDBACK_DOCUMENT_KIND_DIRS[document_kind]


def _extract_usefulness_tokens(text: str) -> set[str]:
    """유용성 판단에 사용할 검색어 토큰을 추출하고 지나치게 일반적인 단어를 제외한다."""
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text)}
    specific_tokens = tokens - _GENERIC_QUERY_TOKENS
    return specific_tokens or tokens


def _score_context_usefulness(query: str, content: str) -> float:
    """검색 질의와 문서 청크의 토큰 겹침 비율로 문서 근거의 유용성 점수를 계산한다."""
    query_tokens = _extract_usefulness_tokens(query)
    if not query_tokens:
        return 0.0
    content_tokens = {token.lower() for token in _TOKEN_PATTERN.findall(content)}
    if not content_tokens:
        return 0.0
    matched_tokens = query_tokens & content_tokens
    return round(len(matched_tokens) / len(query_tokens), 4)


def _build_usefulness_reason(
    score: float,
    threshold: float,
    *,
    is_fallback: bool = False,
) -> str:
    """유용성 점수와 기준값을 사람이 확인할 수 있는 짧은 설명으로 만든다."""
    if is_fallback:
        return (
            f"fallback: no context met threshold; best token overlap score "
            f"{score:.2f} < threshold {threshold:.2f}"
        )
    return f"query-content token overlap score {score:.2f} >= threshold {threshold:.2f}"


def _record_to_advice_context(
    *,
    record: dict[str, str | int | None],
    query: str,
    document_kind: DocumentKind | None = None,
    usefulness_score: float | None = None,
    usefulness_threshold: float | None = None,
    is_fallback: bool = False,
) -> RetrievedAdviceContext:
    """검색 record를 최종 피드백 프롬프트에 전달할 RAG 근거 모델로 변환한다."""
    page_number = record.get("page_number")
    reason = (
        _build_usefulness_reason(
            usefulness_score,
            usefulness_threshold,
            is_fallback=is_fallback,
        )
        if usefulness_score is not None and usefulness_threshold is not None
        else None
    )
    return RetrievedAdviceContext(
        query=query,
        source=str(record["source"]),
        content=str(record["content"]),
        page_number=page_number if isinstance(page_number, int) else None,
        document_kind=document_kind.value if document_kind is not None else None,
        usefulness_score=usefulness_score,
        usefulness_reason=reason,
    )


def _select_balanced_contexts(
    contexts: Sequence[RetrievedAdviceContext],
    max_contexts: int,
) -> list[RetrievedAdviceContext]:
    """문서 종류별 후보를 순환 선택해 앞선 문서 종류가 최종 결과를 독점하지 않게 한다."""
    if max_contexts <= 0:
        return []

    grouped_contexts: dict[str | None, list[RetrievedAdviceContext]] = {}
    for context in contexts:
        grouped_contexts.setdefault(context.document_kind, []).append(context)

    selected_contexts: list[RetrievedAdviceContext] = []
    while len(selected_contexts) < max_contexts:
        selected_in_round = False
        for grouped_items in grouped_contexts.values():
            if not grouped_items:
                continue
            selected_contexts.append(grouped_items.pop(0))
            selected_in_round = True
            if len(selected_contexts) >= max_contexts:
                break
        if not selected_in_round:
            break
    return selected_contexts


def _iter_action_missions(action_result: object) -> list[ActionMission]:
    """해석 결과의 개선 후보 모델 또는 dict에서 RAG 검색 후보 목록을 추출한다."""
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
    """원인 해석 모델 또는 dict에서 RAG 검색용 개입 타겟 후보를 추출한다."""
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
    """개입 타겟 후보에서 RAG 검색에 사용할 질의 문구를 선택한다."""
    return target.query_hint or target.title


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

    cause_result = (interpretation_result or {}).get("cause_result")
    for target in _iter_intervention_targets(cause_result):
        _append_unique_query(queries, f"{_intervention_target_query_text(target)} 절약 방법")

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
    document_kinds: Sequence[DocumentKind | str] | None = None,
    usefulness_threshold: float = _DEFAULT_USEFULNESS_THRESHOLD,
    settings: Settings | None = None,
) -> list[RetrievedAdviceContext]:
    """문서 종류별 RAG 검색을 실행하고 유용한 청크만 최종 피드백 근거로 수집한다."""
    if raw_data_dir is not None or source_files is not None:
        return _retrieve_feedback_contexts_from_explicit_source(
            queries,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            source_files=source_files,
            settings=settings,
        )

    config = settings or get_settings()
    query_texts = _build_feedback_query_texts(queries)
    normalized_document_kinds = _normalize_feedback_document_kinds(document_kinds)
    contexts: list[RetrievedAdviceContext] = []
    fallback_candidates: list[RetrievedAdviceContext] = []
    seen_contexts: set[tuple[str, str, int | None, str]] = set()

    for document_kind in normalized_document_kinds:
        for query_text in query_texts:
            query = _build_document_kind_query(document_kind, query_text)
            records = retrieve_context_records(
                query,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                raw_data_dir=_get_document_kind_raw_data_dir(config, document_kind),
                source_files=None,
                settings=config,
            )
            for record in records:
                content = str(record["content"])
                usefulness_score = _score_context_usefulness(query_text, content)
                page_number = record.get("page_number")
                actual_page_number = page_number if isinstance(page_number, int) else None
                source = str(record["source"])
                dedupe_key = (document_kind.value, source, actual_page_number, content)
                if dedupe_key in seen_contexts:
                    continue
                seen_contexts.add(dedupe_key)
                context = _record_to_advice_context(
                    record=record,
                    query=query,
                    document_kind=document_kind,
                    usefulness_score=usefulness_score,
                    usefulness_threshold=usefulness_threshold,
                )
                if usefulness_score < usefulness_threshold:
                    fallback_candidates.append(
                        _record_to_advice_context(
                            record=record,
                            query=query,
                            document_kind=document_kind,
                            usefulness_score=usefulness_score,
                            usefulness_threshold=usefulness_threshold,
                            is_fallback=True,
                        )
                    )
                    continue
                contexts.append(context)

    if not contexts and fallback_candidates:
        ranked_fallback_candidates = sorted(
            fallback_candidates,
            key=lambda context: context.usefulness_score or 0.0,
            reverse=True,
        )
        return _select_balanced_contexts(ranked_fallback_candidates, top_k)
    return _select_balanced_contexts(contexts, top_k)


def _retrieve_feedback_contexts_from_explicit_source(
    queries: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    raw_data_dir: Path | str | None,
    source_files: Sequence[Path] | None,
    settings: Settings | None,
) -> list[RetrievedAdviceContext]:
    """호출자가 명시한 단일 코퍼스 설정으로 기존 방식의 RAG 검색을 수행한다."""
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
                _record_to_advice_context(
                    record=record,
                    query=query,
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
    compact_input: bool = True,
) -> dict[str, str]:
    """최종 일일 피드백 체인에 전달할 분석, 해석, RAG, 개인화 컨텍스트 입력을 만든다."""
    if not compact_input:
        return {
            "daily_json": user_data.model_dump_json(),
            "interpretation_json": serialize_interpretation_result(interpretation_result),
            "retrieved_contexts": serialize_advice_contexts(advice_contexts),
            "user_profile_json": serialize_context_object(user_profile or {}),
            "memory_context_json": serialize_context_object(memory_context or {}),
        }

    return {
        "daily_json": _compact_json_dumps(_compact_daily_analysis(user_data)),
        "interpretation_json": _serialize_compact_interpretation_result(interpretation_result),
        "retrieved_contexts": _serialize_compact_advice_contexts(advice_contexts),
        "user_profile_json": _serialize_compact_user_profile(user_profile or {}),
        "memory_context_json": _serialize_compact_memory_context(memory_context or {}),
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
    interpretation_mode: DailyFeedbackInterpretationMode = "split",
    compact_feedback_input: bool = True,
    timing_callback: DailyFeedbackTimingCallback | None = None,
) -> DailyFeedbackServiceResult:
    """일일 소비 분석, 해석, RAG 검색, 최종 소비 피드백 생성을 한 번에 실행한다."""
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
        daily_payload = _run_timed_daily_feedback_step(
            step_key="daily_analysis",
            step_name="일일 소비 분석 JSON 생성",
            detail="SQLite transactions 조회와 pandas 지표 계산",
            timing_callback=timing_callback,
            operation=lambda: build_daily_consumption_analysis_json(
                member_id=member_id,
                analysis_date=analysis_day,
                previous_date=previous_day,
                settings=config,
            ),
        )
        user_data = _run_timed_daily_feedback_step(
            step_key="parse_daily_analysis",
            step_name="일일 분석 모델 검증",
            detail="분석 JSON을 UserSpendingData Pydantic 모델로 변환",
            timing_callback=timing_callback,
            operation=lambda: parse_user_spending_data(daily_payload),
        )
        user_profile = _run_timed_daily_feedback_step(
            step_key="user_profile",
            step_name="사용자 프로필 조회",
            detail="SQLite users 테이블 조회",
            timing_callback=timing_callback,
            operation=lambda: load_user_profile_context(
                member_id=member_id,
                settings=config,
            ),
        )
        interpretation_chain = _build_daily_interpretation_chain(
            mode=interpretation_mode,
            settings=config,
            temperature=interpretation_temperature,
        )
        interpretation_input = make_spending_analysis_input(
            user_data,
            user_profile=user_profile,
        )
        interpretation_result = cast(
            dict[str, object],
            _run_timed_daily_feedback_step(
                step_key="interpretation_chain",
                step_name=f"소비 해석 체인 실행 ({interpretation_mode})",
                detail=(
                    "split=기존 4회 호출, balanced=패턴/문제 분리+원인/행동 통합, "
                    "unified=전체 통합 1회 호출"
                ),
                timing_callback=timing_callback,
                operation=lambda: interpretation_chain.invoke(interpretation_input),
            ),
        )
        retrieval_queries = _run_timed_daily_feedback_step(
            step_key="retrieval_queries",
            step_name="RAG 검색 질의 생성",
            detail=f"max_queries={max_queries}",
            timing_callback=timing_callback,
            operation=lambda: build_feedback_retrieval_queries(
                user_data,
                interpretation_result=interpretation_result,
                user_profile=user_profile,
                max_queries=max_queries,
            ),
        )
        advice_contexts = _run_timed_daily_feedback_step(
            step_key="rag_retrieval",
            step_name="RAG 문서 검색",
            detail=(
                f"queries={len(retrieval_queries)}, top_k={top_k}, "
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
                document_kinds=_DAILY_FEEDBACK_DOCUMENT_KINDS,
                settings=config,
            ),
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

        memory_context = _run_timed_daily_feedback_step(
            step_key="memory_context",
            step_name="피드백 메모리 조회",
            detail="SQLite user_memories와 최근 daily session 조회",
            timing_callback=timing_callback,
            operation=lambda: load_daily_feedback_memory_context(
                member_id=member_id,
                analysis_date=analysis_day,
                settings=config,
            ),
        )
        feedback_chain = build_daily_feedback_chain(
            settings=config,
            temperature=feedback_temperature,
            persona_key=persona_key,
        )
        feedback_input = make_daily_feedback_input(
            user_data=user_data,
            interpretation_result=interpretation_result,
            advice_contexts=advice_contexts,
            user_profile=user_profile,
            memory_context=memory_context,
            compact_input=compact_feedback_input,
        )
        feedback = _run_timed_daily_feedback_step(
            step_key="feedback_chain",
            step_name="최종 피드백 체인 실행",
            detail="분석/해석/RAG/프로필/메모리 기반 구조화 LLM 호출",
            timing_callback=timing_callback,
            operation=lambda: feedback_chain.invoke(feedback_input),
        )
        feedback_result = (
            feedback
            if isinstance(feedback, DailyFeedbackResult)
            else DailyFeedbackResult.model_validate(feedback)
        )
        feedback_result.scolding_message = coerce_ratio_context_message(
            message=feedback_result.scolding_message,
            warnings=collect_ratio_context_warnings(
                user_data.stable_metrics.category_ratio_changes
            ),
            period_label="오늘",
            mission=feedback_result.tomorrow_mission,
        )
        _run_timed_daily_feedback_step(
            step_key="save_session",
            step_name="피드백 세션 저장",
            detail="SQLite session 테이블 저장 또는 갱신",
            timing_callback=timing_callback,
            operation=lambda: save_daily_feedback_session(
                member_id=member_id,
                analysis_date=analysis_day,
                daily_analysis=user_data,
                feedback=feedback_result,
                settings=config,
            ),
        )
        _run_timed_daily_feedback_step(
            step_key="refresh_memory",
            step_name="장기 메모리 요약 갱신",
            detail=f"최근 최대 {_DEFAULT_MEMORY_SESSION_LIMIT}개 세션을 LLM으로 요약",
            timing_callback=timing_callback,
            operation=lambda: _refresh_daily_user_memory_if_possible(
                member_id=member_id,
                settings=config,
            ),
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
