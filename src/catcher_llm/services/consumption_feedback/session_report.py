from __future__ import annotations

import json
from datetime import date
from typing import cast

from pydantic import BaseModel, ValidationError

from catcher_llm.db.models import SessionModel
from catcher_llm.schemas.consumption_feedback import (
    DailyFeedbackEvidence,
    DailyFeedbackResult,
    DailyFeedbackServiceResult,
    JsonObject,
    MonthlyFeedbackEvidence,
    MonthlyFeedbackResult,
    MonthlyFeedbackServiceResult,
    MonthlySpendingData,
    UserSpendingData,
    WeeklyFeedbackEvidence,
    WeeklyFeedbackResult,
    WeeklyFeedbackServiceResult,
    WeeklySpendingData,
)


def _stored_text(value: str | None, fallback: str) -> str:
    """세션에 저장된 문자열을 정리하고 비어 있으면 화면 표시용 기본 문구를 반환한다."""
    if value is None:
        return fallback

    stripped = value.strip()
    if stripped == "":
        return fallback

    return stripped


def _parse_json_objects(raw_json: str | None) -> list[JsonObject]:
    """세션 JSON 문자열에서 객체 배열만 안전하게 추출한다."""
    if raw_json is None or raw_json.strip() == "":
        return []

    try:
        loaded: object = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(loaded, list):
        return []

    return [cast(JsonObject, item) for item in loaded if isinstance(item, dict)]


def _parse_model_json[ModelT: BaseModel](
    raw_json: str | None,
    model_type: type[ModelT],
) -> ModelT | None:
    """세션에 저장된 단일 Pydantic JSON을 모델로 복원하고 실패하면 None을 반환한다."""
    if raw_json is None or raw_json.strip() == "":
        return None

    try:
        return model_type.model_validate_json(raw_json)
    except ValidationError:
        return None


def _parse_model_list[ModelT: BaseModel](
    raw_json: str | None,
    model_type: type[ModelT],
) -> list[ModelT]:
    """세션에 저장된 Pydantic 모델 배열 JSON에서 유효한 항목만 복원한다."""
    parsed: list[ModelT] = []
    for item in _parse_json_objects(raw_json):
        try:
            parsed.append(model_type.model_validate(item))
        except ValidationError:
            continue
    return parsed


def build_daily_report_result_from_session(session_row: SessionModel) -> DailyFeedbackServiceResult:
    """저장된 daily 세션 행을 일일 리포트 렌더링에 필요한 서비스 결과로 변환한다."""
    feedback = DailyFeedbackResult(
        summary_title="저장된 일일 소비 피드백",
        scolding_message=_stored_text(
            session_row.feedback_message,
            "저장된 일일 피드백 메시지가 없습니다.",
        ),
        key_evidences=_parse_model_list(session_row.feedback_reason, DailyFeedbackEvidence),
        action_items=[],
        tomorrow_mission=_stored_text(
            session_row.todo_tomorrow,
            "저장된 일일 미션이 없습니다.",
        ),
    )

    return DailyFeedbackServiceResult(
        member_id=session_row.user_id,
        analysis_date=session_row.analysis_date,
        feedback=feedback,
        daily_analysis=_parse_model_json(session_row.analysis_result, UserSpendingData),
    )


def build_weekly_report_result_from_session(
    session_row: SessionModel,
    *,
    week_end: date,
) -> WeeklyFeedbackServiceResult:
    """저장된 weekly 세션 행을 주간 리포트 렌더링에 필요한 서비스 결과로 변환한다."""
    feedback = WeeklyFeedbackResult(
        summary_title="저장된 주간 소비 피드백",
        feedback_message=_stored_text(
            session_row.feedback_message,
            "저장된 주간 피드백 메시지가 없습니다.",
        ),
        key_evidences=_parse_model_list(session_row.feedback_reason, WeeklyFeedbackEvidence),
        action_items=[],
        next_week_mission=_stored_text(
            session_row.todo_tomorrow,
            "저장된 주간 미션이 없습니다.",
        ),
    )

    return WeeklyFeedbackServiceResult(
        member_id=session_row.user_id,
        week_start=session_row.analysis_date,
        week_end=str(week_end),
        feedback=feedback,
        weekly_analysis=_parse_model_json(session_row.analysis_result, WeeklySpendingData),
    )


def build_monthly_report_result_from_session(
    session_row: SessionModel,
) -> MonthlyFeedbackServiceResult:
    """저장된 monthly 세션 행을 월간 리포트 렌더링에 필요한 서비스 결과로 변환한다."""
    feedback = MonthlyFeedbackResult(
        summary_title="저장된 월간 소비 피드백",
        feedback_message=_stored_text(
            session_row.feedback_message,
            "저장된 월간 피드백 메시지가 없습니다.",
        ),
        key_evidences=_parse_model_list(session_row.feedback_reason, MonthlyFeedbackEvidence),
        action_items=[],
        next_month_mission=_stored_text(
            session_row.todo_tomorrow,
            "저장된 월간 미션이 없습니다.",
        ),
    )

    return MonthlyFeedbackServiceResult(
        member_id=session_row.user_id,
        analysis_month=session_row.analysis_date,
        feedback=feedback,
        monthly_analysis=_parse_model_json(session_row.analysis_result, MonthlySpendingData),
    )


__all__ = [
    "build_daily_report_result_from_session",
    "build_monthly_report_result_from_session",
    "build_weekly_report_result_from_session",
]
