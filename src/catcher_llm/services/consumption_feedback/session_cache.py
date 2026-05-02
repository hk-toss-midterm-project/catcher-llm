from __future__ import annotations

from catcher_llm.db.models import SessionModel

_EMPTY_JSON_TEXTS = {"[]", "{}", "null"}


def _has_text(value: str | None) -> bool:
    """세션 캐시 판정에 사용할 문자열 값이 실제 내용을 담고 있는지 확인한다."""
    if value is None:
        return False
    stripped = value.strip()
    return stripped != "" and stripped not in _EMPTY_JSON_TEXTS


def has_stored_feedback_payload(session_row: SessionModel | None) -> bool:
    """세션 행에 재사용 가능한 피드백 생성 산출물이 저장되어 있는지 판단한다."""
    if session_row is None:
        return False

    return any(
        _has_text(value)
        for value in (
            session_row.feedback_message,
            session_row.feedback_reason,
            session_row.todo_tomorrow,
            session_row.analysis_result,
        )
    )


__all__ = ["has_stored_feedback_payload"]
