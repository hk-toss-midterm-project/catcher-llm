from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

from sqlalchemy import select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import SessionModel, UserFeedbackMemoryModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.user_data_service import ensure_user_database

type FeedbackReactionValue = Literal["like", "dislike"]
type FeedbackPeriodType = Literal["daily", "weekly", "monthly"]

_VALID_FEEDBACK_REACTIONS: set[str] = {"like", "dislike"}
_VALID_FEEDBACK_PERIOD_TYPES: set[str] = {"daily", "weekly", "monthly"}


@dataclass(frozen=True, slots=True)
class FeedbackReactionSaveResult:
    """피드백 반응 저장 후 화면에서 확인할 수 있는 세션 반응 상태를 표현한다."""

    member_id: int
    analysis_date: str
    period_type: FeedbackPeriodType
    reaction: FeedbackReactionValue
    reason: str | None


def _normalize_analysis_date(analysis_date: date | str) -> str:
    """날짜 객체와 문자열을 세션 테이블의 analysis_date 저장 형식으로 정규화한다."""
    if isinstance(analysis_date, date):
        return analysis_date.isoformat()
    return analysis_date


def _validate_reaction(reaction: str) -> FeedbackReactionValue:
    """피드백 반응 값이 저장 가능한 좋아요/싫어요 값인지 검증한다."""
    if reaction not in _VALID_FEEDBACK_REACTIONS:
        msg = "피드백 반응은 'like' 또는 'dislike'만 저장할 수 있습니다."
        raise ValueError(msg)
    return cast("FeedbackReactionValue", reaction)


def _validate_period_type(period_type: str) -> FeedbackPeriodType:
    """피드백 기간 유형이 일·주·월 세션 중 하나인지 검증한다."""
    if period_type not in _VALID_FEEDBACK_PERIOD_TYPES:
        msg = "피드백 기간 유형은 'daily', 'weekly', 'monthly'만 저장할 수 있습니다."
        raise ValueError(msg)
    return cast("FeedbackPeriodType", period_type)


def _normalize_reason(reason: str | None) -> str | None:
    """사용자가 입력한 반응 사유를 공백 제거 후 빈 값이면 None으로 정리한다."""
    if reason is None:
        return None
    stripped = reason.strip()
    if stripped == "":
        return None
    return stripped


def save_session_feedback_reaction(
    *,
    member_id: int,
    analysis_date: date | str,
    period_type: str,
    reaction: str,
    reason: str | None,
    settings: Settings | None = None,
) -> FeedbackReactionSaveResult:
    """피드백 반응과 선택적으로 입력된 반응 사유를 세션 테이블에 저장한다."""
    config = settings or get_settings()
    normalized_period_type = _validate_period_type(period_type)
    normalized_reaction = _validate_reaction(reaction)
    normalized_date = _normalize_analysis_date(analysis_date)
    normalized_reason = _normalize_reason(reason)

    ensure_user_database(config)
    with session_scope(config) as db_session:
        session_row = db_session.scalar(
            select(SessionModel).where(
                SessionModel.user_id == member_id,
                SessionModel.analysis_date == normalized_date,
                SessionModel.period_type == normalized_period_type,
            )
        )
        if session_row is None:
            session_row = SessionModel(
                user_id=member_id,
                analysis_date=normalized_date,
                period_type=normalized_period_type,
            )
            db_session.add(session_row)
            db_session.flush()

        session_row.feedback_reaction = normalized_reaction
        session_row.feedback_reaction_reason = normalized_reason
        db_session.flush()

        # dislike + reason 이 있을 때 user_memories 에 이유 텍스트만 누적 후 구체성 순 재정렬
        if normalized_reaction == "dislike" and normalized_reason:
            from sqlalchemy import select as _select

            memory_row = db_session.scalar(
                _select(UserMemoryModel).where(
                    UserMemoryModel.user_id == member_id,
                    UserMemoryModel.period_type == normalized_period_type,
                )
            )
            if memory_row is None:
                memory_row = UserMemoryModel(
                    user_id=member_id,
                    period_type=normalized_period_type,
                    summary="",
                )
                db_session.add(memory_row)
                db_session.flush()
            # 기존 피드백 메모리 항목 조회 (user_feedback_memories 테이블)
            existing_fb_rows = list(
                db_session.scalars(
                    select(UserFeedbackMemoryModel)
                    .where(
                        UserFeedbackMemoryModel.user_id == memory_row.user_id,
                        UserFeedbackMemoryModel.period_type == normalized_period_type,
                    )
                    .order_by(UserFeedbackMemoryModel.created_at.asc())
                )
            )
            existing_entries = [r.reason for r in existing_fb_rows]
            all_entries = existing_entries + [normalized_reason]

            # 항목이 2개 이상일 때만 LLM 구체성 기반 재정렬
            if len(all_entries) >= 2:
                from catcher_llm.chains.consumption_feedback import (
                    build_feedback_memory_rank_chain,
                )

                rank_chain = build_feedback_memory_rank_chain(config, temperature=0.0)
                ranked_text = rank_chain.invoke({"entries": "\n".join(all_entries)})
                ranked_lines = [line.strip() for line in ranked_text.splitlines() if line.strip()]
                # LLM 재정렬이 성공하면 기존 행 삭제 후 순서대로 재삽입
                if len(ranked_lines) == len(all_entries):
                    for fb_row in existing_fb_rows:
                        db_session.delete(fb_row)
                    db_session.flush()
                    for reason_text in ranked_lines:
                        db_session.add(UserFeedbackMemoryModel(
                            user_id=memory_row.user_id,
                            period_type=normalized_period_type,
                            reason=reason_text,
                        ))
                else:
                    # 재정렬 실패 시 신규 항목만 추가
                    db_session.add(UserFeedbackMemoryModel(
                        user_id=memory_row.user_id,
                        period_type=normalized_period_type,
                        reason=normalized_reason,
                    ))
            else:
                db_session.add(UserFeedbackMemoryModel(
                    user_id=memory_row.user_id,
                    period_type=normalized_period_type,
                    reason=normalized_reason,
                ))
            db_session.flush()

        return FeedbackReactionSaveResult(
            member_id=session_row.user_id,
            analysis_date=session_row.analysis_date,
            period_type=normalized_period_type,
            reaction=normalized_reaction,
            reason=session_row.feedback_reaction_reason,
        )
