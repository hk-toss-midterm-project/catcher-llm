from __future__ import annotations

from datetime import date
from typing import Literal, cast

import streamlit as st

from catcher_llm.config.settings import Settings
from catcher_llm.services.consumption_feedback.feedback_reaction import (
    save_session_feedback_reaction,
)

type FeedbackReactionValue = Literal["like", "dislike"]

_REACTION_LABELS: dict[FeedbackReactionValue, str] = {
    "like": "좋아요",
    "dislike": "싫어요",
}
_REACTION_OPTIONS: tuple[FeedbackReactionValue, FeedbackReactionValue] = ("like", "dislike")


def _normalize_reaction(value: object) -> FeedbackReactionValue | None:
    """세션 상태나 DB에서 읽은 반응 값을 UI에서 쓰는 제한된 값으로 정규화한다."""
    if value in _REACTION_LABELS:
        return cast("FeedbackReactionValue", value)
    return None


def _build_reaction_key(key_prefix: str, suffix: str) -> str:
    """피드백 반응 UI가 페이지와 날짜별로 충돌하지 않는 Streamlit 키를 만든다."""
    return f"{key_prefix}_feedback_reaction_{suffix}"


def _get_reason_initial_value(
    *,
    selected_reaction: FeedbackReactionValue | None,
    current_reaction: FeedbackReactionValue | None,
    current_reason: str | None,
) -> str:
    """현재 선택된 반응에 맞춰 반응 사유 입력란의 초기값을 결정한다."""
    if selected_reaction is not None and selected_reaction == current_reaction:
        return current_reason or ""
    return ""


def _save_reaction_click(
    *,
    member_id: int,
    analysis_date: date | str,
    period_type: str,
    reaction: FeedbackReactionValue,
    reason: str,
    settings: Settings,
) -> str:
    """좋아요/싫어요 버튼 클릭으로 선택된 반응을 즉시 세션 테이블에 저장한다."""
    saved_reaction = save_session_feedback_reaction(
        member_id=member_id,
        analysis_date=analysis_date,
        period_type=period_type,
        reaction=reaction,
        reason=reason,
        settings=settings,
    )
    return saved_reaction.reason or ""


def render_feedback_reaction_controls(
    *,
    member_id: int,
    analysis_date: date | str,
    period_type: str,
    settings: Settings,
    key_prefix: str,
    current_reaction: str | None = None,
    current_reason: str | None = None,
) -> None:
    """피드백 좋아요/싫어요 반응과 반응 사유 저장 UI를 렌더링한다."""
    stored_reaction = _normalize_reaction(current_reaction)
    selected_key = _build_reaction_key(key_prefix, "selected")
    reason_key = _build_reaction_key(key_prefix, "reason")

    if selected_key not in st.session_state:
        st.session_state[selected_key] = stored_reaction

    selected_reaction = _normalize_reaction(st.session_state.get(selected_key))
    st.markdown("---")
    st.subheader("피드백 반응")
    if stored_reaction is not None:
        st.caption(f"저장된 반응: {_REACTION_LABELS[stored_reaction]}")

    reaction_columns = st.columns(2)
    for reaction, column in zip(_REACTION_OPTIONS, reaction_columns, strict=True):
        button_type = "primary" if selected_reaction == reaction else "secondary"
        if column.button(
            _REACTION_LABELS[reaction],
            key=_build_reaction_key(key_prefix, reaction),
            type=button_type,
            width="stretch",
        ):
            click_reason = (current_reason or "") if reaction == stored_reaction else ""
            try:
                saved_reason = _save_reaction_click(
                    member_id=member_id,
                    analysis_date=analysis_date,
                    period_type=period_type,
                    reaction=reaction,
                    reason=click_reason,
                    settings=settings,
                )
            except Exception as exc:
                st.error(f"피드백 반응 저장 실패: {exc}")
                return
            st.session_state[selected_key] = reaction
            st.session_state[reason_key] = saved_reason
            selected_reaction = reaction
            st.success(f"{_REACTION_LABELS[reaction]} 반응을 저장했습니다.")

    # 싫어요를 선택한 경우에만 이유 입력란 표시
    if selected_reaction != "dislike":
        return

    if reason_key not in st.session_state:
        st.session_state[reason_key] = _get_reason_initial_value(
            selected_reaction=selected_reaction,
            current_reaction=stored_reaction,
            current_reason=current_reason,
        )

    st.text_area(
        "어떤 점이 마음에 들지 않으셨나요?",
        key=reason_key,
        placeholder="피드백이 아쉬웠던 이유를 적어주세요.",
    )
    if st.button(
        "전송",
        key=_build_reaction_key(key_prefix, "submit_reason"),
        width="stretch",
    ):
        submitted_reason = str(st.session_state.get(reason_key, ""))
        try:
            save_session_feedback_reaction(
                member_id=member_id,
                analysis_date=analysis_date,
                period_type=period_type,
                reaction=selected_reaction,
                reason=submitted_reason,
                settings=settings,
            )
        except Exception as exc:
            st.error(f"반응 사유 저장 실패: {exc}")
            return
        st.success("반응 사유를 저장했습니다.")
