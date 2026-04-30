from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.config.settings import get_settings
from catcher_llm.db.models import SessionModel
from catcher_llm.prompts.persona_prompt import PERSONAS
from catcher_llm.schemas.consumption_feedback import (
    MonthlyFeedbackAction,
    MonthlyFeedbackEvidence,
    MonthlySpendingData,
    RetrievedAdviceContext,
    UserProfileContext,
)
from catcher_llm.services.consumption_feedback.monthly_feedback import (
    generate_monthly_feedback,
    load_monthly_session_for_date,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)
from catcher_llm.ui.feedback_reaction import render_feedback_reaction_controls

_MONTHLY_PERSONA_KEY = "monthly_persona"
_MONTHLY_REGEN_KEY = "monthly_force_regen"

settings = get_settings()


def _models_to_frame(models: Sequence[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([model.model_dump() for model in models])


def _format_amount(value: int | float) -> str:
    return f"{value:,.0f}원"


def _format_percent(value: int | float) -> str:
    return f"{value:,.2f}%"


def _render_evidence_table(evidences: Sequence[MonthlyFeedbackEvidence]) -> None:
    frame = _models_to_frame(evidences)
    if frame.empty:
        st.info("표시할 피드백 근거가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_action_table(actions: Sequence[MonthlyFeedbackAction]) -> None:
    frame = _models_to_frame(actions)
    if frame.empty:
        st.info("표시할 행동 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_contexts(contexts: Sequence[RetrievedAdviceContext]) -> None:
    if not contexts:
        st.info("검색된 문서 근거가 없습니다.")
        return
    for index, context in enumerate(contexts, start=1):
        page_label = f" | p.{context.page_number}" if context.page_number is not None else ""
        with st.expander(
            f"{index}. {context.query} | {context.source}{page_label}",
            expanded=index == 1,
        ):
            st.write(context.content)


def _render_monthly_analysis_summary(monthly_analysis: MonthlySpendingData | None) -> None:
    if monthly_analysis is None:
        return
    monthly_summary = monthly_analysis.monthly_summary
    fixed_variable = monthly_analysis.fixed_variable
    saving_potential = monthly_analysis.saving_potential
    st.subheader("월간 소비 핵심 지표")
    metric_columns = st.columns(5)
    metric_columns[0].metric("이번 달 총 소비", _format_amount(monthly_summary.this_month_total))
    metric_columns[1].metric("전월 대비", _format_percent(monthly_summary.diff_rate_percent))
    metric_columns[2].metric("고정비", _format_amount(fixed_variable.fixed_total))
    metric_columns[3].metric("활동일", f"{monthly_summary.active_days}일")
    metric_columns[4].metric(
        "절약 가능액",
        _format_amount(saving_potential.total_potential_saving),
    )


def _render_profile_context(user_profile: UserProfileContext | None) -> None:
    with st.expander("사용자 프로필 JSON"):
        st.json(user_profile.model_dump() if user_profile is not None else {})


def _render_cached_monthly_session(session_row: SessionModel) -> None:
    """DB에 저장된 monthly 세션 데이터를 화면에 표시한다."""
    if session_row.feedback_message:
        st.write(session_row.feedback_message)
    if session_row.todo_tomorrow:
        st.info(session_row.todo_tomorrow)
    if session_row.analysis_result:
        try:
            monthly_analysis = MonthlySpendingData.model_validate_json(session_row.analysis_result)
            _render_monthly_analysis_summary(monthly_analysis)
            with st.expander("월간 분석 JSON"):
                st.json(monthly_analysis.model_dump())
        except Exception:
            pass
    if session_row.feedback_reason:
        try:
            evidences = [
                MonthlyFeedbackEvidence.model_validate(e)
                for e in json.loads(session_row.feedback_reason)
            ]
            st.subheader("피드백 근거")
            _render_evidence_table(evidences)
        except Exception:
            pass

    render_feedback_reaction_controls(
        member_id=session_row.user_id,
        analysis_date=session_row.analysis_date,
        period_type="monthly",
        settings=settings,
        key_prefix=f"monthly_feedback_{session_row.user_id}_{session_row.analysis_date}",
        current_reaction=session_row.feedback_reaction,
        current_reason=session_row.feedback_reaction_reason,
    )


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("월간 분석, 해석, RAG 조회, 최종 월간 소비 피드백을 한 번에 실행합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Raw data: `{settings.raw_data_dir}`")


st.title("🧾 월간 피드백")
st.caption("generate_monthly_feedback 서비스를 실행해 최종 월간 소비 피드백 결과를 점검합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_month = select_month(
        "분석 월",
        default_month=DEFAULT_CALENDAR_MONTH,
        key="monthly_feedback_month",
    )

retrieval_controls = st.columns(4)
chunk_size = retrieval_controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_controls[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_controls[3].number_input("Max queries", min_value=1, value=4, step=1)

if _MONTHLY_PERSONA_KEY not in st.session_state:
    st.session_state[_MONTHLY_PERSONA_KEY] = None

_monthly_persona_label_to_key = {info["label"]: key for key, info in PERSONAS.items()}
_monthly_persona_labels = list(_monthly_persona_label_to_key.keys())
_monthly_current_key = st.session_state[_MONTHLY_PERSONA_KEY]
_monthly_expander_title = (
    "🎭 페르소나"
    if _monthly_current_key is None
    else f"🎭 페르소나 — {PERSONAS[_monthly_current_key]['label']}"
)
_monthly_current_index = (
    None
    if _monthly_current_key is None
    else _monthly_persona_labels.index(PERSONAS[_monthly_current_key]["label"])
)
with st.expander(_monthly_expander_title, expanded=False):
    _monthly_selected = st.radio(
        "피드백을 전달할 페르소나를 선택하세요",
        options=_monthly_persona_labels,
        index=_monthly_current_index,
        key=f"{_MONTHLY_PERSONA_KEY}_radio",
    )
    if _monthly_selected is not None:
        st.session_state[_MONTHLY_PERSONA_KEY] = _monthly_persona_label_to_key[_monthly_selected]

# ── 캐시 확인 → 있으면 불러오기, 없으면 생성 ──────────────────────────────────
_regen_month_key = f"{_MONTHLY_REGEN_KEY}_{member_id}_{analysis_month}"
_force_regen = st.session_state.get(_regen_month_key, False)

cached_session = load_monthly_session_for_date(
    member_id=int(member_id),
    analysis_month=analysis_month,
    settings=settings,
)
_has_cache = cached_session is not None and cached_session.feedback_message

if _has_cache and not _force_regen:
    _render_cached_monthly_session(cached_session)

    if st.button("월간 피드백 재생성", width="stretch"):
        st.session_state[_regen_month_key] = True
        st.rerun()

else:
    should_generate = bool(_force_regen)
    if _force_regen:
        st.session_state.pop(_regen_month_key, None)

    if st.button("월간 피드백 생성", width="stretch") or should_generate:
        spinner_label = "월간 피드백 재생성 중..." if should_generate else "월간 피드백 생성 중..."
        with st.spinner(spinner_label):
            result = generate_monthly_feedback(
                member_id=int(member_id),
                analysis_month=analysis_month,
                settings=settings,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                top_k=int(top_k),
                max_queries=int(max_queries),
                persona_key=st.session_state.get(_MONTHLY_PERSONA_KEY),
            )

        if result.error:
            st.error(f"월간 피드백 생성 실패: {result.error}")
            if result.retrieval_queries:
                st.subheader("생성된 RAG 검색 질의")
                st.write(result.retrieval_queries)
            _render_monthly_analysis_summary(result.monthly_analysis)
            if result.monthly_analysis is not None:
                with st.expander("월간 분석 JSON"):
                    st.json(result.monthly_analysis.model_dump())
            if result.interpretation_result:
                with st.expander("소비 해석 JSON"):
                    st.json(result.interpretation_result)
            _render_profile_context(result.user_profile)
            st.stop()

        if result.feedback is None:
            st.warning("피드백 결과가 비어 있습니다.")
            st.stop()

        feedback = result.feedback

        st.subheader(feedback.summary_title)
        st.write(feedback.feedback_message)
        st.info(feedback.next_month_mission)

        _render_monthly_analysis_summary(result.monthly_analysis)

        st.subheader("피드백 근거")
        _render_evidence_table(feedback.key_evidences)

        st.subheader("다음 달 할 일")
        _render_action_table(feedback.action_items)

        st.subheader("RAG 검색 질의")
        st.write(result.retrieval_queries)

        st.subheader("검색된 문서 근거")
        _render_contexts(result.retrieved_contexts)

        _render_profile_context(result.user_profile)

        with st.expander("월간 분석 JSON"):
            if result.monthly_analysis is not None:
                st.json(result.monthly_analysis.model_dump())

        with st.expander("소비 해석 JSON"):
            st.json(result.interpretation_result or {})

        with st.expander("최종 피드백 JSON"):
            st.json(feedback.model_dump())

        render_feedback_reaction_controls(
            member_id=int(member_id),
            analysis_date=result.analysis_month,
            period_type="monthly",
            settings=settings,
            key_prefix=f"monthly_feedback_{member_id}_{result.analysis_month}",
        )
