from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.config.settings import get_settings
from catcher_llm.db.models import SessionModel
from catcher_llm.prompts.persona_prompt import PERSONAS
from catcher_llm.schemas.consumption_feedback import (
    DailyFeedbackAction,
    DailyFeedbackEvidence,
    DailyFeedbackMemoryContext,
    DailyFeedbackServiceResult,
    RetrievedAdviceContext,
    UserProfileContext,
    UserSpendingData,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    _extract_daily_total_summary,
    generate_daily_feedback,
    load_all_daily_sessions,
    load_daily_session_for_date,
)
from catcher_llm.services.consumption_feedback.interpretation import extract_feedback_reason_summary
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)
from catcher_llm.ui.feedback_reaction import render_feedback_reaction_controls

_DAILY_PERSONA_KEY = "daily_persona"
_DAILY_REGEN_KEY = "daily_force_regen"

settings = get_settings()


def _models_to_frame(models: Sequence[BaseModel]) -> pd.DataFrame:
    """Pydantic 모델 목록을 Streamlit 표로 렌더링할 DataFrame으로 변환한다."""
    return pd.DataFrame([model.model_dump() for model in models])


def _format_amount(value: int | float) -> str:
    return f"{value:,.0f}원"


def _format_percent(value: int | float) -> str:
    return f"{value:,.2f}%"


def _render_evidence_table(evidences: Sequence[DailyFeedbackEvidence]) -> None:
    frame = _models_to_frame(evidences)
    if frame.empty:
        st.info("표시할 피드백 근거가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_action_table(actions: Sequence[DailyFeedbackAction]) -> None:
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
            f"{index}. {context.query} | {context.source}{page_label}", expanded=index == 1
        ):
            st.write(context.content)


def _render_daily_analysis_summary(daily_analysis: UserSpendingData | None) -> None:
    if daily_analysis is None:
        return
    frictionless_spending = daily_analysis.payment_behavior_analysis.frictionless_spending
    transaction_density = daily_analysis.payment_behavior_analysis.transaction_density
    st.subheader("지출 마찰력 및 결제 밀도")
    cols = st.columns(5)
    cols[0].metric("마찰력 없는 지출 비중", _format_percent(frictionless_spending.ratio_percent))
    cols[1].metric("마찰력 없는 지출액", _format_amount(frictionless_spending.total_amount))
    cols[2].metric("마찰력 없는 결제 건수", f"{frictionless_spending.transaction_count}건")
    cols[3].metric("오늘 결제 횟수", f"{transaction_density.transaction_count}건")
    cols[4].metric(
        "건당 평균 금액", _format_amount(transaction_density.average_amount_per_transaction)
    )


def _render_cached_daily_session(session_row: SessionModel) -> None:
    """DB에 저장된 daily 세션 데이터를 화면에 표시한다."""
    # 피드백 메시지
    if session_row.feedback_message:
        st.write(session_row.feedback_message)
    if session_row.todo_tomorrow:
        st.info(session_row.todo_tomorrow)

    # 분석 지표 (analysis_result JSON → UserSpendingData)
    if session_row.analysis_result:
        try:
            daily_analysis = UserSpendingData.model_validate_json(session_row.analysis_result)
            _render_daily_analysis_summary(daily_analysis)
            with st.expander("일일 분석 JSON"):
                st.json(daily_analysis.model_dump())
        except Exception:
            pass

    # 피드백 근거 (feedback_reason JSON → DailyFeedbackEvidence list)
    if session_row.feedback_reason:
        try:
            evidences = [
                DailyFeedbackEvidence.model_validate(e)
                for e in json.loads(session_row.feedback_reason)
            ]
            st.subheader("피드백 근거")
            _render_evidence_table(evidences)
        except Exception:
            pass

    render_feedback_reaction_controls(
        member_id=session_row.user_id,
        analysis_date=session_row.analysis_date,
        period_type="daily",
        settings=settings,
        key_prefix=f"daily_feedback_{session_row.user_id}_{session_row.analysis_date}",
        current_reaction=session_row.feedback_reaction,
        current_reason=session_row.feedback_reaction_reason,
    )


def _render_profile_and_memory_context(
    *,
    user_profile: UserProfileContext | None,
    memory_context: DailyFeedbackMemoryContext | None,
) -> None:
    with st.expander("사용자 프로필 JSON"):
        st.json(user_profile.model_dump() if user_profile is not None else {})
    with st.expander("메모리/세션 컨텍스트 JSON"):
        st.json(memory_context.model_dump() if memory_context is not None else {})


def _render_session_history(member_id: int) -> None:
    sessions = load_all_daily_sessions(member_id=member_id, settings=settings)
    if not sessions:
        st.info("저장된 일일 세션 기록이 없습니다.")
        return
    rows = []
    for s in sessions:
        rows.append(
            {
                "날짜": s.analysis_date,
                "오늘 지출": _extract_daily_total_summary(s.analysis_result),
                "피드백 메시지": (s.feedback_message or "-"),
                "피드백 핵심 근거": extract_feedback_reason_summary(s.feedback_reason),
                "다음 미션": s.todo_tomorrow or "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_daily_feedback_result(
    *,
    result: DailyFeedbackServiceResult,
    member_id: int,
) -> None:
    """새로 생성한 일일 피드백 결과와 관련 컨텍스트를 화면에 표시한다."""
    if result.error:
        st.error(f"일일 피드백 생성 실패: {result.error}")
        if result.retrieval_queries:
            st.subheader("생성된 RAG 검색 질의")
            st.write(result.retrieval_queries)
        if result.daily_analysis is not None:
            _render_daily_analysis_summary(result.daily_analysis)
            with st.expander("일일 분석 JSON"):
                st.json(result.daily_analysis.model_dump())
        if result.interpretation_result:
            with st.expander("소비 해석 JSON"):
                st.json(result.interpretation_result)
        _render_profile_and_memory_context(
            user_profile=result.user_profile,
            memory_context=result.memory_context,
        )
        st.markdown("---")
        st.subheader("📋 저장된 일일 세션 기록")
        _render_session_history(member_id)
        st.stop()

    if result.feedback is None:
        st.warning("피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback

    st.subheader(feedback.summary_title)
    st.write(feedback.scolding_message)
    st.info(feedback.tomorrow_mission)

    _render_daily_analysis_summary(result.daily_analysis)

    st.subheader("피드백 근거")
    _render_evidence_table(feedback.key_evidences)

    st.subheader("오늘 할 일")
    _render_action_table(feedback.action_items)

    st.subheader("RAG 검색 질의")
    st.write(result.retrieval_queries)

    st.subheader("검색된 문서 근거")
    _render_contexts(result.retrieved_contexts)

    _render_profile_and_memory_context(
        user_profile=result.user_profile,
        memory_context=result.memory_context,
    )

    with st.expander("일일 분석 JSON"):
        if result.daily_analysis is not None:
            st.json(result.daily_analysis.model_dump())

    with st.expander("소비 해석 JSON"):
        st.json(result.interpretation_result or {})

    with st.expander("최종 피드백 JSON"):
        st.json(feedback.model_dump())

    render_feedback_reaction_controls(
        member_id=member_id,
        analysis_date=result.analysis_date,
        period_type="daily",
        settings=settings,
        key_prefix=f"daily_feedback_{member_id}_{result.analysis_date}",
    )

    st.markdown("---")
    st.subheader("📋 저장된 일일 세션 기록")
    _render_session_history(member_id)


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("일일 분석, 해석, RAG 조회, 최종 소비 피드백을 한 번에 실행합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Raw data: `{settings.raw_data_dir}`")

# ── 제목 ──────────────────────────────────────────────────────────────────────
st.title("📣 일일 피드백")
st.caption("generate_daily_feedback 서비스를 실행해 최종 일일 소비 잔소리 결과를 점검합니다.")

# ── 날짜 / Member 선택 ────────────────────────────────────────────────────────
render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_feedback_day",
    )
previous_day = analysis_day - timedelta(days=1)

# ── RAG / 페르소나 설정 ───────────────────────────────────────────────────────
retrieval_controls = st.columns(4)
chunk_size = retrieval_controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_controls[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_controls[3].number_input("Max queries", min_value=1, value=4, step=1)

if _DAILY_PERSONA_KEY not in st.session_state:
    st.session_state[_DAILY_PERSONA_KEY] = None

_daily_persona_label_to_key = {info["label"]: key for key, info in PERSONAS.items()}
_daily_persona_labels = list(_daily_persona_label_to_key.keys())
_daily_current_key = st.session_state[_DAILY_PERSONA_KEY]
_daily_expander_title = (
    "🎭 페르소나"
    if _daily_current_key is None
    else f"🎭 페르소나 — {PERSONAS[_daily_current_key]['label']}"
)
_daily_current_index = (
    None
    if _daily_current_key is None
    else _daily_persona_labels.index(PERSONAS[_daily_current_key]["label"])
)
with st.expander(_daily_expander_title, expanded=False):
    _daily_selected = st.radio(
        "피드백을 전달할 페르소나를 선택하세요",
        options=_daily_persona_labels,
        index=_daily_current_index,
        key=f"{_DAILY_PERSONA_KEY}_radio",
    )
    if _daily_selected is not None:
        st.session_state[_DAILY_PERSONA_KEY] = _daily_persona_label_to_key[_daily_selected]

# ── 캐시 확인 → 있으면 불러오기, 없으면 생성 ──────────────────────────────────
_regen_date_key = f"{_DAILY_REGEN_KEY}_{member_id}_{analysis_day}"
_force_regen = st.session_state.get(_regen_date_key, False)

cached_session = load_daily_session_for_date(
    member_id=int(member_id),
    analysis_date=analysis_day,
    settings=settings,
)
_has_cache = cached_session is not None and cached_session.feedback_message

if _has_cache and not _force_regen:
    _render_cached_daily_session(cached_session)

    if st.button("일일 피드백 재생성", width="stretch"):
        st.session_state[_regen_date_key] = True
        st.rerun()

    st.markdown("---")
    st.subheader("📋 저장된 일일 세션 기록")
    _render_session_history(int(member_id))

else:
    # 강제 재생성 플래그 초기화
    should_generate = bool(_force_regen)
    if _force_regen:
        st.session_state.pop(_regen_date_key, None)

    if st.button("일일 피드백 생성", width="stretch") or should_generate:
        spinner_label = "일일 피드백 재생성 중..." if should_generate else "일일 피드백 생성 중..."
        with st.spinner(spinner_label):
            result = generate_daily_feedback(
                member_id=int(member_id),
                analysis_date=analysis_day,
                previous_date=previous_day,
                settings=settings,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                top_k=int(top_k),
                max_queries=int(max_queries),
                persona_key=st.session_state.get(_DAILY_PERSONA_KEY),
            )
        _render_daily_feedback_result(result=result, member_id=int(member_id))
