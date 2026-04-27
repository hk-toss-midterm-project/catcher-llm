from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import cast

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import (
    DailyFeedbackAction,
    DailyFeedbackEvidence,
    RetrievedAdviceContext,
)
from catcher_llm.services.consumption_feedback.daily_feedback import generate_daily_feedback

settings = get_settings()


def _models_to_frame(models: Sequence[BaseModel]) -> pd.DataFrame:
    """Pydantic 모델 목록을 Streamlit 표로 렌더링할 DataFrame으로 변환한다."""
    return pd.DataFrame([model.model_dump() for model in models])


def _render_evidence_table(evidences: Sequence[DailyFeedbackEvidence]) -> None:
    """일일 피드백 근거 목록을 표로 표시한다."""
    frame = _models_to_frame(evidences)
    if frame.empty:
        st.info("표시할 피드백 근거가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_action_table(actions: Sequence[DailyFeedbackAction]) -> None:
    """일일 피드백 행동 항목 목록을 표로 표시한다."""
    frame = _models_to_frame(actions)
    if frame.empty:
        st.info("표시할 행동 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_contexts(contexts: Sequence[RetrievedAdviceContext]) -> None:
    """RAG에서 수집한 문서 청크를 쿼리와 출처별 expander로 표시한다."""
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


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("일일 분석, 해석, RAG 조회, 최종 소비 피드백을 한 번에 실행합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Raw data: `{settings.raw_data_dir}`")


st.title("📣 일일 피드백")
st.caption("generate_daily_feedback 서비스를 실행해 최종 일일 소비 잔소리 결과를 점검합니다.")

controls = st.columns(3)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
analysis_day_input = controls[1].date_input("분석 기준일", value=date(2024, 3, 31))
analysis_day = cast(date, analysis_day_input)
previous_day_input = controls[2].date_input(
    "전일 비교 기준일",
    value=analysis_day - timedelta(days=1),
)
previous_day = cast(date, previous_day_input)

retrieval_controls = st.columns(4)
chunk_size = retrieval_controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = retrieval_controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = retrieval_controls[2].number_input("Top K", min_value=1, value=3, step=1)
max_queries = retrieval_controls[3].number_input("Max queries", min_value=1, value=4, step=1)

if st.button("일일 피드백 생성", width="stretch"):
    with st.spinner("일일 피드백 생성 중..."):
        result = generate_daily_feedback(
            member_id=int(member_id),
            analysis_date=analysis_day,
            previous_date=previous_day,
            settings=settings,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            top_k=int(top_k),
            max_queries=int(max_queries),
        )

    if result.error:
        st.error(f"일일 피드백 생성 실패: {result.error}")
        if result.retrieval_queries:
            st.subheader("생성된 RAG 검색 질의")
            st.write(result.retrieval_queries)
        if result.daily_analysis is not None:
            with st.expander("일일 분석 JSON"):
                st.json(result.daily_analysis.model_dump())
        if result.interpretation_result:
            with st.expander("소비 해석 JSON"):
                st.json(result.interpretation_result)
        st.stop()

    if result.feedback is None:
        st.warning("피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback

    # --- DB 저장 로직 시작 ---
    try:
        import json
        from catcher_llm.db.session import session_scope
        from catcher_llm.db.models import SessionModel, UserMemoryModel
        from catcher_llm.llm.models import get_chat_model
        from langchain_core.messages import HumanMessage

        with session_scope(settings) as db_session:
            # 1. Session 데이터 생성 (날짜 중복 방지 - 기존에 있으면 Update)
            existing_session = db_session.query(SessionModel).filter_by(
                user_id=int(member_id), analysis_date=str(analysis_day)
            ).first()

            if existing_session:
                existing_session.daily_analysis_result = result.daily_analysis.model_dump_json() if result.daily_analysis else None
                existing_session.feedback_reason = json.dumps([e.model_dump() for e in feedback.key_evidences], ensure_ascii=False)
                existing_session.todo_tomorrow = feedback.tomorrow_mission
            else:
                new_session = SessionModel(
                    user_id=int(member_id),
                    analysis_date=str(analysis_day),
                    daily_analysis_result=result.daily_analysis.model_dump_json() if result.daily_analysis else None,
                    feedback_reason=json.dumps([e.model_dump() for e in feedback.key_evidences], ensure_ascii=False),
                    todo_tomorrow=feedback.tomorrow_mission
                )
                db_session.add(new_session)
                
            db_session.flush() # 쿼리를 실행해 오류 체크

            # 2. 유저의 모든 Session 데이터 가져오기 (시간순 정렬)
            all_sessions = db_session.query(SessionModel).filter_by(
                user_id=int(member_id)
            ).order_by(SessionModel.analysis_date).all()
            
            # 세션 기록을 하나의 텍스트로 합치기
            session_history_text = ""
            for s in all_sessions:
                session_history_text += f"\n[날짜: {s.analysis_date}]\n"
                session_history_text += f"소비 분석: {s.daily_analysis_result}\n"
                session_history_text += f"잔소리 근거: {s.feedback_reason}\n"
                session_history_text += f"내일 할 일: {s.todo_tomorrow}\n"

            # 3. LLM 전체 요약 생성 (모든 누적 세션 데이터 기반)
            chat_model = get_chat_model(settings)
            summary_prompt = f"""
            당신은 사용자의 소비 습관을 분석하고 기억하는 AI입니다.
            지금까지 누적된 사용자의 모든 일일 소비 분석 및 피드백 기록이 아래에 주어집니다.
            이 기록들을 바탕으로, 사용자의 전반적인 소비 패턴의 변화 흐름, 눈에 띄는 문제점, 그리고 앞으로의 조언을 
            핵심만 3~4문장으로 요약해 주세요. (이 요약은 사용자의 장기 메모리로 사용됩니다.)
            
            [사용자의 누적 소비 및 피드백 기록]
            {session_history_text}
            """
            
            summary_response = chat_model.invoke([HumanMessage(content=summary_prompt)])
            summary_text = summary_response.content if isinstance(summary_response.content, str) else str(summary_response.content)
            
            # 4. UserMemory 업데이트 (항상 1개의 행 유지)
            period_type = "daily"
            existing_memory = db_session.query(UserMemoryModel).filter_by(
                user_id=int(member_id), period_type=period_type
            ).first()

            if existing_memory:
                existing_memory.summary = summary_text
            else:
                new_memory = UserMemoryModel(
                    user_id=int(member_id),
                    period_type=period_type,
                    summary=summary_text
                )
                db_session.add(new_memory)
                
        st.success("✅ 세션 저장 및 과거-현재가 통합된 요약(User Memory)이 성공적으로 저장되었습니다!")

        st.info(f"💡 AI 요약 내용: {summary_text}")
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")
    # --- DB 저장 로직 끝 ---

    st.subheader(feedback.summary_title)
    st.write(feedback.scolding_message)
    st.info(feedback.tomorrow_mission)

    st.subheader("피드백 근거")
    _render_evidence_table(feedback.key_evidences)

    st.subheader("오늘 할 일")
    _render_action_table(feedback.action_items)

    st.subheader("RAG 검색 질의")
    st.write(result.retrieval_queries)

    st.subheader("검색된 문서 근거")
    _render_contexts(result.retrieved_contexts)

    with st.expander("일일 분석 JSON"):
        if result.daily_analysis is not None:
            st.json(result.daily_analysis.model_dump())

    with st.expander("소비 해석 JSON"):
        st.json(result.interpretation_result or {})

    with st.expander("최종 피드백 JSON"):
        st.json(feedback.model_dump())
