from __future__ import annotations

import os
import re
import time
from datetime import timedelta

import pandas as pd
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from catcher_llm.chains.consumption_feedback import build_daily_feedback_chain
from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackResult
from catcher_llm.services.consumption_feedback.daily_feedback import (
    generate_daily_feedback,
    make_daily_feedback_input,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

FORBIDDEN_RE = re.compile(r"급증하여|평소\s*\d+[\d.]*%와|diff_point|avg_ratio")
_ACTION_PATS = [
    r"줄여", r"줄이", r"아껴", r"절약", r"목표", r"계획", r"조언",
    r"추천", r"해보", r"미션", r"포인트", r"도전", r"실천", r"관리", r"방법",
]

def score_accuracy(text: str) -> float:
    return 1.0 if re.search(r"\d+(\.\d+)?%", text) else 0.0

def score_utility(text: str) -> float:
    hits = sum(1 for p in _ACTION_PATS if re.search(p, text))
    return round(min(hits / 3.0, 1.0), 3)

def score_format(text: str) -> float:
    no_forb = 0.0 if FORBIDDEN_RE.search(text) else 1.0
    length_score = min(len(text) / 350.0, 1.0)
    return round(0.6 * no_forb + 0.4 * length_score, 3)

def _get_temperature(model: str) -> float:
    return 1.0 if "gpt-5" in model else 0.0

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-5-nano",
    "gpt-5-mini",
]

settings = get_settings()

with st.sidebar:
    st.title("🤖 모델 비교 (토큰 효율)")
    st.caption("여러 GPT 모델의 피드백 생성 품질과 토큰 사용량을 비교합니다.")

st.title("🤖 일일 피드백 모델 비교")
st.caption("동일한 분석/해석/RAG 결과를 바탕으로 여러 LLM의 최종 피드백 품질과 토큰 효율을 벤치마킹합니다.")

render_date_picker_styles()
control_columns = st.columns(3)
member_id = control_columns[0].number_input("Member ID", min_value=1, value=1, step=1)
with control_columns[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="model_comp_day",
    )
with control_columns[2]:
    previous_day = select_daily_date(
        "전일 기준일",
        default=DEFAULT_CALENDAR_DATE - timedelta(days=1),
        key="model_comp_prev_day",
    )

selected_models = st.multiselect(
    "비교할 모델 선택",
    options=DEFAULT_MODELS + ["gpt-4o", "gpt-4-turbo"],
    default=DEFAULT_MODELS,
)

if st.button("성능 비교 실행", width="stretch", type="primary"):
    if not selected_models:
        st.warning("비교할 모델을 최소 1개 이상 선택해주세요.")
        st.stop()

    with st.spinner("공통 입력 생성 중 (분석 → 해석 → RAG)..."):
        baseline_result = generate_daily_feedback(
            member_id=int(member_id),
            analysis_date=analysis_day,
            previous_date=previous_day,
            settings=settings,
        )
        if baseline_result.error:
            st.error(f"공통 입력 생성 실패: {baseline_result.error}")
            st.stop()

        feedback_input = make_daily_feedback_input(
            user_data=baseline_result.daily_analysis,
            interpretation_result=baseline_result.interpretation_result,
            advice_contexts=baseline_result.retrieved_contexts,
            user_profile=baseline_result.user_profile,
            memory_context=baseline_result.memory_context,
        )

    st.success("공통 입력 생성 완료. 각 모델별 피드백 체인 실행을 시작합니다.")

    token_records = []
    progress_bar = st.progress(0)

    for i, model in enumerate(selected_models):
        with st.spinner(f"[{model}] 실행 중..."):
            llm = ChatOpenAI(
                api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
                model=model,
                temperature=_get_temperature(model),
            )
            chain = build_daily_feedback_chain(llm=llm)

            t0 = time.perf_counter()
            scolding_message = ""
            tomorrow_mission = ""
            summary_title = ""

            try:
                with get_openai_callback() as cb:
                    result = chain.invoke(feedback_input)

                latency = round(time.perf_counter() - t0, 3)
                if isinstance(result, DailyFeedbackResult):
                    scolding_message = result.scolding_message
                    tomorrow_mission = result.tomorrow_mission
                    summary_title = result.summary_title
                elif isinstance(result, dict):
                    scolding_message = result.get("scolding_message", "")
                    tomorrow_mission = result.get("tomorrow_mission", "")
                    summary_title = result.get("summary_title", "")

                p_tok, c_tok, t_tok = cb.prompt_tokens, cb.completion_tokens, cb.total_tokens
                status = "성공"
            except Exception as e:
                latency = round(time.perf_counter() - t0, 3)
                p_tok = c_tok = t_tok = 0
                status = f"오류: {str(e)[:40]}"

            spd = round(c_tok / latency, 1) if latency > 0 and c_tok else 0.0

            acc = score_accuracy(scolding_message)
            util = score_utility(scolding_message)
            fmt = score_format(scolding_message)
            quality = round(acc * 0.4 + util * 0.3 + fmt * 0.3, 3)
            q_per_1k = round(quality / (t_tok / 1000), 4) if t_tok > 0 else 0.0

            token_records.append({
                "모델": model,
                "입력토큰": p_tok,
                "출력토큰": c_tok,
                "전체토큰": t_tok,
                "지연(초)": latency,
                "속도(tok/s)": spd,
                "정확성(0.4)": acc,
                "구용성(0.3)": util,
                "형식(0.3)": fmt,
                "종합품질": quality,
                "품질/1K토큰": q_per_1k,
                "상태": status,
                "피드백": scolding_message,
                "제목": summary_title,
                "미션": tomorrow_mission,
            })

        progress_bar.progress((i + 1) / len(selected_models))

    df = pd.DataFrame(token_records)

    st.subheader("📊 성능 및 토큰 효율 비교 결과")
    display_cols = ["모델", "전체토큰", "지연(초)", "속도(tok/s)", "종합품질", "품질/1K토큰", "상태"]
    st.dataframe(df[display_cols].sort_values("품질/1K토큰", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("📝 상세 결과")
    for row in df.itertuples():
        with st.expander(f"🤖 {row.모델} (품질/1K토큰: {getattr(row, '품질/1K토큰'):.4f})"):
            st.write(f"**제목**: {row.제목}")
            st.write(f"**피드백**: {row.피드백}")
            st.write(f"**미션**: {row.미션}")
            st.caption(
                f"⏱️ {getattr(row, '지연(초)')}초 | 🪙 전체토큰: {row.전체토큰} (입력 {row.입력토큰} / 출력 {row.출력토큰}) | "
                f"⚡ {getattr(row, '속도(tok/s)')} tok/s | "
                f"✅ 정확성: {getattr(row, '정확성(0.4)')} | 구용성: {getattr(row, '구용성(0.3)')} | 형식: {getattr(row, '형식(0.3)')}"
            )
