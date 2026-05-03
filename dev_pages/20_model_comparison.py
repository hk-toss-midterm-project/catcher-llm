"""GPT 모델 토큰 대비 성능 비교 개발 페이지.

노트북 langsmith_test_GPT.ipynb의 섹션 2·3 로직을 Streamlit UI로 구현한다.
- 섹션 2: 실제 Catcher 피드백 파이프라인 기반 모델 성능 비교 (LLM 기반 정밀 평가)
- 섹션 3: 토큰 사용량 캡처 및 품질/1K토큰 효율 분석

사용 방법:
  uv run streamlit run dev_app.py → 사이드바에서 'GPT 모델 성능 비교' 선택
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from catcher_llm.chains.consumption_feedback import build_daily_feedback_chain
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackResult
from catcher_llm.services.consumption_feedback.daily_feedback import (
    generate_daily_feedback,
    make_daily_feedback_input,
)
from catcher_llm.ui.date_picker import (
    render_date_picker_styles,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
_COMPARE_RESULTS_KEY = "model_compare_results"
_SHARED_INPUTS_KEY = "model_compare_shared_inputs"

_DEFAULT_MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-5-nano",
    "gpt-5-mini",
]

# 금지 패턴: 노트북과 동일하게 유지
_FORBIDDEN_RE = re.compile(r"급증하여|평소\s*\d+[\d.]*%와|diff_point|avg_ratio")

# 행동 유도 키워드: 노트북과 동일하게 유지
_ACTION_PATS: list[str] = [
    r"줄여",
    r"줄이",
    r"아껴",
    r"절약",
    r"목표",
    r"계획",
    r"조언",
    r"추천",
    r"해보",
    r"미션",
    r"포인트",
    r"도전",
    r"실천",
    r"관리",
    r"방법",
]


# ── 데이터 모델 ────────────────────────────────────────────────────────────────
@dataclass
class SharedInput:
    """공통 파이프라인 입력(분석→해석→RAG)과 기준선 피드백을 담는 구조체."""

    date: str
    feedback_input: dict[str, str]
    baseline_feedback: DailyFeedbackResult | None = None


@dataclass
class ModelRunResult:
    """단일 모델·날짜 조합의 최종 피드백 체인 실행 결과를 담는 구조체."""

    model: str
    analysis_date: str
    latency_sec: float
    scolding_message: str
    tomorrow_mission: str
    summary_title: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str | None = None


@dataclass
class CompareResults:
    """비교 실행 전체 결과를 담는 컨테이너."""

    shared_inputs: list[SharedInput] = field(default_factory=list)
    model_runs: list[ModelRunResult] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


# ── 채점 함수 ──────────────────────────────────────────────────────────────────
def _score_accuracy(text: str) -> float:
    """정확성 휴리스틱: 텍스트 내 수치(%) 포함 여부로 채점한다."""
    return 1.0 if re.search(r"\d+(\.\d+)?%", text) else 0.0


def _score_utility(text: str) -> float:
    """구용성 휴리스틱: 행동 유도 키워드 빈도로 채점한다."""
    hits = sum(1 for p in _ACTION_PATS if re.search(p, text))
    return round(min(hits / 3.0, 1.0), 3)


def _score_format(text: str) -> float:
    """형식 휴리스틱: 금지 패턴 부재(60%) + 적절한 길이(40%)로 채점한다."""
    no_forb = 0.0 if _FORBIDDEN_RE.search(text) else 1.0
    length_score = min(len(text) / 350.0, 1.0)
    return round(0.6 * no_forb + 0.4 * length_score, 3)


def _composite_quality(text: str) -> float:
    """정확성(40%) + 구용성(30%) + 형식(30%)으로 종합 품질 점수를 계산한다."""
    return round(
        _score_accuracy(text) * 0.4 + _score_utility(text) * 0.3 + _score_format(text) * 0.3,
        3,
    )


# ── 모델 실행 유틸 ─────────────────────────────────────────────────────────────
def _get_temperature(model: str) -> float:
    """모델명에 따라 추천 temperature를 반환한다."""
    return 1.0 if "gpt-5" in model else 0.0


def _run_feedback_for_model(
    *,
    model: str,
    feedback_input: dict[str, str],
    analysis_date: str,
    settings: Settings,
) -> ModelRunResult:
    """지정 모델로 최종 피드백 체인을 실행하고 토큰 사용량·지연을 함께 반환한다."""
    temperature = _get_temperature(model)
    llm = ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=model,
        temperature=temperature,
    )
    chain = build_daily_feedback_chain(llm=llm)
    scolding = ""
    mission = ""
    title = ""
    t0 = time.perf_counter()
    try:
        with get_openai_callback() as cb:
            result = chain.invoke(feedback_input)
        latency = round(time.perf_counter() - t0, 3)
        if isinstance(result, DailyFeedbackResult):
            scolding = result.scolding_message
            mission = result.tomorrow_mission
            title = result.summary_title
        p_tok = cb.prompt_tokens
        c_tok = cb.completion_tokens
        t_tok = cb.total_tokens
        spd = round(c_tok / latency, 1) if latency > 0 and c_tok else 0.0
        return ModelRunResult(
            model=model,
            analysis_date=analysis_date,
            latency_sec=latency,
            scolding_message=scolding,
            tomorrow_mission=mission,
            summary_title=title,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            tokens_per_sec=spd,
        )
    except Exception as exc:
        latency = round(time.perf_counter() - t0, 3)
        return ModelRunResult(
            model=model,
            analysis_date=analysis_date,
            latency_sec=latency,
            scolding_message="",
            tomorrow_mission="",
            summary_title="",
            error=str(exc)[:200],
        )


# ── 집계 DataFrame 생성 ────────────────────────────────────────────────────────
def _build_summary_frame(model_runs: Sequence[ModelRunResult]) -> pd.DataFrame:
    """모델별 평균 지표를 집계한 요약 DataFrame을 반환한다."""
    rows: list[dict[str, object]] = []
    for run in model_runs:
        if run.error:
            continue
        msg = run.scolding_message
        quality = _composite_quality(msg)
        q_per_1k = round(quality / (run.total_tokens / 1000), 4) if run.total_tokens > 0 else 0.0
        rows.append(
            {
                "모델": run.model,
                "날짜": run.analysis_date,
                "입력토큰": run.prompt_tokens,
                "출력토큰": run.completion_tokens,
                "전체토큰": run.total_tokens,
                "지연(초)": run.latency_sec,
                "속도(tok/s)": run.tokens_per_sec,
                "정확성": _score_accuracy(msg),
                "구용성": _score_utility(msg),
                "형식": _score_format(msg),
                "종합품질": quality,
                "품질/1K토큰": q_per_1k,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("모델")
        .agg(
            평균입력토큰=("입력토큰", "mean"),
            평균출력토큰=("출력토큰", "mean"),
            평균전체토큰=("전체토큰", "mean"),
            평균지연=("지연(초)", "mean"),
            평균속도=("속도(tok/s)", "mean"),
            평균정확성=("정확성", "mean"),
            평균구용성=("구용성", "mean"),
            평균형식=("형식", "mean"),
            평균종합품질=("종합품질", "mean"),
            품질_1K토큰=("품질/1K토큰", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("품질_1K토큰", ascending=False)
        .reset_index(drop=True)
    )
    return agg


# ── 렌더링 함수 ────────────────────────────────────────────────────────────────
def _render_shared_inputs_summary(shared_inputs: Sequence[SharedInput]) -> None:
    """공통 파이프라인 입력 생성 결과(기준선)를 간략히 표시한다."""
    if not shared_inputs:
        return
    st.subheader("📋 기준선 (기본 모델 결과)")
    for inp in shared_inputs:
        baseline = inp.baseline_feedback
        with st.expander(f"📅 {inp.date}", expanded=False):
            if baseline is None:
                st.warning("기준선 피드백을 불러오지 못했습니다.")
            else:
                st.markdown(f"**{baseline.summary_title}**")
                st.write(baseline.scolding_message)
                st.info(baseline.tomorrow_mission)


def _render_summary_table(agg: pd.DataFrame) -> None:
    """모델별 평균 지표 요약 표를 렌더링한다."""
    if agg.empty:
        st.info("표시할 결과가 없습니다.")
        return
    st.dataframe(agg, width="stretch", hide_index=True)


def _render_quality_per_token_chart(agg: pd.DataFrame) -> None:
    """품질/1K토큰 막대 차트를 렌더링한다."""
    if agg.empty:
        return
    fig = px.bar(
        agg,
        x="모델",
        y="품질_1K토큰",
        text="품질_1K토큰",
        color="모델",
        title="토큰 효율 (품질 / 1K토큰) — 높을수록 효율적",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
        yaxis_title="품질 / 1K토큰",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(agg: pd.DataFrame) -> None:
    """모델별 정확성·구용성·형식 레이더 차트를 렌더링한다."""
    if agg.empty:
        return
    categories = ["평균정확성", "평균구용성", "평균형식"]
    fig = go.Figure()
    for _, row in agg.iterrows():
        values = [cast(float, row[cat]) for cat in categories]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=["정확성", "구용성", "형식", "정확성"],
                fill="toself",
                name=str(row["모델"]),
                opacity=0.7,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        title="모델별 품질 레이더 차트",
        margin=dict(t=60, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_latency_token_scatter(agg: pd.DataFrame) -> None:
    """지연 시간 vs. 전체 토큰 산점도를 렌더링한다."""
    if agg.empty:
        return
    fig = px.scatter(
        agg,
        x="평균전체토큰",
        y="평균지연",
        color="모델",
        size="평균종합품질",
        text="모델",
        title="지연(초) vs. 전체 토큰 — 버블 크기: 종합 품질",
        size_max=40,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
        xaxis_title="평균 전체 토큰",
        yaxis_title="평균 지연(초)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_feedback_cards(
    model_runs: Sequence[ModelRunResult],
    shared_inputs: Sequence[SharedInput],
    selected_date: str,
) -> None:
    """선택한 날짜의 모델별 피드백 내용을 카드로 표시한다."""
    runs_for_date = [r for r in model_runs if r.analysis_date == selected_date]
    if not runs_for_date:
        st.info(f"{selected_date} 날짜의 결과가 없습니다.")
        return

    # 기준선 표시
    baseline_entry = next((inp for inp in shared_inputs if inp.date == selected_date), None)
    if baseline_entry and baseline_entry.baseline_feedback:
        baseline = baseline_entry.baseline_feedback
        with st.expander("📌 기준선 (기본 모델)", expanded=True):
            st.markdown(f"**{baseline.summary_title}**")
            st.write(baseline.scolding_message)
            st.info(baseline.tomorrow_mission)

    # 모델별 피드백 표시
    for run in runs_for_date:
        if run.error:
            with st.expander(f"❌ {run.model} — 오류"):
                st.error(run.error)
            continue
        quality = _composite_quality(run.scolding_message)
        with st.expander(
            f"🤖 {run.model}  |  {run.latency_sec:.2f}초  |  {run.total_tokens}토큰"
            f"  |  품질 {quality:.3f}",
            expanded=False,
        ):
            st.markdown(f"**{run.summary_title}**")
            st.write(run.scolding_message)
            st.info(run.tomorrow_mission)
            metric_cols = st.columns(4)
            metric_cols[0].metric("지연", f"{run.latency_sec:.2f}초")
            metric_cols[1].metric("전체 토큰", f"{run.total_tokens:,}")
            metric_cols[2].metric("속도", f"{run.tokens_per_sec:.1f} tok/s")
            metric_cols[3].metric("종합 품질", f"{quality:.3f}")


def _render_ranking(agg: pd.DataFrame) -> None:
    """토큰 효율 및 절대 품질 랭킹을 표시한다."""
    if agg.empty:
        return
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    best_eff = agg.iloc[0]
    best_qual = agg.loc[agg["평균종합품질"].idxmax()]
    least_tok = agg.loc[agg["평균전체토큰"].idxmin()]
    fastest = agg.loc[agg["평균속도"].idxmax()]
    col1.metric(
        "🏆 토큰 효율 1위", str(best_eff["모델"]), f"품질/1K = {best_eff['품질_1K토큰']:.4f}"
    )
    col2.metric(
        "🎯 절대 품질 1위", str(best_qual["모델"]), f"품질 = {best_qual['평균종합품질']:.3f}"
    )
    col3.metric("💡 최소 토큰", str(least_tok["모델"]), f"평균 {least_tok['평균전체토큰']:.0f}tok")
    col4.metric("⚡ 최고 속도", str(fastest["모델"]), f"{fastest['평균속도']:.1f} tok/s")


# ── 페이지 구성 ────────────────────────────────────────────────────────────────
settings = get_settings()

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("GPT 모델별 토큰 대비 성능을 실제 피드백 파이프라인으로 비교합니다.")
    st.write(f"기본 모델: `{settings.chat_provider} / {settings.chat_model_name}`")

st.title("🔬 GPT 모델 성능 비교")
st.caption(
    "실제 Catcher 소비 피드백 파이프라인으로 여러 GPT 모델의 피드백 품질과 토큰 효율을 비교합니다.\n"
    "분석→해석→RAG 단계는 기본 모델로 1회 실행(공통 입력)하고, 최종 피드백 체인만 각 모델로 실행합니다."
)

# ── 컨트롤 ────────────────────────────────────────────────────────────────────
render_date_picker_styles()
st.subheader("⚙️ 비교 설정")

model_col, date_col = st.columns([1, 2])
with model_col:
    selected_models = st.multiselect(
        "비교할 모델",
        options=_DEFAULT_MODELS,
        default=_DEFAULT_MODELS,
        key="model_compare_models",
    )

with date_col:
    date_entries_raw = st.text_area(
        "분석 날짜 목록 (줄바꿈으로 구분, YYYY-MM-DD)",
        value="2026-04-29\n2026-04-25\n2026-04-20",
        height=100,
        key="model_compare_dates",
    )

date_lines = [d.strip() for d in date_entries_raw.strip().splitlines() if d.strip()]

control_cols = st.columns(2)
member_id = control_cols[0].number_input("Member ID", min_value=1, value=1, step=1)
previous_offset = control_cols[1].number_input(
    "전일 오프셋 (분석일 - N일)", min_value=1, value=1, step=1
)

if not selected_models:
    st.warning("비교할 모델을 최소 1개 선택해주세요.")
    st.stop()

if not date_lines:
    st.warning("분석 날짜를 1개 이상 입력해주세요.")
    st.stop()

# ── 실행 ──────────────────────────────────────────────────────────────────────
if st.button("🚀 비교 실행", use_container_width=True):
    compare_results = CompareResults()

    # 1단계: 공통 파이프라인 입력 생성 (분석→해석→RAG)
    base_settings = Settings()
    shared_inputs_list: list[SharedInput] = []
    progress_bar = st.progress(0, text="공통 파이프라인 입력 생성 중...")
    total_steps = len(date_lines) + len(date_lines) * len(selected_models)
    current_step = 0

    for date_str in date_lines:
        try:
            from datetime import date as date_type

            analysis_day = date_type.fromisoformat(date_str)
            previous_day = analysis_day - timedelta(days=int(previous_offset))
            current_step += 1
            progress_bar.progress(
                current_step / total_steps,
                text=f"공통 입력 생성 중: {date_str}...",
            )
            baseline_result = generate_daily_feedback(
                member_id=int(member_id),
                analysis_date=analysis_day,
                previous_date=previous_day,
                settings=base_settings,
            )
            if baseline_result.error:
                st.warning(f"{date_str}: 파이프라인 오류 — {baseline_result.error}")
                compare_results.errors.append({"date": date_str, "error": baseline_result.error})
                continue
            if (
                baseline_result.daily_analysis is None
                or baseline_result.interpretation_result is None
            ):
                st.warning(f"{date_str}: 분석 또는 해석 결과가 없습니다.")
                continue
            feedback_input = make_daily_feedback_input(
                user_data=baseline_result.daily_analysis,
                interpretation_result=baseline_result.interpretation_result,
                advice_contexts=baseline_result.retrieved_contexts,
                user_profile=baseline_result.user_profile,
                memory_context=baseline_result.memory_context,
            )
            shared_inputs_list.append(
                SharedInput(
                    date=date_str,
                    feedback_input=feedback_input,
                    baseline_feedback=baseline_result.feedback,
                )
            )
        except Exception as exc:
            st.warning(f"{date_str}: 예외 발생 — {exc}")
            compare_results.errors.append({"date": date_str, "error": str(exc)})

    compare_results.shared_inputs = shared_inputs_list

    # 2단계: 모델별 최종 피드백 체인 실행
    model_runs: list[ModelRunResult] = []
    for shared_inp in shared_inputs_list:
        for model_name in selected_models:
            current_step += 1
            progress_bar.progress(
                min(current_step / total_steps, 1.0),
                text=f"실행 중: {model_name} @ {shared_inp.date}",
            )
            run = _run_feedback_for_model(
                model=model_name,
                feedback_input=shared_inp.feedback_input,
                analysis_date=shared_inp.date,
                settings=base_settings,
            )
            model_runs.append(run)
            if run.error:
                compare_results.errors.append(
                    {"model": model_name, "date": shared_inp.date, "error": run.error}
                )

    compare_results.model_runs = model_runs
    progress_bar.progress(1.0, text="완료!")
    st.session_state[_COMPARE_RESULTS_KEY] = compare_results
    st.session_state[_SHARED_INPUTS_KEY] = shared_inputs_list

# ── 결과 표시 ──────────────────────────────────────────────────────────────────
compare_results = cast(CompareResults | None, st.session_state.get(_COMPARE_RESULTS_KEY))

if compare_results is None:
    st.info("설정을 확인한 뒤 '비교 실행' 버튼을 눌러주세요.")
    st.stop()

if compare_results.errors:
    with st.expander(f"⚠️ 오류 {len(compare_results.errors)}건"):
        st.json(compare_results.errors)

if not compare_results.model_runs:
    st.error("실행된 결과가 없습니다. 날짜와 Member ID를 확인하고 다시 실행해주세요.")
    st.stop()

agg = _build_summary_frame(compare_results.model_runs)

# 랭킹 메트릭
_render_ranking(agg)

# 탭 구성
tab_summary, tab_charts, tab_feedback, tab_baseline = st.tabs(
    ["📊 요약 테이블", "📈 차트", "💬 피드백 내용", "📋 기준선"]
)

with tab_summary:
    st.subheader("모델별 평균 지표 (정렬: 품질/1K토큰 내림차순)")
    st.caption(
        "정확성(40%) + 구용성(30%) + 형식(30%) = 종합품질. "
        "품질/1K토큰이 높을수록 같은 비용으로 더 좋은 피드백을 생성합니다."
    )
    _render_summary_table(agg)

    # 날짜별 상세
    if compare_results.model_runs:
        st.subheader("날짜 × 모델 상세 결과")
        detail_rows: list[dict[str, object]] = []
        for run in compare_results.model_runs:
            if run.error:
                continue
            msg = run.scolding_message
            detail_rows.append(
                {
                    "모델": run.model,
                    "날짜": run.analysis_date,
                    "전체토큰": run.total_tokens,
                    "지연(초)": run.latency_sec,
                    "속도(tok/s)": run.tokens_per_sec,
                    "정확성": _score_accuracy(msg),
                    "구용성": _score_utility(msg),
                    "형식": _score_format(msg),
                    "종합품질": _composite_quality(msg),
                    "피드백 길이(자)": len(msg),
                }
            )
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

with tab_charts:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        _render_quality_per_token_chart(agg)
    with chart_col2:
        _render_radar_chart(agg)
    _render_latency_token_scatter(agg)

with tab_feedback:
    dates_available = sorted({run.analysis_date for run in compare_results.model_runs})
    selected_date = st.selectbox("날짜 선택", options=dates_available)
    if selected_date:
        _render_model_feedback_cards(
            compare_results.model_runs,
            compare_results.shared_inputs,
            selected_date,
        )

with tab_baseline:
    _render_shared_inputs_summary(compare_results.shared_inputs)
