"""GPT 및 Claude 모델 토큰 대비 성능 비교 개발 페이지.

노트북 langsmith_test_GPT.ipynb의 섹션 2·3 로직을 Streamlit UI로 구현한다.
- 섹션 2: 실제 Catcher 피드백 파이프라인 기반 모델 성능 비교 (LangSmith 연동)
- 섹션 3: 토큰 사용량 캡처 및 품질/1K토큰 효율 분석
- 모델 비교 결과 사이드바/사이드-바이-사이드 표시

사용 방법:
  uv run streamlit run dev_app.py → 사이드바에서 'GPT 모델 성능 비교' 선택
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import timedelta
from functools import lru_cache
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from pydantic import SecretStr
from streamlit_date_picker import PickerType

from catcher_llm.chains.consumption_feedback import (
    build_daily_feedback_chain,
    build_monthly_feedback_chain,
    build_weekly_feedback_chain,
)
from catcher_llm.config.settings import Settings, configure_langsmith_env, get_settings
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackResult
from catcher_llm.schemas.consumption_feedback.monthly import MonthlyFeedbackResult
from catcher_llm.schemas.consumption_feedback.weekly import WeeklyFeedbackResult
from catcher_llm.services.consumption_feedback.daily_feedback import (
    generate_daily_feedback,
    make_daily_feedback_input,
)
from catcher_llm.services.consumption_feedback.monthly_feedback import (
    generate_monthly_feedback,
    make_monthly_feedback_input,
)
from catcher_llm.services.consumption_feedback.weekly_feedback import (
    generate_weekly_feedback,
    make_weekly_feedback_input,
)
from catcher_llm.ui.date_picker import (
    _get_state_date,
    _render_picker_popover,
    _session_key,
    render_date_picker_styles,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
_COMPARE_RESULTS_KEY = "model_compare_results"
_SHARED_INPUTS_KEY = "model_compare_shared_inputs"

_DEFAULT_MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt5.4",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
]

_EVAL_DATASET_NAME = "catcher-feedback-model-eval"
_GPT5_MAX_COMPLETION_TOKENS = 8192
_GPT5_REASONING_EFFORT = "minimal"

# ── 모델 단가 테이블 (input_$/1M, output_$/1M) ────────────────────────────────
# 출처: OpenAI / Anthropic 공식 pricing 페이지 기준
# gpt-5 계열은 아직 공식 확정 전으로 추정치 사용 — 실제 과금 후 업데이트 필요
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (10.00, 40.00),  # 추정치
    "gpt-5-mini": (1.00, 4.00),  # 추정치
    "gpt-5-nano": (0.50, 2.00),  # 추정치
    "gpt5.4": (10.00, 40.00),  # 추정치
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
}
_KRW_PER_USD: float = 1_380.0  # 참고용 환율 (변동 가능)
_LLM_JUDGE_MODEL = "claude-sonnet-4-6"
_LLM_JUDGE_RESULTS_KEY = "llm_judge_results"


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """모델명·토큰 수로 예상 비용(USD)을 계산한다.

    _MODEL_PRICING에서 가장 긴 prefix가 일치하는 항목을 선택한다.
    매칭 실패 시 0.0 반환.
    """
    m = model.lower()
    matched: tuple[float, float] | None = None
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if m.startswith(key.lower()):
            matched = _MODEL_PRICING[key]
            break
    if matched is None:
        return 0.0
    in_cost = prompt_tokens * matched[0] / 1_000_000
    out_cost = completion_tokens * matched[1] / 1_000_000
    return round(in_cost + out_cost, 7)


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

    date: str  # 일일: YYYY-MM-DD / 주간: YYYY-MM-DD (week_start) / 월간: YYYY-MM
    feedback_input: dict[str, str]
    period: str = "daily"  # "daily" | "weekly" | "monthly"
    baseline_feedback: DailyFeedbackResult | WeeklyFeedbackResult | MonthlyFeedbackResult | None = (
        None
    )
    # 렌더링용 정규화 필드
    baseline_title: str = ""
    baseline_message: str = ""
    baseline_mission: str = ""


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
    cost_usd: float = 0.0  # 예상 비용 (USD)
    llm_quality_score: float | None = None
    llm_quality_reason: str = ""
    error: str | None = None


@dataclass
class CompareResults:
    """비교 실행 전체 결과를 담는 컨테이너."""

    shared_inputs: list[SharedInput] = field(default_factory=list)
    model_runs: list[ModelRunResult] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


# ── 채점 함수 (Heuristics) ──────────────────────────────────────────────────────
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


# ── CSV 날짜 유틸 ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _load_available_dates(csv_path: str, user_id: int) -> tuple[date_type, ...]:
    """transactions CSV에서 특정 user_id의 거래 날짜 목록을 읽어 정렬된 튜플로 반환한다."""
    try:
        df = pd.read_csv(csv_path)
        df["_date"] = pd.to_datetime(df["transaction_time"]).dt.date
        mask = df["user_id"] == user_id
        dates = sorted(df.loc[mask, "_date"].unique())
        return tuple(dates)
    except Exception:
        return ()


def _get_available_dates_for_member(member_id: int, settings: Settings) -> list[date_type]:
    """settings에서 CSV 경로를 가져와 해당 유저의 거래 날짜 목록을 반환한다."""
    csv_path = str(settings.consumption_csv_path)
    return list(_load_available_dates(csv_path, member_id))


# ── 캘린더 날짜 선택 UI ────────────────────────────────────────────────────────
_DATE_PICKER_IDX_KEY = "model_compare_date_picker_idx"
_DATE_LINES_KEY = "model_compare_selected_dates"


def _render_date_picker_with_available(
    available_dates: list[date_type],
    idx: int,
    key_suffix: str,
) -> date_type | None:
    """가능한 날짜 집합에서 하나의 날짜를 캘린더 팝오버로 선택하고 반환한다."""
    if not available_dates:
        return None
    default = available_dates[-1]
    selected = _get_state_date(key_suffix, default)
    # 선택값이 유효한 날짜 안에 없으면 가장 가까운 날짜로 보정
    available_set = set(available_dates)
    if selected not in available_set:
        selected = min(available_dates, key=lambda d: abs((d - selected).days))
        st.session_state[_session_key(key_suffix)] = selected.isoformat()

    _render_picker_popover(
        label=f"날짜 {idx + 1}",
        value_label=selected.isoformat(),
        picker_type=PickerType.date,
        selected_date=selected,
        key=key_suffix,
    )
    # 선택 결과가 available_dates 안에 없으면 가장 가까운 날짜로 스냅
    final = _get_state_date(key_suffix, default)
    if final not in available_set:
        final = min(available_dates, key=lambda d: abs((d - final).days))
        st.session_state[_session_key(key_suffix)] = final.isoformat()
    return final


# ── 모델 실행 유틸 ─────────────────────────────────────────────────────────────
def _get_temperature(model: str) -> float:
    """모델명에 따라 추천 temperature를 반환한다."""
    m = model.lower()
    return 1.0 if ("gpt-5" in m or m.startswith("gpt5")) else 0.0


@traceable(name="run_feedback_for_model", tags=["model-comparison"])
def _run_feedback_for_model(
    *,
    model: str,
    feedback_input: dict[str, str],
    analysis_date: str,
    settings: Settings,
    period: str = "daily",
) -> ModelRunResult:
    """지정 모델로 최종 피드백 체인을 실행하고 토큰 사용량·지연을 함께 반환한다."""
    temperature = _get_temperature(model)

    # 모델 인스턴스 생성
    if "claude" in model.lower():
        llm = ChatAnthropic(
            api_key=SecretStr(settings.anthropic_api_key),
            model_name=model,
            temperature=temperature,
        )
    else:
        # GPT-5 등 특정 모델은 max_completion_tokens 사용이 필요할 수 있음 (OpenAI SDK/LangChain 대응 상황에 따라 다름)
        # 여기서는 ChatOpenAI의 기본 구성을 사용하되 필요한 경우 추가 인자를 전달할 수 있음
        kwargs: dict[str, Any] = {
            "api_key": SecretStr(settings.openai_api_key),
            "model": model,
            "temperature": temperature,
        }
        _ml = model.lower()
        if _ml.startswith("gpt-5") or _ml.startswith("gpt5"):
            # GPT-5 구조화 출력은 reasoning 토큰까지 completion 한도를 공유하므로 여유를 둔다.
            kwargs["max_completion_tokens"] = _GPT5_MAX_COMPLETION_TOKENS
            kwargs["reasoning_effort"] = _GPT5_REASONING_EFFORT

        llm = ChatOpenAI(**kwargs)

    if period == "weekly":
        chain = build_weekly_feedback_chain(llm=llm)
    elif period == "monthly":
        chain = build_monthly_feedback_chain(llm=llm)
    else:
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
        elif isinstance(result, WeeklyFeedbackResult):
            scolding = result.feedback_message
            mission = result.next_week_mission
            title = result.summary_title
        elif isinstance(result, MonthlyFeedbackResult):
            scolding = result.feedback_message
            mission = result.next_month_mission
            title = result.summary_title

        p_tok = cb.prompt_tokens
        c_tok = cb.completion_tokens
        t_tok = cb.total_tokens

        # Anthropic 모델의 경우 callback에서 토큰이 0으로 나올 수 있음 (기본 연동 기준)
        # 이 경우 대략적인 추정치를 사용하거나 로깅에만 의존
        spd = round(c_tok / latency, 1) if latency > 0 and c_tok else 0.0
        cost = _calc_cost(model, p_tok, c_tok)

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
            cost_usd=cost,
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
        cost = (
            run.cost_usd
            if run.cost_usd
            else _calc_cost(run.model, run.prompt_tokens, run.completion_tokens)
        )
        rows.append(
            {
                "모델": run.model,
                "날짜": run.analysis_date,
                "입력토큰": run.prompt_tokens,
                "출력토큰": run.completion_tokens,
                "전체토큰": run.total_tokens,
                "지연(초)": run.latency_sec,
                "속도(tok/s)": run.tokens_per_sec,
                "비용($)": cost,
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
            평균비용=("비용($)", "mean"),
            평균정확성=("정확성", "mean"),
            평균구용성=("구용성", "mean"),
            평균형식=("형식", "mean"),
            평균종합품질=("종합품질", "mean"),
            품질_1K토큰=("품질/1K토큰", "mean"),
        )
        .reset_index()
    )
    # 비용은 소수점 6자리, 나머지는 3자리 반올림
    cost_col = agg["평균비용"]
    agg = agg.round(3)
    agg["평균비용"] = cost_col.round(6)
    agg = agg.sort_values("품질_1K토큰", ascending=False).reset_index(drop=True)
    return agg


# ── LangSmith 연동 ────────────────────────────────────────────────────────────
def _sync_to_langsmith(shared_inputs: list[SharedInput]) -> None:
    """공통 입력을 LangSmith 데이터셋으로 업로드한다."""
    client = Client()
    datasets = list(client.list_datasets(dataset_name=_EVAL_DATASET_NAME))
    if datasets:
        dataset = datasets[0]
        st.success(f"기존 LangSmith 데이터셋 사용: `{_EVAL_DATASET_NAME}`")
    else:
        dataset = client.create_dataset(
            _EVAL_DATASET_NAME,
            description="Catcher 일일 피드백 모델 비교용 데이터셋",
        )
        for entry in shared_inputs:
            client.create_example(
                inputs={"feedback_input": entry.feedback_input, "analysis_date": entry.date},
                outputs={"analysis_date": entry.date},
                dataset_id=dataset.id,
            )
        st.success(f"LangSmith 데이터셋 생성 완료: `{_EVAL_DATASET_NAME}` ({len(shared_inputs)}건)")


# ── 렌더링 함수 ────────────────────────────────────────────────────────────────


# ── LLM 자율 평가 함수 ─────────────────────────────────────────────────────────
def _llm_judge_feedback(text: str, settings: Settings) -> tuple[float, str]:
    """claude-sonnet-4-6으로 Catcher 피드백 품질을 자율 평가한다.

    Catcher 소비 분석 프로젝트에 특화된 기준으로 채점한다.
    Returns: (score 0.0~1.0, reason str)
    """
    _sys = """당신은 Catcher 소비 분석 피드백의 품질을 평가하는 전문가입니다.
Catcher는 사용자의 실제 소비 데이터를 분석해 맞춤형 피드백과 미션을 제공하는 앱입니다.

다음 기준으로 피드백을 0.0~1.0 점수로 평가하세요:
- 구체성 (핵심): 실제 소비 수치, 금액, 카테고리를 구체적으로 언급하는가?
- 실용성: 사용자가 실제로 따를 수 있는 구체적인 절약/소비 조언을 제공하는가?
- 동기부여: 사용자에게 긍정적 동기와 행동 변화를 유도하는가?
- 자연스러움: 부자연스러운 표현이나 반복 없이 자연스러운 한국어로 작성되었는가?
- 적절한 톤: 지나치게 훈계하거나 딱딱하지 않고 친근하면서도 진지한가?

점수 기준:
- 0.9~1.0: 매우 훌륭함. 모든 기준을 만족하며 실제로 도움이 되는 피드백
- 0.7~0.9: 좋음. 대부분의 기준을 만족하지만 일부 개선 여지 있음
- 0.5~0.7: 보통. 기본적인 내용은 있으나 구체성이나 실용성이 부족
- 0.3~0.5: 미흡. 막연하거나 피드백 내용이 소비 데이터와 연결이 약함
- 0.0~0.3: 불량. 소비 분석 피드백으로서 기능을 하지 못함

JSON만 반환하세요 (다른 텍스트 절대 금지):
{"score": 0.85, "reason": "평가 이유를 2문장 이내로 간결하게"}"""

    llm = ChatAnthropic(
        api_key=SecretStr(settings.anthropic_api_key),
        model_name=_LLM_JUDGE_MODEL,
        temperature=0.0,
    )
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": _sys},
                {"role": "user", "content": f"피드백:\n{text}"},
            ]
        )
        content = str(response.content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return float(data.get("score", 0.5)), str(data.get("reason", ""))
    except Exception as exc:
        return 0.5, f"평가 오류: {str(exc)[:80]}"
    return 0.5, "파싱 실패"


def _render_shared_inputs_summary(shared_inputs: Sequence[SharedInput]) -> None:
    """공통 파이프라인 입력 생성 결과(기준선)를 간략히 표시한다."""
    if not shared_inputs:
        return
    st.subheader("📋 기준선 (기본 모델 결과)")
    for inp in shared_inputs:
        with st.expander(f"📅 {inp.date}", expanded=False):
            if not inp.baseline_message:
                st.warning("기준선 피드백을 불러오지 못했습니다.")
            else:
                st.markdown(f"**{inp.baseline_title}**")
                st.write(inp.baseline_message)
                st.info(inp.baseline_mission)


def _render_summary_table(agg: pd.DataFrame) -> None:
    """모델별 평균 지표 요약 표를 렌더링한다."""
    if agg.empty:
        st.info("표시할 결과가 없습니다.")
        return
    _disp = agg.rename(columns={"평균종합품질": "서빈님의 임의 기준 품질 평가"})
    st.dataframe(_disp, width="stretch", hide_index=True)


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
    st.plotly_chart(fig, width="stretch")


def _render_model_feedback_columns(
    model_runs: Sequence[ModelRunResult],
    shared_inputs: Sequence[SharedInput],
    selected_date: str,
    selected_models: list[str],
) -> None:
    """선택한 날짜의 모델별 피드백 내용을 가로로 나란히 표시한다."""
    runs_for_date = [r for r in model_runs if r.analysis_date == selected_date]
    if not runs_for_date:
        st.info(f"{selected_date} 날짜의 결과가 없습니다.")
        return

    # 기준선 (상단)
    baseline_entry = next((inp for inp in shared_inputs if inp.date == selected_date), None)
    if baseline_entry and baseline_entry.baseline_message:
        with st.expander("📌 기준선 (기본 모델)", expanded=False):
            st.markdown(f"**{baseline_entry.baseline_title}**")
            st.write(baseline_entry.baseline_message)
            st.caption(f"미션: {baseline_entry.baseline_mission}")

    # 모델별 컬럼 생성
    cols = st.columns(len(selected_models))
    for i, model_name in enumerate(selected_models):
        with cols[i]:
            st.markdown(f"### 🤖 {model_name}")
            run = next((r for r in runs_for_date if r.model == model_name), None)
            if run is None:
                st.warning("결과 없음")
            elif run.error:
                st.error(f"오류: {run.error}")
            else:
                quality = _composite_quality(run.scolding_message)
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <h4 style="margin-top:0;">{run.summary_title}</h4>
                        <p style="font-size: 0.9em; line-height: 1.6;">{run.scolding_message}</p>
                        <hr style="opacity: 0.2;">
                        <p style="font-size: 0.85em; color: #aaa;"><b>🎯 내일의 미션</b><br>{run.tomorrow_mission}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                cost = run.cost_usd or _calc_cost(
                    run.model, run.prompt_tokens, run.completion_tokens
                )
                krw = cost * _KRW_PER_USD
                st.metric("서빈님의 임의 기준 품질 평가", f"{quality:.3f}")
                st.caption(
                    f"⏱️ {run.latency_sec:.2f}s | 🪙 {run.total_tokens}tok | "
                    f"💵 ${cost:.5f} (≈₩{krw:.2f})"
                )


def _render_ranking(agg: pd.DataFrame) -> None:
    """토큰 효율·절대 품질·비용 랭킹을 표시한다."""
    if agg.empty:
        return
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    best_eff = agg.iloc[0]
    best_qual = agg.loc[agg["평균종합품질"].idxmax()]
    least_tok = agg.loc[agg["평균전체토큰"].idxmin()]
    fastest = agg.loc[agg["평균속도"].idxmax()]
    cheapest = agg.loc[agg["평균비용"].idxmin()]
    col1.metric(
        "🏆 토큰 효율 1위", str(best_eff["모델"]), f"품질/1K = {best_eff['품질_1K토큰']:.4f}"
    )
    col2.metric(
        "🎯 임의 기준 품질 1위",
        str(best_qual["모델"]),
        f"서빈기준 = {best_qual['평균종합품질']:.3f}",
    )
    col3.metric("💡 최소 토큰", str(least_tok["모델"]), f"평균 {least_tok['평균전체토큰']:.0f}tok")
    col4.metric("⚡ 최고 속도", str(fastest["모델"]), f"{fastest['평균속도']:.1f} tok/s")
    col5.metric(
        "💰 최저 비용",
        str(cheapest["모델"]),
        f"${cheapest['평균비용']:.5f} (≈₩{cheapest['평균비용'] * _KRW_PER_USD:.1f})",
    )


# ── 페이지 구성 ────────────────────────────────────────────────────────────────
settings = get_settings()
configure_langsmith_env(settings)

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("GPT & Claude 모델별 성능을 비교합니다.")
    st.write(f"기본 모델: `{settings.chat_provider} / {settings.chat_model_name}`")
    if settings.has_langsmith_key:
        st.success("LangSmith 연동 활성화됨")
    else:
        st.warning("LangSmith API Key 없음")

st.title("🔬 모델 성능 비교 (LangSmith)")
st.caption(
    "여러 모델의 피드백 품질과 토큰 효율을 한 페이지에서 비교합니다.\n"
    "분석→해석→RAG 단계는 기본 모델로 1회 실행(공통 입력)하고, 최종 피드백 체인만 각 모델로 실행합니다."
)

# ── 컨트롤 ────────────────────────────────────────────────────────────────────
render_date_picker_styles()
st.subheader("⚙️ 비교 설정")

# ── 분석 주기 선택 ────────────────────────────────────────────────────────────
period_label = st.radio(
    "분석 주기",
    options=["일일 📆", "주간 📅", "월간 🗓️"],
    horizontal=True,
    key="period_select",
    label_visibility="collapsed",
)
# 내부 키 정규화
_PERIOD_MAP = {"일일 📆": "daily", "주간 📅": "weekly", "월간 🗓️": "monthly"}
period = _PERIOD_MAP[period_label]

# ── Member ID 및 기본 설정 ────────────────────────────────────────────────────
config_cols = st.columns([1, 1, 2])
member_id = config_cols[0].number_input("Member ID", min_value=1, value=1, step=1)
sync_ls = config_cols[1].checkbox("LangSmith 동기화", value=True)

# ── 모델 선택 ─────────────────────────────────────────────────────────────────
selected_models = config_cols[2].multiselect(
    "비교할 모델",
    options=_DEFAULT_MODELS,
    default=_DEFAULT_MODELS,
    key="model_compare_models",
)

# ── 날짜 선택 (주기별) ────────────────────────────────────────────────────────
_base_settings_for_dates = Settings()
_available_dates = _get_available_dates_for_member(int(member_id), _base_settings_for_dates)
_default_date = _available_dates[-1] if _available_dates else date_type(2026, 4, 29)

# 주기에 따른 날짜 입력 UI
if period == "daily":
    date_col, _ = st.columns([1, 2])
    with date_col:
        picked_date = _render_date_picker_with_available(_available_dates, 0, "mc_daily_date")
        if "catcher_date_picker_mc_daily_date" not in st.session_state:
            st.session_state["catcher_date_picker_mc_daily_date"] = _default_date.isoformat()
            picked_date = _default_date
    analysis_date = picked_date if picked_date is not None else _default_date
    # 전일 자동 계산
    previous_date = analysis_date - timedelta(days=1)
    date_label = analysis_date.isoformat()
    st.caption(f"분석일: **{analysis_date}** | 비교 기준일(전일): **{previous_date}** (자동 설정)")

elif period == "weekly":
    date_col, _ = st.columns([1, 2])
    with date_col:
        picked_date = _render_date_picker_with_available(_available_dates, 0, "mc_weekly_date")
        if "catcher_date_picker_mc_weekly_date" not in st.session_state:
            st.session_state["catcher_date_picker_mc_weekly_date"] = _default_date.isoformat()
            picked_date = _default_date
    week_start = picked_date if picked_date is not None else _default_date
    week_end = week_start + timedelta(days=6)
    date_label = week_start.isoformat()
    st.caption(f"주간 범위: **{week_start}** ~ **{week_end}** (7일, 자동 설정)")

else:  # monthly
    _available_months = sorted({d.strftime("%Y-%m") for d in _available_dates})
    if not _available_months:
        _available_months = [_default_date.strftime("%Y-%m")]
    month_col, _ = st.columns([1, 2])
    with month_col:
        selected_month = st.selectbox(
            "분석 월",
            options=_available_months,
            index=len(_available_months) - 1,
            key="mc_monthly_month",
        )
    date_label = selected_month
    st.caption(f"분석 월: **{selected_month}**")

if not selected_models:
    st.warning("비교할 모델을 최소 1개 선택해주세요.")
    st.stop()


# ── 실행 ──────────────────────────────────────────────────────────────────────
if st.button("🚀 비교 실행", width="stretch"):
    compare_results = CompareResults()
    base_settings = Settings()
    shared_inputs_list: list[SharedInput] = []

    progress_bar = st.progress(0, text="공통 파이프라인 입력 생성 중...")
    total_steps = 2  # 단계 1 + 단계 2 (모델 수로 보정)

    # ── 1단계: 공통 파이프라인 입력 생성 (분석→해석→RAG) ──────────────────────
    try:
        if period == "daily":
            progress_bar.progress(0.1, text=f"일일 피드백 파이프라인 실행 중: {analysis_date}...")
            baseline_result = generate_daily_feedback(
                member_id=int(member_id),
                analysis_date=analysis_date,
                previous_date=previous_date,
                settings=base_settings,
            )
            if baseline_result.error:
                st.warning(f"파이프라인 오류: {baseline_result.error}")
                compare_results.errors.append({"date": date_label, "error": baseline_result.error})
            elif (
                baseline_result.daily_analysis is None
                or baseline_result.interpretation_result is None
            ):
                st.warning("분석 또는 해석 결과가 없습니다.")
            else:
                feedback_input = make_daily_feedback_input(
                    user_data=baseline_result.daily_analysis,
                    interpretation_result=baseline_result.interpretation_result,
                    advice_contexts=baseline_result.retrieved_contexts,
                    user_profile=baseline_result.user_profile,
                    memory_context=baseline_result.memory_context,
                )
                fb = baseline_result.feedback
                shared_inputs_list.append(
                    SharedInput(
                        date=date_label,
                        feedback_input=feedback_input,
                        period=period,
                        baseline_feedback=fb,
                        baseline_title=fb.summary_title if fb else "",
                        baseline_message=fb.scolding_message if fb else "",
                        baseline_mission=fb.tomorrow_mission if fb else "",
                    )
                )

        elif period == "weekly":
            progress_bar.progress(
                0.1, text=f"주간 피드백 파이프라인 실행 중: {week_start} ~ {week_end}..."
            )
            baseline_result = generate_weekly_feedback(
                member_id=int(member_id),
                week_start=week_start,
                week_end=week_end,
                settings=base_settings,
            )
            if baseline_result.error:
                st.warning(f"파이프라인 오류: {baseline_result.error}")
                compare_results.errors.append({"date": date_label, "error": baseline_result.error})
            elif (
                baseline_result.weekly_analysis is None
                or baseline_result.interpretation_result is None
            ):
                st.warning("주간 분석 또는 해석 결과가 없습니다.")
            else:
                feedback_input = make_weekly_feedback_input(
                    weekly_data=baseline_result.weekly_analysis,
                    interpretation_result=baseline_result.interpretation_result,
                    advice_contexts=baseline_result.retrieved_contexts,
                    user_profile=baseline_result.user_profile,
                    memory_context=baseline_result.memory_context,
                )
                fb = baseline_result.feedback
                shared_inputs_list.append(
                    SharedInput(
                        date=date_label,
                        feedback_input=feedback_input,
                        period=period,
                        baseline_feedback=fb,
                        baseline_title=fb.summary_title if fb else "",
                        baseline_message=fb.feedback_message if fb else "",
                        baseline_mission=fb.next_week_mission if fb else "",
                    )
                )

        else:  # monthly
            progress_bar.progress(0.1, text=f"월간 피드백 파이프라인 실행 중: {selected_month}...")
            baseline_result = generate_monthly_feedback(
                member_id=int(member_id),
                analysis_month=selected_month,
                settings=base_settings,
            )
            if baseline_result.error:
                st.warning(f"파이프라인 오류: {baseline_result.error}")
                compare_results.errors.append({"date": date_label, "error": baseline_result.error})
            elif (
                baseline_result.monthly_analysis is None
                or baseline_result.interpretation_result is None
            ):
                st.warning("월간 분석 또는 해석 결과가 없습니다.")
            else:
                feedback_input = make_monthly_feedback_input(
                    monthly_data=baseline_result.monthly_analysis,
                    interpretation_result=baseline_result.interpretation_result,
                    advice_contexts=baseline_result.retrieved_contexts,
                    user_profile=baseline_result.user_profile,
                    memory_context=baseline_result.memory_context,
                )
                fb = baseline_result.feedback
                shared_inputs_list.append(
                    SharedInput(
                        date=date_label,
                        feedback_input=feedback_input,
                        period=period,
                        baseline_feedback=fb,
                        baseline_title=fb.summary_title if fb else "",
                        baseline_message=fb.feedback_message if fb else "",
                        baseline_mission=fb.next_month_mission if fb else "",
                    )
                )

    except Exception as exc:
        st.warning(f"공통 파이프라인 예외: {exc}")
        compare_results.errors.append({"date": date_label, "error": str(exc)})

    compare_results.shared_inputs = shared_inputs_list

    # ── 2단계: 모델별 최종 피드백 체인 실행 ───────────────────────────────────
    model_runs: list[ModelRunResult] = []
    n_models = len(selected_models)
    for idx, shared_inp in enumerate(shared_inputs_list):
        for m_idx, model_name in enumerate(selected_models):
            step_frac = 0.2 + 0.8 * (idx * n_models + m_idx + 1) / max(n_models, 1)
            progress_bar.progress(
                min(step_frac, 0.99),
                text=f"실행 중: {model_name} @ {shared_inp.date}",
            )
            run = _run_feedback_for_model(
                model=model_name,
                feedback_input=shared_inp.feedback_input,
                analysis_date=shared_inp.date,
                settings=base_settings,
                period=period,
            )
            model_runs.append(run)
            if run.error:
                compare_results.errors.append(
                    {"model": model_name, "date": shared_inp.date, "error": run.error}
                )

    compare_results.model_runs = model_runs
    progress_bar.progress(1.0, text="완료!")

    # ── 3단계: LangSmith 동기화 ───────────────────────────────────────────────
    if sync_ls and settings.has_langsmith_key:
        with st.spinner("LangSmith 데이터셋 업로드 중..."):
            _sync_to_langsmith(shared_inputs_list)

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
tab_feedback, tab_summary, tab_charts, tab_baseline = st.tabs(
    ["💬 피드백 비교 (Side-by-Side)", "📊 요약 테이블", "📈 차트", "📋 기준선"]
)

with tab_feedback:
    st.subheader("모델별 피드백 결과 나란히 보기")
    dates_available = sorted({run.analysis_date for run in compare_results.model_runs})
    selected_date = st.selectbox("날짜 선택", options=dates_available, key="feedback_date_select")
    if selected_date:
        _render_model_feedback_columns(
            compare_results.model_runs,
            compare_results.shared_inputs,
            selected_date,
            selected_models,
        )

with tab_summary:
    st.subheader("모델별 평균 지표 (정렬: 품질/1K토큰 내림차순)")
    st.caption(
        "정확성(40%) + 구용성(30%) + 형식(30%) = 서빈님의 임의 기준 품질 평가. "
        "품질/1K토큰이 높을수록 같은 비용으로 더 좋은 피드백을 생성합니다."
    )
    _render_summary_table(agg)

    # 날짜별 상세
    if compare_results.model_runs:
        st.subheader("날짜 x 모델 상세 결과")
        detail_rows: list[dict[str, object]] = []
        for run in compare_results.model_runs:
            if run.error:
                continue
            msg = run.scolding_message
            cost = run.cost_usd or _calc_cost(run.model, run.prompt_tokens, run.completion_tokens)
            detail_rows.append(
                {
                    "모델": run.model,
                    "날짜": run.analysis_date,
                    "입력토큰": run.prompt_tokens,
                    "출력토큰": run.completion_tokens,
                    "전체토큰": run.total_tokens,
                    "지연(초)": run.latency_sec,
                    "속도(tok/s)": run.tokens_per_sec,
                    "비용($)": cost,
                    "비용(₩)": round(cost * _KRW_PER_USD, 2),
                    "정확성": _score_accuracy(msg),
                    "구용성": _score_utility(msg),
                    "형식": _score_format(msg),
                    "서빈님의 임의 기준 품질 평가": _composite_quality(msg),
                    "피드백 길이(자)": len(msg),
                }
            )
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

with tab_charts:
    # 상단: 토큰 효율 + 레이더
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        _render_quality_per_token_chart(agg)
    with chart_col2:
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
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # 하단: 비용 분석
    if "평균비용" in agg.columns:
        cost_col1, cost_col2 = st.columns(2)

        with cost_col1:
            agg_cost = agg.copy()
            agg_cost["비용(₩)"] = (agg_cost["평균비용"] * _KRW_PER_USD).round(2)
            fig_cost = px.bar(
                agg_cost,
                x="모델",
                y="평균비용",
                text=agg_cost["평균비용"].apply(lambda v: f"${v:.5f}"),
                color="모델",
                title="모델별 호출 평균 비용 (USD)",
                hover_data={"비용(₩)": True, "평균비용": True},
            )
            fig_cost.update_traces(textposition="outside")
            fig_cost.update_layout(
                height=400,
                margin=dict(t=50, b=20, l=20, r=20),
                showlegend=False,
                yaxis_title="비용 (USD)",
                xaxis_title=None,
            )
            st.plotly_chart(fig_cost, width="stretch")

        with cost_col2:
            fig_scatter = px.scatter(
                agg,
                x="평균비용",
                y="평균종합품질",
                text="모델",
                size="평균전체토큰",
                color="모델",
                title="비용 vs 품질 (버블=전체토큰)",
                labels={
                    "평균비용": "평균 비용 (USD)",
                    "평균종합품질": "서빈님의 임의 기준 품질 평가",
                },
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(
                height=400,
                margin=dict(t=50, b=20, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_scatter, width="stretch")

        with st.expander("사용된 모델 단가표 (수동 업데이트 필요)"):
            pricing_rows = [
                {
                    "모델 (prefix)": k,
                    "입력 ($/1M)": v[0],
                    "출력 ($/1M)": v[1],
                    "입력 (₩/1M)": round(v[0] * _KRW_PER_USD),
                    "출력 (₩/1M)": round(v[1] * _KRW_PER_USD),
                }
                for k, v in _MODEL_PRICING.items()
            ]
            st.dataframe(pd.DataFrame(pricing_rows), hide_index=True)
            st.caption("gpt-5 계열은 공식 출시 전 추정치입니다.")

with tab_baseline:
    _render_shared_inputs_summary(compare_results.shared_inputs)


# ── LLM 자율 평가 섹션 ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 LLM 자율 평가 (claude-sonnet-4-6)")
st.caption(
    "규칙 기반 점수(서빈님의 임의 기준)와 별개로, claude-sonnet-4-6이 "
    "Catcher 피드백 품질을 자율적으로 채점합니다. "
    "수치 인용·실용성·동기부여·자연스러움·톤을 종합 평가합니다."
)

if st.button("🤖 LLM 평가 실행", key="run_llm_judge", width="stretch"):
    _llm_res: dict[tuple[str, str], dict[str, object]] = {}
    _base_s2 = Settings()
    _runs_eval = [r for r in compare_results.model_runs if not r.error and r.scolding_message]
    if not _runs_eval:
        st.warning("평가할 결과가 없습니다. 먼저 비교 실행을 해주세요.")
    else:
        _prog2 = st.progress(0, text="LLM 평가 준비 중...")
        for _i2, _run2 in enumerate(_runs_eval):
            _pct2 = (_i2 + 1) / max(len(_runs_eval), 1)
            _prog2.progress(
                _pct2,
                text=f"LLM 평가 중: {_run2.model} @ {_run2.analysis_date} ({_i2 + 1}/{len(_runs_eval)})",
            )
            _sc, _rs = _llm_judge_feedback(_run2.scolding_message, _base_s2)
            _llm_res[(_run2.model, _run2.analysis_date)] = {"score": _sc, "reason": _rs}
        _prog2.progress(1.0, text="LLM 평가 완료!")
        st.session_state[_LLM_JUDGE_RESULTS_KEY] = _llm_res

_llm_data = cast(dict | None, st.session_state.get(_LLM_JUDGE_RESULTS_KEY))
if _llm_data:
    _jrows = [
        {
            "모델": k[0],
            "날짜": k[1],
            "LLM 평가 점수": v["score"],
            "평가 이유 (claude-sonnet-4-6)": v["reason"],
        }
        for k, v in _llm_data.items()
    ]
    _jdf = pd.DataFrame(_jrows)

    # 모델별 평균 집계 바 차트
    _agg_j = _jdf.groupby("모델")["LLM 평가 점수"].mean().reset_index()
    _agg_j = _agg_j.sort_values("LLM 평가 점수", ascending=False).reset_index(drop=True)
    _fig_j = px.bar(
        _agg_j,
        x="모델",
        y="LLM 평가 점수",
        color="모델",
        title="모델별 LLM 자율 평가 평균 점수 (claude-sonnet-4-6 기준)",
        text=_agg_j["LLM 평가 점수"].apply(lambda v: f"{v:.3f}"),
    )
    _fig_j.update_traces(textposition="outside")
    _fig_j.update_layout(
        height=420,
        showlegend=False,
        yaxis_range=[0, 1.15],
        yaxis_title="LLM 평가 점수 (0~1)",
        xaxis_title=None,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(_fig_j, width="stretch")

    # 상세 결과 테이블
    st.dataframe(_jdf, hide_index=True, width="stretch")
