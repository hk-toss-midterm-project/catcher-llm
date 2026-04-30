from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing import Literal, cast

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.runnables import Runnable

from catcher_llm.chains import consumption_feedback as consumption_chains
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.consumption_feedback import UserProfileContext, UserSpendingData
from catcher_llm.services.consumption_feedback import daily_analysis, interpretation
from catcher_llm.services.consumption_feedback.daily_feedback import load_user_profile_context
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

type InterpretationMode = Literal["split", "balanced", "unified"]

_MODE_LABELS: dict[InterpretationMode, str] = {
    "split": "기존 방식",
    "balanced": "균형 방식",
    "unified": "통합 방식",
}
_MODE_DETAILS: dict[InterpretationMode, str] = {
    "split": "패턴/문제 병렬 + 원인 + 행동으로 나누는 기존 4회 호출 방식",
    "balanced": "패턴/문제는 분리하고 원인/행동을 한 번에 만드는 3회 호출 방식",
    "unified": "패턴/문제/원인/행동을 한 번에 만드는 1회 호출 방식",
}
_COMPARE_RESULTS_KEY = "daily_interpretation_compare_results"
_COMPARE_INPUT_KEY = "daily_interpretation_compare_input"


@dataclass(frozen=True)
class InterpretationCompareResult:
    """해석 방식 비교 화면에 표시할 단일 모드 실행 결과를 담는다."""

    mode: InterpretationMode
    label: str
    detail: str
    elapsed_seconds: float
    result: dict[str, object] | None = None
    error: str | None = None


def _format_seconds(seconds: float) -> str:
    """초 단위 실행 시간을 화면 표시 문자열로 변환한다."""
    return f"{seconds:.3f}s"


def _get_mode_chain_builder(
    mode: InterpretationMode,
) -> Callable[[Settings, float], Runnable[dict[str, str], dict[str, object]]]:
    """해석 모드에 맞는 체인 빌더 함수를 반환한다."""
    if mode == "split":
        return lambda settings, temperature: consumption_chains.build_spending_analysis_chain(
            settings=settings,
            temperature=temperature,
        )
    if mode == "balanced":
        return lambda settings, temperature: (
            consumption_chains.build_balanced_spending_analysis_chain(
                settings=settings,
                temperature=temperature,
            )
        )
    return lambda settings, temperature: consumption_chains.build_unified_spending_analysis_chain(
        settings=settings,
        temperature=temperature,
    )


def _run_mode(
    *,
    mode: InterpretationMode,
    analysis_input: dict[str, str],
    settings: Settings,
    temperature: float,
) -> InterpretationCompareResult:
    """단일 해석 모드를 실행하고 결과와 소요 시간을 함께 반환한다."""
    start = perf_counter()
    try:
        chain = _get_mode_chain_builder(mode)(settings, temperature)
        result = chain.invoke(analysis_input)
    except Exception as exc:
        return InterpretationCompareResult(
            mode=mode,
            label=_MODE_LABELS[mode],
            detail=_MODE_DETAILS[mode],
            elapsed_seconds=perf_counter() - start,
            error=str(exc),
        )

    return InterpretationCompareResult(
        mode=mode,
        label=_MODE_LABELS[mode],
        detail=_MODE_DETAILS[mode],
        elapsed_seconds=perf_counter() - start,
        result=cast(dict[str, object], result),
    )


def _build_analysis_input(
    *,
    member_id: int,
    analysis_day,
    previous_day,
    settings: Settings,
) -> tuple[UserSpendingData, UserProfileContext, dict[str, str]]:
    """일일 분석 JSON과 사용자 프로필을 읽어 세 해석 모드가 공유할 입력을 만든다."""
    raw_analysis = daily_analysis.build_daily_consumption_analysis_json(
        member_id=member_id,
        analysis_date=analysis_day,
        previous_date=previous_day,
        settings=settings,
    )
    user_data = interpretation.parse_user_spending_data(raw_analysis)
    user_profile = load_user_profile_context(member_id=member_id, settings=settings)
    analysis_input = interpretation.make_spending_analysis_input(
        user_data,
        user_profile=user_profile,
    )
    return user_data, user_profile, analysis_input


def _results_to_frame(results: Sequence[InterpretationCompareResult]) -> pd.DataFrame:
    """해석 방식 비교 결과를 표와 차트에 사용할 DataFrame으로 변환한다."""
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "방식": result.label,
                "모드": result.mode,
                "상태": "실패" if result.error else "성공",
                "소요 시간(초)": round(result.elapsed_seconds, 3),
                "설명": result.detail,
                "오류": result.error or "",
            }
        )
    return pd.DataFrame(rows)


def _render_result_chart(frame: pd.DataFrame) -> None:
    """해석 방식별 소요 시간을 막대 차트로 표시한다."""
    if frame.empty:
        return
    fig = px.bar(
        frame,
        x="방식",
        y="소요 시간(초)",
        color="상태",
        text="소요 시간(초)",
        color_discrete_map={"성공": "#2563eb", "실패": "#dc2626"},
    )
    fig.update_traces(texttemplate="%{text:.3f}s", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(t=20, b=20, l=20, r=20),
        yaxis_title="소요 시간(초)",
        xaxis_title=None,
        legend_title_text=None,
    )
    st.plotly_chart(fig, width="stretch")


def _render_mode_json(results: Sequence[InterpretationCompareResult]) -> None:
    """모드별 해석 결과 JSON을 expander로 표시한다."""
    for result in results:
        with st.expander(f"{result.label} 결과 JSON"):
            if result.error:
                st.error(result.error)
            else:
                st.json(result.result or {})


def _render_analysis_context(
    *,
    user_data: UserSpendingData | None,
    user_profile: UserProfileContext | None,
    analysis_input: dict[str, str] | None,
) -> None:
    """세 모드가 공유한 일일 분석 입력과 사용자 프로필을 표시한다."""
    with st.expander("공유 입력 JSON"):
        st.json(
            {
                "daily_analysis": user_data.model_dump() if user_data is not None else {},
                "user_profile": user_profile.model_dump() if user_profile is not None else {},
                "analysis_input_keys": list(analysis_input or {}),
            }
        )


settings = get_settings()

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("일일 소비 해석 체인의 기존/균형/통합 방식을 같은 입력으로 비교합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")
    st.write(f"Chat model: `{settings.chat_provider} / {settings.chat_model_name}`")

st.title("🧪 일일 해석 방식 비교")
st.caption("같은 일일 분석 입력으로 기존 방식, 균형 방식, 통합 방식의 결과물과 시간을 비교합니다.")

render_date_picker_styles()
control_columns = st.columns(3)
member_id = control_columns[0].number_input("Member ID", min_value=1, value=1, step=1)
with control_columns[1]:
    analysis_day = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_interpretation_compare_day",
    )
with control_columns[2]:
    previous_day = select_daily_date(
        "전일 기준일",
        default=DEFAULT_CALENDAR_DATE - timedelta(days=1),
        key="daily_interpretation_compare_previous_day",
    )

temperature = st.slider("해석 체인 temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
selected_labels = st.multiselect(
    "비교할 방식",
    options=list(_MODE_LABELS.values()),
    default=list(_MODE_LABELS.values()),
)
selected_modes = [mode for mode, label in _MODE_LABELS.items() if label in selected_labels]

if st.button("해석 방식 비교 실행", width="stretch"):
    with st.spinner("같은 입력으로 해석 체인들을 실행 중입니다..."):
        user_data, user_profile, analysis_input = _build_analysis_input(
            member_id=int(member_id),
            analysis_day=analysis_day,
            previous_day=previous_day,
            settings=settings,
        )
        results = [
            _run_mode(
                mode=mode,
                analysis_input=analysis_input,
                settings=settings,
                temperature=float(temperature),
            )
            for mode in selected_modes
        ]

    st.session_state[_COMPARE_INPUT_KEY] = {
        "user_data": user_data,
        "user_profile": user_profile,
        "analysis_input": analysis_input,
    }
    st.session_state[_COMPARE_RESULTS_KEY] = results

stored_input = st.session_state.get(_COMPARE_INPUT_KEY, {})
stored_results = cast(
    Sequence[InterpretationCompareResult],
    st.session_state.get(_COMPARE_RESULTS_KEY, []),
)

if not stored_results:
    st.info("Member ID와 날짜를 선택한 뒤 해석 방식 비교 실행을 눌러주세요.")
    st.stop()

st.subheader("모드별 실행 시간")
frame = _results_to_frame(stored_results)
st.dataframe(frame, width="stretch", hide_index=True)
_render_result_chart(frame)
_render_mode_json(stored_results)
_render_analysis_context(
    user_data=cast(UserSpendingData | None, stored_input.get("user_data")),
    user_profile=cast(UserProfileContext | None, stored_input.get("user_profile")),
    analysis_input=cast(dict[str, str] | None, stored_input.get("analysis_input")),
)
