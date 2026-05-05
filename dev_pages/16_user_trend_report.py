from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from scripts.generate_user_trend_metrics import (
    DEFAULT_OUTPUT_DIR as DEFAULT_METRICS_OUTPUT_DIR,
)
from scripts.generate_user_trend_metrics import (
    DEFAULT_TRANSACTIONS_PATH,
    DEFAULT_USERS_PATH,
    TrendOutputPaths,
    build_and_save_trend_metrics,
)
from scripts.generate_user_trend_report import (
    DEFAULT_REPORT_ROOT,
    ReportGenerationResult,
    generate_latest_user_trend_report,
    generate_user_trend_rag_reports_from_metrics,
    resolve_metrics_dir,
)

DEFAULT_START_DATE = date(2026, 1, 1)
DEFAULT_END_DATE = date(2026, 4, 30)
SESSION_METRICS_DIR_KEY = "user_trend_metrics_dir"
SESSION_REPORT_PATH_KEY = "user_trend_report_path"

settings = get_settings()
configure_langsmith_env(settings)

st.set_page_config(
    page_title="사용자 동향 보고서 생성",
    page_icon="📑",
    layout="wide",
)


def _path_from_input(value: str) -> Path:
    """화면 입력 문자열을 공백이 제거된 Path 객체로 변환한다."""
    return Path(value.strip()).expanduser()


def _path_label(path: Path) -> str:
    """Streamlit 화면에 표시할 경로 문자열을 만든다."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _latest_metrics_dir(metrics_root: Path) -> Path | None:
    """보고서 생성에 사용할 수 있는 최신 지표 디렉터리를 조회한다."""
    try:
        return resolve_metrics_dir(metrics_root=metrics_root)
    except FileNotFoundError:
        return None


def _selected_metrics_dir(*, metrics_root: Path, manual_metrics_dir: str) -> Path:
    """수동 입력, 방금 생성한 지표, 최신 지표 순서로 보고서 대상 디렉터리를 결정한다."""
    if manual_metrics_dir.strip():
        return resolve_metrics_dir(
            _path_from_input(manual_metrics_dir),
            metrics_root=metrics_root,
        )

    session_value = st.session_state.get(SESSION_METRICS_DIR_KEY)
    if isinstance(session_value, str) and session_value.strip():
        return resolve_metrics_dir(
            Path(session_value),
            metrics_root=metrics_root,
        )

    return resolve_metrics_dir(metrics_root=metrics_root)


def _render_generated_paths(
    title: str,
    paths: dict[str, Path],
    *,
    max_items: int = 8,
) -> None:
    """생성된 CSV 또는 차트 경로 목록을 접힌 영역에 표시한다."""
    with st.expander(title, expanded=False):
        if not paths:
            st.info("생성된 파일이 없습니다.")
            return
        for name, path in list(paths.items())[:max_items]:
            st.write(f"- `{name}`: `{_path_label(path)}`")
        if len(paths) > max_items:
            st.caption(f"외 {len(paths) - max_items}개 파일")


def _render_csv_preview(metrics_dir: Path) -> None:
    """LLM 입력용 지표 CSV와 월별 지표 일부를 표로 미리 보여준다."""
    preview_targets = {
        "LLM long-form 지표": metrics_dir / "llm_trend_metrics.csv",
        "월별 지표": metrics_dir / "monthly_trends.csv",
    }
    tabs = st.tabs(list(preview_targets))
    for tab, (_, csv_path) in zip(tabs, preview_targets.items(), strict=True):
        with tab:
            if not csv_path.exists():
                st.info(f"`{_path_label(csv_path)}` 파일이 없습니다.")
                continue
            preview_frame = pd.read_csv(csv_path, encoding="utf-8-sig").head(20)
            st.dataframe(preview_frame, width="stretch", hide_index=True)


def _render_chart_preview(metrics_dir: Path) -> None:
    """생성된 matplotlib PNG 차트를 화면에서 빠르게 확인하게 한다."""
    chart_paths = sorted((metrics_dir / "charts").glob("*.png"))
    if not chart_paths:
        st.info("표시할 차트가 없습니다.")
        return

    columns = st.columns(2)
    for index, chart_path in enumerate(chart_paths[:4]):
        with columns[index % 2]:
            st.image(str(chart_path), caption=chart_path.name, width="stretch")


def _render_report_preview(report_path: Path) -> None:
    """생성된 Markdown 보고서의 앞부분을 화면에 렌더링한다."""
    if not report_path.exists():
        st.info("아직 생성된 보고서가 없습니다.")
        return
    report_text = report_path.read_text(encoding="utf-8")
    st.markdown(report_text[:6000])
    if len(report_text) > 6000:
        st.caption(
            "보고서가 길어 앞부분만 표시합니다. 전체 내용은 저장된 Markdown 파일에서 확인하세요."
        )


st.title("사용자 동향 지표·보고서 생성")
st.caption(
    "현재 v3 사용자·거래 CSV를 기간 기준으로 집계하고, 생성된 지표 CSV와 차트를 근거로 "
    "LangChain Markdown 보고서를 생성합니다."
)

with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("사용자 동향 지표와 보고서 생성 작업을 실행합니다.")

    users_path_text = st.text_input("사용자 CSV", value=str(DEFAULT_USERS_PATH))
    transactions_path_text = st.text_input("거래 CSV", value=str(DEFAULT_TRANSACTIONS_PATH))
    metrics_root_text = st.text_input("지표 출력 루트", value=str(DEFAULT_METRICS_OUTPUT_DIR))
    report_root_text = st.text_input("보고서 출력 루트", value=str(DEFAULT_REPORT_ROOT))
    manual_metrics_dir_text = st.text_input(
        "보고서 대상 지표 디렉터리",
        value="",
        placeholder="비워두면 방금 생성한 지표 또는 최신 지표 사용",
    )
    report_temperature = st.slider(
        "보고서 temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )
    report_mode = st.selectbox(
        "보고서 생성 방식",
        options=("월간+분기 RAG 보고서", "기간 전체 보고서"),
        index=0,
    )

users_path = _path_from_input(users_path_text)
transactions_path = _path_from_input(transactions_path_text)
metrics_root = _path_from_input(metrics_root_text)
report_root = _path_from_input(report_root_text)

controls = st.columns([1, 1, 1, 1])
with controls[0]:
    start_date = cast(
        date,
        st.date_input(
            "지표 시작일",
            value=DEFAULT_START_DATE,
            key="user_trend_start_date",
        ),
    )
with controls[1]:
    end_date = cast(
        date,
        st.date_input(
            "지표 종료일",
            value=DEFAULT_END_DATE,
            key="user_trend_end_date",
        ),
    )
with controls[2]:
    st.metric("사용자 CSV", "있음" if users_path.exists() else "없음")
with controls[3]:
    st.metric("거래 CSV", "있음" if transactions_path.exists() else "없음")

if start_date > end_date:
    st.error("지표 시작일은 종료일보다 늦을 수 없습니다.")
    st.stop()

latest_dir = _latest_metrics_dir(metrics_root)
status_columns = st.columns(3)
status_columns[0].write(f"지표 루트: `{_path_label(metrics_root)}`")
status_columns[1].write(
    f"최신 지표: `{_path_label(latest_dir)}`" if latest_dir else "최신 지표: 없음"
)
status_columns[2].write(f"보고서 루트: `{_path_label(report_root)}`")

action_columns = st.columns(2)
metrics_result: TrendOutputPaths | None = None
with action_columns[0]:
    if st.button("지표 생성", type="primary", width="stretch"):
        try:
            with st.spinner("사용자 동향 지표와 차트를 생성하는 중..."):
                metrics_result = build_and_save_trend_metrics(
                    users_path=users_path,
                    transactions_path=transactions_path,
                    output_dir=metrics_root,
                    start_date=start_date,
                    end_date=end_date,
                )
            st.session_state[SESSION_METRICS_DIR_KEY] = str(metrics_result.output_dir)
            st.success(f"지표 생성 완료: `{_path_label(metrics_result.output_dir)}`")
            _render_generated_paths("생성된 CSV", metrics_result.csv_paths)
            _render_generated_paths("생성된 차트", metrics_result.chart_paths)
        except Exception as error:
            st.error(f"지표 생성 실패: {error}")

with action_columns[1]:
    if st.button("보고서 생성", width="stretch"):
        try:
            target_metrics_dir = _selected_metrics_dir(
                metrics_root=metrics_root,
                manual_metrics_dir=manual_metrics_dir_text,
            )
            with st.spinner("LangChain으로 Markdown 보고서를 생성하는 중..."):
                if report_mode == "월간+분기 RAG 보고서":
                    batch_result = generate_user_trend_rag_reports_from_metrics(
                        metrics_dir=target_metrics_dir,
                        output_root=report_root,
                        settings=settings,
                        temperature=float(report_temperature),
                    )
                    report_paths = {
                        (
                            report.report_scope.period
                            if report.report_scope is not None
                            else report.output_path.stem
                        ): report.output_path
                        for report in batch_result.reports
                    }
                    st.session_state[SESSION_METRICS_DIR_KEY] = str(batch_result.metrics_dir)
                    if batch_result.reports:
                        st.session_state[SESSION_REPORT_PATH_KEY] = str(
                            batch_result.reports[-1].output_path
                        )
                    st.success(f"RAG 보고서 {len(batch_result.reports)}개 생성 완료")
                    _render_generated_paths("생성된 RAG 보고서", report_paths, max_items=20)
                else:
                    report_result: ReportGenerationResult = generate_latest_user_trend_report(
                        metrics_dir=target_metrics_dir,
                        metrics_root=metrics_root,
                        output_root=report_root,
                        settings=settings,
                        temperature=float(report_temperature),
                    )
                    st.session_state[SESSION_METRICS_DIR_KEY] = str(report_result.metrics_dir)
                    st.session_state[SESSION_REPORT_PATH_KEY] = str(report_result.output_path)
                    st.success(f"보고서 생성 완료: `{_path_label(report_result.output_path)}`")
        except Exception as error:
            st.error(f"보고서 생성 실패: {error}")

active_metrics_dir_value = st.session_state.get(SESSION_METRICS_DIR_KEY)
active_metrics_dir = (
    Path(active_metrics_dir_value)
    if isinstance(active_metrics_dir_value, str) and active_metrics_dir_value
    else latest_dir
)

if active_metrics_dir is not None and active_metrics_dir.exists():
    st.divider()
    st.subheader("생성 지표 미리보기")
    st.write(f"대상 지표 디렉터리: `{_path_label(active_metrics_dir)}`")
    _render_csv_preview(active_metrics_dir)
    _render_chart_preview(active_metrics_dir)

report_path_value = st.session_state.get(SESSION_REPORT_PATH_KEY)
if isinstance(report_path_value, str) and report_path_value:
    report_path = Path(report_path_value)
    st.divider()
    st.subheader("보고서 미리보기")
    st.write(f"보고서 경로: `{_path_label(report_path)}`")
    _render_report_preview(report_path)
