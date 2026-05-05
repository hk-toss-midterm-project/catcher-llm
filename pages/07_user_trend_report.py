from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.generate_user_trend_metrics import DEFAULT_OUTPUT_DIR

st.set_page_config(
    page_title="사용자 동향 지표 미리보기",
    page_icon="📑",
    layout="wide",
)

CHART_CAPTIONS: dict[str, str] = {
    "category_monthly_amount.png": "카테고리별 월별 소비 금액",
    "monthly_total_amount.png": "월별 전체 소비 금액",
    "segment_monthly_amount.png": "세그먼트별 월별 소비 금액",
    "payment_behavior_ratios.png": "결제 행동 비율",
    "weekly_total_amount.png": "주별 전체 소비 금액",
}


def _path_label(path: Path) -> str:
    """Streamlit 화면에 표시할 상대 경로 문자열을 만든다."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _metrics_directories(metrics_root: Path) -> list[Path]:
    """그래프가 저장된 사용자 동향 지표 디렉터리를 최신순으로 조회한다."""
    if not metrics_root.exists():
        return []

    return sorted(
        [
            child
            for child in metrics_root.iterdir()
            if child.is_dir() and (child / "charts").is_dir()
        ],
        key=lambda path: path.name,
        reverse=True,
    )


def _chart_paths(metrics_dir: Path) -> list[Path]:
    """선택한 지표 디렉터리에서 표시 가능한 PNG 그래프 경로를 정렬해 반환한다."""
    charts_dir = metrics_dir / "charts"
    if not charts_dir.exists():
        return []

    paths_by_name = {path.name: path for path in charts_dir.glob("*.png")}
    ordered_paths = [paths_by_name[name] for name in CHART_CAPTIONS if name in paths_by_name]
    extra_paths = sorted(path for name, path in paths_by_name.items() if name not in CHART_CAPTIONS)
    return [*ordered_paths, *extra_paths]


def _render_csv_preview(metrics_dir: Path) -> None:
    """LLM 입력용 지표 CSV와 월별 지표 일부를 탭으로 미리 보여준다."""
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
    """생성된 PNG 차트를 화면에서 빠르게 확인하게 한다."""
    chart_paths = _chart_paths(metrics_dir)
    if not chart_paths:
        st.info("표시할 차트가 없습니다.")
        return

    columns = st.columns(2)
    for index, chart_path in enumerate(chart_paths):
        with columns[index % 2]:
            st.image(
                str(chart_path),
                caption=CHART_CAPTIONS.get(chart_path.name, chart_path.stem),
                width="stretch",
            )


st.title("사용자 동향 지표 미리보기")
st.caption("이미 생성된 사용자 동향 지표 CSV와 차트 미리보기만 조회합니다.")

metrics_root = DEFAULT_OUTPUT_DIR
metrics_dirs = _metrics_directories(metrics_root)

if not metrics_dirs:
    st.info(f"표시할 그래프가 없습니다. `{_path_label(metrics_root)}` 아래 지표를 먼저 생성하세요.")
    st.stop()

with st.sidebar:
    st.title("사용자 동향")
    selected_metrics_dir = st.selectbox(
        "지표 기간",
        options=metrics_dirs,
        index=0,
        format_func=lambda path: path.name,
    )
    st.caption(f"지표 루트: `{_path_label(metrics_root)}`")

st.subheader("생성 지표 미리보기")
st.write(f"선택한 지표 디렉터리: `{_path_label(selected_metrics_dir)}`")

_render_csv_preview(selected_metrics_dir)
_render_chart_preview(selected_metrics_dir)
