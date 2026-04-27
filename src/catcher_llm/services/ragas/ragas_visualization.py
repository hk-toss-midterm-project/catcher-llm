from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

RAGAS_METRIC_COLUMNS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_ragas_result_frame(csv_path: str | Path) -> pd.DataFrame:
    """ragas 결과 CSV를 읽고 점수 컬럼을 수치형으로 정리한 DataFrame을 반환한다."""
    frame = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in ("user_input", *RAGAS_METRIC_COLUMNS)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"ragas 결과 CSV에 필요한 컬럼이 없습니다: {missing_columns}")

    normalized_frame = frame.copy()
    for column in RAGAS_METRIC_COLUMNS:
        normalized_frame[column] = pd.to_numeric(normalized_frame[column], errors="raise")

    return normalized_frame


def build_ragas_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """ragas 점수 DataFrame에서 메트릭별 평균 점수 요약표를 만든다."""
    summary = pd.DataFrame(
        {
            "metric": list(RAGAS_METRIC_COLUMNS),
            "score": [float(frame[column].mean()) for column in RAGAS_METRIC_COLUMNS],
        }
    )
    return summary.sort_values("score", ascending=False, ignore_index=True)


def build_ragas_question_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """질문별 메트릭 점수를 긴 형식으로 변환해 시각화에 바로 쓸 수 있게 만든다."""
    return frame.melt(
        id_vars=["user_input"],
        value_vars=list(RAGAS_METRIC_COLUMNS),
        var_name="metric",
        value_name="score",
    )


def build_ragas_dashboard_html(frame: pd.DataFrame, title: str = "RAGAS Evaluation Dashboard") -> str:
    """ragas 결과 DataFrame을 요약 막대그래프와 질문별 히트맵 HTML로 렌더링한다."""
    summary = build_ragas_metric_summary(frame)
    question_scores = build_ragas_question_score_frame(frame)

    summary_figure = px.bar(
        summary,
        x="metric",
        y="score",
        title="메트릭 평균 점수",
        text_auto=".3f",
        range_y=[0, 1],
    )
    summary_figure.update_layout(yaxis_title="score", xaxis_title="metric")

    heatmap_figure = px.imshow(
        frame.loc[:, list(RAGAS_METRIC_COLUMNS)].transpose(),
        labels={"x": "question", "y": "metric", "color": "score"},
        x=frame["user_input"].tolist(),
        y=list(RAGAS_METRIC_COLUMNS),
        title="질문별 메트릭 점수 히트맵",
        text_auto=".2f",
        aspect="auto",
        zmin=0,
        zmax=1,
    )

    detail_figure = px.bar(
        question_scores,
        x="user_input",
        y="score",
        color="metric",
        barmode="group",
        title="질문별 메트릭 비교",
        range_y=[0, 1],
    )
    detail_figure.update_layout(xaxis_title="question", yaxis_title="score")

    summary_html = summary_figure.to_html(full_html=False, include_plotlyjs="cdn")
    heatmap_html = heatmap_figure.to_html(full_html=False, include_plotlyjs=False)
    detail_html = detail_figure.to_html(full_html=False, include_plotlyjs=False)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f8fafc;
      color: #0f172a;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    p {{
      margin-top: 0;
      color: #475569;
    }}
    .chart {{
      background: #ffffff;
      border-radius: 16px;
      padding: 16px;
      margin-top: 20px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>ragas 평가 결과를 메트릭 평균, 질문별 히트맵, 질문별 비교 막대그래프로 정리했습니다.</p>
  <section class="chart">{summary_html}</section>
  <section class="chart">{heatmap_html}</section>
  <section class="chart">{detail_html}</section>
</body>
</html>
"""


def save_ragas_dashboard(
    csv_path: str | Path,
    output_path: str | Path = "kca_ragas_dashboard.html",
    *,
    title: str = "RAGAS Evaluation Dashboard",
) -> Path:
    """ragas 결과 CSV를 읽어 대시보드 HTML 파일로 저장하고 경로를 반환한다."""
    output = Path(output_path)
    frame = load_ragas_result_frame(csv_path)
    html = build_ragas_dashboard_html(frame, title=title)
    output.write_text(html, encoding="utf-8")
    return output
