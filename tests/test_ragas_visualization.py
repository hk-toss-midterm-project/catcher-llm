from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catcher_llm.services.ragas.ragas_visualization import (
    build_ragas_dashboard_html,
    build_ragas_metric_summary,
    load_ragas_result_frame,
    save_ragas_dashboard,
)


class RagasVisualizationTests(unittest.TestCase):
    def test_load_ragas_result_frame_converts_metric_columns_to_float(self) -> None:
        """ragas 결과 CSV의 점수 컬럼을 수치형으로 읽어들이는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "ragas.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "user_input,faithfulness,answer_relevancy,context_precision,context_recall",
                        "질문1,0.5,0.4,1.0,0.0",
                        "질문2,1.0,0.8,0.6,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            frame = load_ragas_result_frame(csv_path)

        self.assertEqual(frame["faithfulness"].dtype.kind, "f")
        self.assertAlmostEqual(float(frame.loc[0, "context_precision"]), 1.0)

    def test_build_ragas_metric_summary_returns_mean_scores(self) -> None:
        """메트릭 요약표가 평균 점수를 계산해 높은 순으로 정렬하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "ragas.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "user_input,faithfulness,answer_relevancy,context_precision,context_recall",
                        "질문1,0.5,0.4,1.0,0.0",
                        "질문2,1.0,0.8,0.6,1.0",
                    ]
                ),
                encoding="utf-8",
            )
            frame = load_ragas_result_frame(csv_path)

        summary = build_ragas_metric_summary(frame)

        self.assertEqual(summary.iloc[0]["metric"], "context_precision")
        self.assertAlmostEqual(float(summary.iloc[0]["score"]), 0.8)
        self.assertAlmostEqual(
            float(summary.loc[summary["metric"] == "faithfulness", "score"].iloc[0]),
            0.75,
        )

    def test_save_ragas_dashboard_writes_html_with_metric_titles(self) -> None:
        """ragas 대시보드 HTML이 저장되고 주요 그래프 제목을 포함하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "ragas.csv"
            output_path = Path(tmp_dir) / "dashboard.html"
            csv_path.write_text(
                "\n".join(
                    [
                        "user_input,faithfulness,answer_relevancy,context_precision,context_recall",
                        "질문1,0.5,0.4,1.0,0.0",
                        "질문2,1.0,0.8,0.6,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            saved_path = save_ragas_dashboard(csv_path, output_path)
            html = saved_path.read_text(encoding="utf-8")

        self.assertEqual(saved_path, output_path)
        self.assertIn("RAGAS Evaluation Dashboard", html)
        self.assertIn("메트릭 평균 점수", html)
        self.assertIn("질문별 메트릭 점수 히트맵", html)
        self.assertIn("질문별 메트릭 비교", html)
        self.assertIn("faithfulness", html)

    def test_save_ragas_dashboard_accepts_custom_title(self) -> None:
        """대시보드 저장 시 사용자 지정 제목이 HTML에 반영되는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "ragas.csv"
            output_path = Path(tmp_dir) / "dashboard.html"
            csv_path.write_text(
                "\n".join(
                    [
                        "user_input,faithfulness,answer_relevancy,context_precision,context_recall",
                        "질문1,0.5,0.4,1.0,0.0",
                    ]
                ),
                encoding="utf-8",
            )

            saved_path = save_ragas_dashboard(
                csv_path,
                output_path,
                title="Saving Tips RAGAS Dashboard",
            )
            html = saved_path.read_text(encoding="utf-8")

        self.assertEqual(saved_path, output_path)
        self.assertIn("Saving Tips RAGAS Dashboard", html)

    def test_build_ragas_dashboard_html_includes_question_labels(self) -> None:
        """대시보드 HTML이 질문 라벨을 포함해 결과 해석에 필요한 축 정보를 남기는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "ragas.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "user_input,faithfulness,answer_relevancy,context_precision,context_recall",
                        "구독 질문,0.5,0.4,1.0,0.0",
                        "결제 질문,1.0,0.8,0.6,1.0",
                    ]
                ),
                encoding="utf-8",
            )
            frame = load_ragas_result_frame(csv_path)

        html = build_ragas_dashboard_html(frame)

        self.assertIn("구독 질문", html)
        self.assertIn("결제 질문", html)


if __name__ == "__main__":
    unittest.main()
