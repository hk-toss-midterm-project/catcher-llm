from __future__ import annotations

import argparse
from pathlib import Path

from catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval import (
    save_catcher_consumption_benchmark_ragas_result,
)
from catcher_llm.services.ragas.ragas_visualization import save_ragas_dashboard


def main() -> None:
    """Catcher 소비 벤치마크 리포트 RAG 응답을 ragas로 평가하고 CSV와 그래프 대시보드를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output", default="catcher_consumption_benchmark_ragas_result.csv")
    parser.add_argument(
        "--dashboard-output", default="catcher_consumption_benchmark_ragas_dashboard.html"
    )
    args = parser.parse_args()

    csv_output = Path(args.csv_output)
    dashboard_output = Path(args.dashboard_output)

    dataframe = save_catcher_consumption_benchmark_ragas_result(output_path=str(csv_output))
    save_ragas_dashboard(
        csv_path=csv_output,
        output_path=dashboard_output,
        title="Catcher Consumption Benchmark RAGAS Dashboard",
    )

    print(dataframe)
    print(f"Saved ragas CSV to: {csv_output}")
    print(f"Saved ragas dashboard to: {dashboard_output}")


if __name__ == "__main__":
    main()
