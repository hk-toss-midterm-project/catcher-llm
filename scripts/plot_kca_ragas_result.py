from __future__ import annotations

import argparse
from pathlib import Path

from catcher_llm.services.ragas.ragas_visualization import save_ragas_dashboard


def main() -> None:
    """ragas 결과 CSV를 읽어 Plotly 대시보드 HTML 파일로 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="kca_ragas_result.csv")
    parser.add_argument("--output", default="kca_ragas_dashboard.html")
    args = parser.parse_args()

    saved_path = save_ragas_dashboard(
        csv_path=Path(args.input),
        output_path=Path(args.output),
    )
    print(f"Saved ragas dashboard to: {saved_path}")


if __name__ == "__main__":
    main()
