from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import pandas as pd

from catcher_llm.analysis.user_daily_analysis import build_daily_consumption_analysis_from_frames
from catcher_llm.config.settings import Settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)


def _write_feedback_seed_csvs(csv_dir: Path) -> None:
    """일일 소비 피드백 서비스 테스트에 사용할 SQLite 시드 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "members_v1.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형",
            ]
        ),
        encoding="utf-8-sig",
    )
    (csv_dir / "consumption_v1.csv").write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인)",
                "1,1,1000,2024-03-30 09:00:00,커피,Y,N,0,-,승인,N,식비,오프라인",
                "1,2,2000,2024-03-31 18:00:00,버스,Y,N,0,-,승인,N,교통,오프라인",
                "1,3,3000,2024-04-01 10:00:00,마트,Y,N,0,-,승인,N,생활,오프라인",
                "1,4,500,2024-04-01 20:00:00,간식,Y,N,0,-,승인,N,식비,오프라인",
            ]
        ),
        encoding="utf-8-sig",
    )


def _make_feedback_settings(root: Path) -> Settings:
    """일일 소비 피드백 서비스가 격리된 SQLite DB를 쓰도록 테스트 설정을 만든다."""
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    _write_feedback_seed_csvs(raw_dir / "csv")
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
    )


class ConsumptionFeedbackDailyAnalysisTests(unittest.TestCase):
    def test_build_daily_consumption_analysis_from_frames_matches_notebook_contract(
        self,
    ) -> None:
        """pandas 기반 분석 함수가 노트북 검증 셀의 핵심 JSON 계약을 재현한다."""
        past_frame = pd.read_csv("data/raw/csv/consumption_v1.csv", encoding="utf-8-sig")
        today_frame = pd.read_csv(
            "notebook/team02/data_pre/data_input_month.csv",
            encoding="utf-8-sig",
        )

        result = build_daily_consumption_analysis_from_frames(
            past_frame,
            today_frame,
            member_id=1,
            analysis_date="2024-04-01",
            previous_date="2024-03-31",
        )
        stable_metrics = cast(JsonObject, result["stable_metrics"])
        anomaly_detection = cast(JsonObject, result["anomaly_detection"])
        previous_day_comparison = cast(JsonObject, result["previous_day_comparison"])
        time_slot_analysis = cast(JsonObject, result["time_slot_analysis"])
        high_spending_items = cast(list[JsonObject], anomaly_detection["high_spending_items"])

        self.assertEqual(result["member_id"], 1)
        self.assertEqual(result["analysis_date"], "2024-04-01")
        self.assertEqual(stable_metrics["today_total"], 133044)
        self.assertEqual(high_spending_items[0]["description"], "SKT통신비")
        self.assertEqual(previous_day_comparison["yesterday_date"], "2024-03-31")
        self.assertEqual(time_slot_analysis["peak_slot"], "2.오전(06-11)")
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_build_daily_consumption_analysis_json_reads_sqlite_transactions(self) -> None:
        """서비스 함수가 SQLite 거래 테이블을 읽어 pandas 분석 함수 결과를 반환하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))

            result = build_daily_consumption_analysis_json(
                member_id=1,
                analysis_date="2024-04-01",
                previous_date="2024-03-31",
                settings=settings,
            )

        stable_metrics = cast(JsonObject, result["stable_metrics"])
        previous_day_comparison = cast(JsonObject, result["previous_day_comparison"])
        source_paths = cast(JsonObject, result["source_paths"])

        self.assertEqual(result["member_id"], 1)
        self.assertEqual(source_paths["past_source"], str(settings.sqlite_db_path))
        self.assertEqual(source_paths["today_source"], str(settings.sqlite_db_path))
        self.assertEqual(stable_metrics["today_total"], 3500)
        self.assertEqual(previous_day_comparison["yesterday_total"], 2000)

    def test_build_daily_consumption_analysis_json_validates_required_columns(self) -> None:
        """필수 CSV 컬럼이 없으면 분석 함수가 입력 오류를 명확한 예외로 알린다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            past_path = root / "past.csv"
            today_path = root / "today.csv"
            past_path.write_text("멤버 id,id,사용 금액\n1,1,1000\n", encoding="utf-8-sig")
            today_path.write_text("멤버 id,id,사용 금액\n1,2,2000\n", encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "필요한 컬럼"):
                build_daily_consumption_analysis_from_frames(
                    pd.read_csv(past_path, encoding="utf-8-sig"),
                    pd.read_csv(today_path, encoding="utf-8-sig"),
                )


if __name__ == "__main__":
    unittest.main()
