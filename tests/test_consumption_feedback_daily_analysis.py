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
    (csv_dir / "users_v3.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형",
            ]
        ),
        encoding="utf-8-sig",
    )
    (csv_dir / "transactions_v3.csv").write_text(
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
    def test_daily_metrics_follow_period_metric_document(self) -> None:
        """문서의 일일 소비 분석 10개 핵심 지표와 특수 지표가 계산되는지 검증한다."""
        past_frame = pd.DataFrame(
            [
                {
                    "멤버 id": 1,
                    "id": 1,
                    "사용 금액": 4000,
                    "사용 시간": "2024-03-30 10:00:00",
                    "결제 내역": "마트",
                    "업종 카테고리": "생활",
                    "결제 방식 (온/오프라인)": "오프라인",
                },
                {
                    "멤버 id": 1,
                    "id": 2,
                    "사용 금액": 6000,
                    "사용 시간": "2024-03-30 20:00:00",
                    "결제 내역": "식당",
                    "업종 카테고리": "식비",
                    "결제 방식 (온/오프라인)": "오프라인",
                },
                {
                    "멤버 id": 1,
                    "id": 3,
                    "사용 금액": 8000,
                    "사용 시간": "2024-03-31 11:00:00",
                    "결제 내역": "쇼핑몰",
                    "업종 카테고리": "쇼핑",
                    "결제 방식 (온/오프라인)": "온라인",
                },
            ]
        )
        today_frame = pd.DataFrame(
            [
                {
                    "멤버 id": 1,
                    "id": 4,
                    "사용 금액": 3000,
                    "사용 시간": "2024-04-01 10:00:00",
                    "결제 내역": "마트",
                    "업종 카테고리": "생활",
                    "결제 방식 (온/오프라인)": "오프라인",
                },
                {
                    "멤버 id": 1,
                    "id": 5,
                    "사용 금액": 7000,
                    "사용 시간": "2024-04-01 22:30:00",
                    "결제 내역": "배달의민족",
                    "업종 카테고리": "식비",
                    "결제 방식 (온/오프라인)": "배달",
                },
            ]
        )

        result = build_daily_consumption_analysis_from_frames(
            past_frame,
            today_frame,
            member_id=1,
            analysis_date="2024-04-01",
            previous_date="2024-03-31",
            daily_budget=8000,
        )

        metrics = cast(JsonObject, result["daily_metrics"])
        self.assertEqual(metrics["daily_total_amount"], 10000)
        self.assertEqual(metrics["daily_transaction_count"], 2)
        self.assertEqual(metrics["daily_average_transaction_amount"], 5000.0)
        self.assertEqual(metrics["daily_max_transaction_amount"], 7000)
        self.assertEqual(metrics["late_night_ratio_percent"], 70.0)
        self.assertEqual(metrics["daily_budget_usage_rate_percent"], 125.0)
        self.assertEqual(metrics["no_spending_day"], False)
        self.assertAlmostEqual(float(metrics["daily_anomaly_score"]), 1.1111)

    def test_build_daily_consumption_analysis_from_frames_matches_notebook_contract(
        self,
    ) -> None:
        """pandas 기반 분석 함수가 v3 거래 CSV의 영문 컬럼 입력을 분석 JSON으로 변환한다."""
        past_frame = pd.read_csv("data/raw/csv/transactions_v3.csv", encoding="utf-8-sig")

        result = build_daily_consumption_analysis_from_frames(
            past_frame,
            past_frame,
            member_id=1,
            analysis_date="2026-03-01",
            previous_date="2026-02-28",
        )
        stable_metrics = cast(JsonObject, result["stable_metrics"])
        anomaly_detection = cast(JsonObject, result["anomaly_detection"])
        previous_day_comparison = cast(JsonObject, result["previous_day_comparison"])
        time_slot_analysis = cast(JsonObject, result["time_slot_analysis"])
        payment_behavior_analysis = cast(JsonObject, result["payment_behavior_analysis"])
        frictionless_spending = cast(
            JsonObject,
            payment_behavior_analysis["frictionless_spending"],
        )
        transaction_density = cast(
            JsonObject,
            payment_behavior_analysis["transaction_density"],
        )

        self.assertEqual(result["member_id"], 1)
        self.assertEqual(result["analysis_date"], "2026-03-01")
        self.assertEqual(stable_metrics["today_total"], 183100)
        self.assertEqual(len(cast(list[object], anomaly_detection["high_spending_items"])), 1)
        self.assertEqual(previous_day_comparison["yesterday_date"], "2026-02-28")
        self.assertEqual(previous_day_comparison["yesterday_total"], 85000)
        self.assertEqual(time_slot_analysis["peak_slot"], "3.점심/오후(11-17)")
        self.assertEqual(frictionless_spending["transaction_count"], 0)
        self.assertEqual(frictionless_spending["total_amount"], 0)
        self.assertAlmostEqual(float(frictionless_spending["ratio_percent"]), 0.0)
        self.assertEqual(transaction_density["transaction_count"], 4)
        self.assertAlmostEqual(
            float(transaction_density["average_amount_per_transaction"]),
            45775.0,
        )
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
        payment_behavior_analysis = cast(JsonObject, result["payment_behavior_analysis"])
        frictionless_spending = cast(
            JsonObject,
            payment_behavior_analysis["frictionless_spending"],
        )
        source_paths = cast(JsonObject, result["source_paths"])

        self.assertEqual(result["member_id"], 1)
        self.assertEqual(source_paths["past_source"], str(settings.sqlite_db_path))
        self.assertEqual(source_paths["today_source"], str(settings.sqlite_db_path))
        self.assertEqual(stable_metrics["today_total"], 3500)
        self.assertEqual(previous_day_comparison["yesterday_total"], 2000)
        self.assertEqual(frictionless_spending["transaction_count"], 0)
        self.assertEqual(frictionless_spending["total_amount"], 0)

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
