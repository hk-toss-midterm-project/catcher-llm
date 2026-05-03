from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import SessionModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.prompts.consumption_feedback import build_weekly_feedback_prompt
from catcher_llm.schemas.consumption_feedback import (
    CauseAnalysisResult,
    InterventionTarget,
    RetrievedAdviceContext,
    UserProfileContext,
    WeeklyFeedbackAction,
    WeeklyFeedbackEvidence,
    WeeklyFeedbackResult,
)
from catcher_llm.services.consumption_feedback.timing import FeedbackTimingRecord
from catcher_llm.services.consumption_feedback.weekly_analysis import (
    build_weekly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.weekly_feedback import (
    build_weekly_feedback_retrieval_queries,
    generate_weekly_feedback,
    make_weekly_feedback_input,
    make_weekly_spending_analysis_input,
    parse_weekly_spending_data,
)
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.user_data_service import ensure_user_database


def _write_weekly_seed_csvs(csv_dir: Path) -> None:
    """주간 피드백 테스트에 사용할 v3 사용자와 거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v3.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,saving_goal_text,target_max_spending_amount",
                "1,김토스,29,개발자,남성,36000000,서울,Gold,절약형,비상금 300만원 만들기,210000",
            ]
        ),
        encoding="utf-8-sig",
    )
    (csv_dir / "transactions_v3.csv").write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인)",
                "1,1,10000,2024-03-25 12:00:00,구내식당,Y,N,0,-,승인,N,식비,오프라인",
                "1,2,12000,2024-03-26 12:00:00,구내식당,Y,N,0,-,승인,N,식비,오프라인",
                "1,3,30000,2024-04-01 22:00:00,배달의민족,Y,N,0,-,승인,N,식비,온라인",
                "1,4,5500,2024-04-02 09:00:00,스타벅스,Y,N,0,-,승인,N,식비,오프라인",
                "1,5,65000,2024-04-03 10:00:00,SKT통신비,Y,N,0,-,승인,N,생활,온라인",
                "1,6,7000,2024-04-04 08:30:00,스타벅스,Y,N,0,-,승인,N,식비,오프라인",
                "1,7,4500,2024-04-05 18:00:00,CU,Y,N,0,-,승인,N,식비,오프라인",
            ]
        ),
        encoding="utf-8-sig",
    )


def _make_weekly_settings(root: Path) -> Settings:
    """주간 피드백 테스트가 격리된 SQLite DB를 쓰도록 설정을 만든다."""
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    _write_weekly_seed_csvs(raw_dir / "csv")
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
        openai_api_key="test-key",
    )


class ConsumptionFeedbackWeeklyFeedbackTests(unittest.TestCase):
    def test_weekly_analysis_json_uses_user_financial_context(self) -> None:
        """주간 분석 서비스가 사용자 연봉과 목표 소비 금액으로 예산·소득 지표를 계산하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_weekly_settings(Path(tmp_dir))
            weekly_payload = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start="2024-04-01",
                week_end="2024-04-07",
                settings=settings,
            )

        weekly_metrics = cast(dict[str, object], weekly_payload["weekly_metrics"])

        self.assertAlmostEqual(
            float(weekly_metrics["weekly_budget_usage_rate_percent"]),
            228.5714,
            places=4,
        )
        self.assertEqual(weekly_metrics["weekly_remaining_budget"], 0)
        self.assertEqual(weekly_metrics["weekly_overspend_amount"], 63000)
        self.assertAlmostEqual(
            float(weekly_metrics["weekly_income_usage_rate_percent"]),
            16.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(weekly_metrics["weekly_budget_burn_rate"]),
            2.2857,
            places=4,
        )
        self.assertAlmostEqual(
            float(weekly_metrics["month_to_date_budget_usage_rate_percent"]),
            53.3333,
            places=4,
        )
        self.assertEqual(
            weekly_metrics["projected_monthly_spending_from_weekly_pace"],
            480000,
        )

    def test_weekly_repeat_merchants_use_sqlite_merchant_name(self) -> None:
        """SQLite merchant_name 컬럼이 있으면 반복 가맹점을 실제 가맹점명으로 집계하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            raw_dir = data_dir / "raw"
            csv_dir = raw_dir / "csv"
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
                        "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                        "1,1,5000,2024-04-01 09:00:00,카페 및 디저트 결제,스타벅스,False,0,False,APPROVED,False,식비,OFFLINE",
                        "2,1,6000,2024-04-02 09:00:00,카페 및 디저트 결제,스타벅스,False,0,False,APPROVED,False,식비,OFFLINE",
                        "3,1,4500,2024-04-03 09:00:00,카페 및 디저트 결제,이디야,False,0,False,APPROVED,False,식비,OFFLINE",
                        "4,1,12000,2024-04-04 12:00:00,외식 결제,홍콩반점,False,0,False,APPROVED,False,식비,OFFLINE",
                    ]
                ),
                encoding="utf-8-sig",
            )
            settings = Settings(
                data_dir=data_dir,
                raw_data_dir=raw_dir,
                processed_data_dir=data_dir / "processed",
                vectorstore_dir=data_dir / "vectordb",
                eval_data_dir=data_dir / "evals",
                sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
            )

            weekly_payload = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start="2024-04-01",
                week_end="2024-04-07",
                settings=settings,
            )

        repeat_patterns = weekly_payload["repeat_patterns"]
        self.assertIsInstance(repeat_patterns, dict)
        top_merchants = cast(dict[str, object], repeat_patterns)["top_merchants"]
        self.assertIsInstance(top_merchants, list)
        merchant_rows: dict[str, dict[str, object]] = {}
        for item in top_merchants:
            self.assertIsInstance(item, dict)
            item_dict = cast(dict[str, object], item)
            merchant = item_dict.get("merchant")
            if isinstance(merchant, str):
                merchant_rows[merchant] = item_dict

        self.assertEqual(merchant_rows["스타벅스"].get("visit_count"), 2)
        self.assertEqual(merchant_rows["이디야"].get("visit_count"), 1)
        self.assertNotIn("카페 및 디저트 결제", merchant_rows)

    def test_weekly_spending_analysis_input_uses_weekly_json_and_profile(self) -> None:
        """주간 분석 JSON을 주간 해석 체인의 raw/indicator/profile 입력으로 변환하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_weekly_settings(Path(tmp_dir))
            weekly_payload = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start="2024-04-01",
                week_end="2024-04-07",
                settings=settings,
            )

        weekly_data = parse_weekly_spending_data(weekly_payload)
        analysis_input = make_weekly_spending_analysis_input(
            weekly_data,
            user_profile={"user_id": 1, "saving_goal_text": "비상금"},
        )

        self.assertEqual(weekly_data.week_start, "2024-04-01")
        self.assertIn("weekly_summary.this_week_total", analysis_input["indicator_json"])
        self.assertIn("weekly_metrics", analysis_input["raw_json"])
        self.assertIn("weekly_comparisons", analysis_input["raw_json"])
        self.assertIn(
            "weekly_metrics.weekend_spending_ratio_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "weekly_metrics.weekly_budget_usage_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "weekly_metrics.weekly_income_usage_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "weekly_metrics.month_to_date_budget_usage_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn("최근 4주 평균 대비 지출 증감률", analysis_input["indicator_json"])
        self.assertIn("지난달 같은 주차 대비 지출 증감률", analysis_input["indicator_json"])
        self.assertIn("주말 과소비 지수", analysis_input["indicator_json"])
        self.assertIn("waste_detection.high_spending.items", analysis_input["indicator_json"])
        self.assertIn("배달의민족", analysis_input["raw_json"])
        self.assertIn("비상금", analysis_input["user_profile_json"])

    def test_weekly_feedback_prompt_requires_weekly_inputs(self) -> None:
        """최종 주간 피드백 프롬프트가 주간 분석·해석·문서 근거를 입력으로 받는지 검증한다."""
        prompt = build_weekly_feedback_prompt()

        self.assertEqual(
            set(prompt.input_variables),
            {
                "weekly_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )

        rendered = prompt.invoke(
            {
                "weekly_json": '{"weekly_summary": {"this_week_total": 112000}}',
                "interpretation_json": '{"cause_result": {"intervention_targets": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "배달비 절약"}]',
                "user_profile_json": '{"saving_goal_text": "비상금"}',
                "memory_context_json": "{}",
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("주간 소비", content)
        self.assertIn("배달비 절약", content)
        self.assertIn("비상금", content)
        self.assertIn("JSON 수치 근거", content)
        self.assertIn("cause_result.intervention_targets는 RAG 검색용 중간 후보", content)
        self.assertIn("action_result가 포함된 경우에도 최종 행동이 아니므로", content)
        self.assertIn("RAG 문서 근거와 사용자 메모리", content)
        self.assertIn("그대로 복사하지 마라", content)
        self.assertIn("feedback_message", content)
        self.assertNotIn("scolding_message", content)

    def test_make_weekly_feedback_input_serializes_contexts(self) -> None:
        """주간 피드백 체인 입력이 주간 분석, 해석, RAG, 프로필을 JSON 문자열로 직렬화하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_weekly_settings(Path(tmp_dir))
            weekly_payload = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start=date(2024, 4, 1),
                week_end=date(2024, 4, 7),
                settings=settings,
            )
        weekly_data = parse_weekly_spending_data(weekly_payload)
        context = RetrievedAdviceContext(
            query="배달 소비 절약 방법",
            source="guide.pdf",
            content="배달 주문 횟수를 미리 제한한다.",
            page_number=2,
        )

        payload = make_weekly_feedback_input(
            weekly_data=weekly_data,
            interpretation_result={"cause_result": CauseAnalysisResult()},
            advice_contexts=[context],
            user_profile=UserProfileContext(
                user_id=1,
                name="김토스",
                saving_goal_text="비상금 300만원 만들기",
            ),
        )

        self.assertEqual(
            set(payload),
            {
                "weekly_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )
        self.assertIn("weekly_summary", payload["weekly_json"])
        self.assertIn("cause_result", payload["interpretation_json"])
        self.assertIn("배달 소비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("비상금 300만원 만들기", payload["user_profile_json"])
        self.assertIn("{}", payload["memory_context_json"])

    def test_generate_weekly_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """주간 분석, 해석, RAG 검색, 최종 피드백 체인이 순서대로 실행되는지 검증한다."""
        interpretation_result = {
            "cause_result": CauseAnalysisResult(
                intervention_targets=[
                    InterventionTarget(
                        target_type="delivery_frequency_review",
                        title="배달 주문 빈도 점검 타겟",
                        linked_cause="반복 배달 소비 가능성",
                        target_json_path="repeat_patterns.delivery.transaction_count",
                        reason="배달 주문 반복성이 주간 지출에 영향을 줄 수 있음",
                        query_hint="배달 주문 빈도 점검",
                    )
                ]
            )
        }
        advice_context = RetrievedAdviceContext(
            query="배달 주문 1회 줄이기 실천 방법",
            source="saving.pdf",
            content="주간 외식 예산을 먼저 정한다.",
        )
        feedback_result = WeeklyFeedbackResult(
            summary_title="이번 주는 배달과 통신비를 같이 봐야 합니다",
            feedback_message="배달과 고정비가 주간 지출을 키웠습니다.",
            key_evidences=[
                WeeklyFeedbackEvidence(
                    evidence_type="spending_metric",
                    title="이번 주 총 소비",
                    detail="이번 주 총 소비가 112000원입니다.",
                    source_json_path="weekly_summary.this_week_total",
                )
            ],
            action_items=[
                WeeklyFeedbackAction(
                    title="배달 주문 1회 줄이기",
                    detail="다음 주 배달 주문 횟수를 미리 제한합니다.",
                    target_json_path="repeat_patterns.delivery",
                    urgency="this_week",
                )
            ],
            next_week_mission="다음 주 배달 주문은 1회만 허용합니다.",
        )
        interpretation_chain = MagicMock()
        interpretation_chain.invoke.return_value = interpretation_result
        feedback_chain = MagicMock()
        feedback_chain.invoke.return_value = feedback_result
        memory_summary_chain = MagicMock()
        memory_summary_chain.invoke.return_value = (
            "지난주 배달 미션과 이번 주 배달 제한 미션을 함께 요약합니다."
        )
        timing_records: list[FeedbackTimingRecord] = []

        with TemporaryDirectory() as tmp_dir:
            settings = _make_weekly_settings(Path(tmp_dir))
            ensure_user_database(settings=settings)
            with session_scope(settings) as db_session:
                db_session.add(
                    UserMemoryModel(
                        user_id=1,
                        period_type="weekly",
                        summary="지난주에는 배달 주문이 반복되었습니다.",
                    )
                )
                db_session.add(
                    SessionModel(
                        user_id=1,
                        analysis_date="2024-03-25",
                        period_type="weekly",
                        analysis_result='{"weekly_summary": {"this_week_total": 22000}}',
                        feedback_reason='[{"title": "지난주 배달 반복"}]',
                        todo_tomorrow="지난주 배달 주문을 1회 줄입니다.",
                    )
                )
            weekly_json = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start="2024-04-01",
                week_end="2024-04-07",
                settings=settings,
            )

            with (
                patch(
                    "catcher_llm.services.consumption_feedback.weekly_feedback."
                    "build_weekly_consumption_analysis_json",
                    return_value=weekly_json,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.weekly_feedback."
                    "build_weekly_spending_analysis_chain",
                    return_value=interpretation_chain,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.weekly_feedback."
                    "retrieve_feedback_contexts",
                    return_value=[advice_context],
                ) as retrieve_contexts,
                patch(
                    "catcher_llm.services.consumption_feedback.weekly_feedback."
                    "build_weekly_feedback_chain",
                    return_value=feedback_chain,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.weekly_feedback."
                    "build_memory_summary_chain",
                    return_value=memory_summary_chain,
                ),
            ):
                result = generate_weekly_feedback(
                    member_id=1,
                    week_start="2024-04-01",
                    week_end="2024-04-07",
                    settings=settings,
                    timing_callback=timing_records.append,
                )

            with session_scope(settings) as db_session:
                saved_session = (
                    db_session.query(SessionModel)
                    .filter_by(
                        user_id=1,
                        analysis_date="2024-04-01",
                        period_type="weekly",
                    )
                    .one()
                )
                refreshed_memory = (
                    db_session.query(UserMemoryModel)
                    .filter_by(user_id=1, period_type="weekly")
                    .one()
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.feedback)
        assert result.feedback is not None
        self.assertEqual(result.feedback.next_week_mission, "다음 주 배달 주문은 1회만 허용합니다.")
        self.assertEqual(result.retrieved_contexts, [advice_context])
        self.assertGreaterEqual(len(result.retrieval_queries), 1)
        retrieve_contexts.assert_called_once()
        self.assertEqual(
            retrieve_contexts.call_args.kwargs["document_kinds"],
            (DocumentKind.USER_REPORT,),
        )
        interpretation_payload = interpretation_chain.invoke.call_args.args[0]
        feedback_payload = feedback_chain.invoke.call_args.args[0]
        retrieval_query_text = "\n".join(result.retrieval_queries)
        self.assertIn("weekly_summary.this_week_total", interpretation_payload["indicator_json"])
        self.assertIn(
            "weekly_metrics.weekend_spending_ratio_percent",
            interpretation_payload["indicator_json"],
        )
        self.assertIn("비상금 300만원 만들기", interpretation_payload["user_profile_json"])
        self.assertIn("배달 주문 빈도 점검", retrieval_query_text)
        self.assertIn("weekly_json", feedback_payload)
        self.assertIn("interpretation_json", feedback_payload)
        self.assertIn("retrieved_contexts", feedback_payload)
        self.assertIn("user_profile_json", feedback_payload)
        self.assertIn("memory_context_json", feedback_payload)
        self.assertIsNotNone(result.memory_context)
        assert result.memory_context is not None
        self.assertEqual(
            result.memory_context.memory_summary, "지난주에는 배달 주문이 반복되었습니다."
        )
        self.assertIn("지난주 배달 주문을 1회 줄입니다", feedback_payload["memory_context_json"])
        self.assertEqual(saved_session.feedback_message, "배달과 고정비가 주간 지출을 키웠습니다.")
        self.assertEqual(saved_session.todo_tomorrow, "다음 주 배달 주문은 1회만 허용합니다.")
        self.assertEqual(
            refreshed_memory.summary,
            "지난주 배달 미션과 이번 주 배달 제한 미션을 함께 요약합니다.",
        )
        self.assertEqual(
            [record.step_key for record in timing_records],
            [
                "weekly_analysis",
                "parse_weekly_analysis",
                "user_profile",
                "interpretation_chain",
                "retrieval_queries",
                "rag_retrieval",
                "memory_context",
                "feedback_chain",
                "save_session",
                "refresh_memory",
            ],
        )
        self.assertTrue(all(record.elapsed_seconds >= 0 for record in timing_records))
        self.assertTrue(all(record.status == "success" for record in timing_records))

    def test_weekly_feedback_retrieval_queries_use_weekly_signals(self) -> None:
        """주간 분석 지표와 행동 미션에서 최종 피드백용 RAG 검색 질의를 생성하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_weekly_settings(Path(tmp_dir))
            weekly_payload = build_weekly_consumption_analysis_json(
                member_id=1,
                week_start="2024-04-01",
                week_end="2024-04-07",
                settings=settings,
            )
        weekly_data = parse_weekly_spending_data(weekly_payload)
        queries = build_weekly_feedback_retrieval_queries(
            weekly_data,
            interpretation_result={
                "cause_result": CauseAnalysisResult(
                    intervention_targets=[
                        InterventionTarget(
                            target_type="cafe_micro_spending_review",
                            title="카페 소액 반복 결제 점검 타겟",
                            linked_cause="소액 반복 소비 가능성",
                            target_json_path="waste_detection.micro_spending.count",
                            reason="소액 결제 누적이 주간 소비에 영향을 줄 수 있음",
                            query_hint="카페 소액 반복 결제 점검",
                        )
                    ]
                )
            },
            user_profile=UserProfileContext(
                user_id=1,
                job="개발자",
                saving_goal_text="비상금 300만원 만들기",
            ),
            max_queries=5,
        )

        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(any("식비" in query or "생활" in query for query in queries))
        self.assertTrue(any("SKT통신비" in query for query in queries))
        self.assertTrue(any("카페 소액 반복 결제 점검" in query for query in queries))
        self.assertTrue(any("비상금" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
