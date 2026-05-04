from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import SessionModel, UserMemoryModel
from catcher_llm.db.session import session_scope
from catcher_llm.prompts.consumption_feedback import build_monthly_feedback_prompt
from catcher_llm.schemas.consumption_feedback import (
    CauseAnalysisResult,
    InterventionTarget,
    MonthlyFeedbackAction,
    MonthlyFeedbackEvidence,
    MonthlyFeedbackResult,
    RetrievedAdviceContext,
    UserProfileContext,
)
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.monthly_feedback import (
    build_monthly_feedback_retrieval_queries,
    generate_monthly_feedback,
    make_monthly_feedback_input,
    make_monthly_spending_analysis_input,
    parse_monthly_spending_data,
)
from catcher_llm.services.consumption_feedback.timing import FeedbackTimingRecord
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.user_data_service import ensure_user_database


def _write_monthly_seed_csvs(csv_dir: Path) -> None:
    """월간 피드백 테스트에 사용할 v4 사용자와 거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v4.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,saving_goal_text,target_max_spending_amount",
                "1,김토스,29,개발자,남성,36000000,서울,Gold,절약형,비상금 300만원 만들기,300000",
            ]
        ),
        encoding="utf-8-sig",
    )
    (csv_dir / "transactions_v4.csv").write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인)",
                "1,1,20000,2024-03-05 10:00:00,배달의민족,Y,N,0,-,승인,N,식비,온라인",
                "1,2,10000,2024-03-10 11:00:00,스타벅스,Y,N,0,-,승인,N,식비,오프라인",
                "1,3,15000,2024-03-15 12:00:00,교통카드,Y,N,0,-,승인,N,교통,오프라인",
                "1,4,65000,2024-03-20 09:00:00,SKT통신비,Y,N,0,-,승인,N,생활,자동이체",
                "1,5,65000,2024-04-01 10:00:00,SKT통신비,Y,N,0,-,승인,N,생활,자동이체",
                "1,6,25000,2024-04-02 22:00:00,배달의민족,Y,N,0,-,승인,N,식비,온라인",
                "1,7,5500,2024-04-03 09:00:00,스타벅스,Y,N,0,-,승인,N,식비,오프라인",
                "1,8,3000,2024-04-04 13:00:00,CU,Y,N,0,-,승인,N,식비,오프라인",
                "1,9,30000,2024-04-06 15:00:00,쿠팡,Y,N,0,-,승인,N,쇼핑,온라인",
                "1,10,22000,2024-04-09 21:00:00,배달의민족,Y,N,0,-,승인,N,식비,온라인",
                "1,11,15000,2024-04-10 14:00:00,카카오택시,Y,N,0,-,승인,N,교통,오프라인",
                "1,12,40000,2024-04-12 18:00:00,병원,Y,N,0,-,승인,N,의료,오프라인",
            ]
        ),
        encoding="utf-8-sig",
    )


def _make_monthly_settings(root: Path) -> Settings:
    """월간 피드백 테스트가 격리된 SQLite DB를 쓰도록 설정을 만든다."""
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    _write_monthly_seed_csvs(raw_dir / "csv")
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
        openai_api_key="test-key",
    )


class ConsumptionFeedbackMonthlyFeedbackTests(unittest.TestCase):
    def test_monthly_analysis_json_uses_user_financial_context(self) -> None:
        """월간 분석 서비스가 사용자 연봉과 목표 소비 금액으로 예산·소득 지표를 계산하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            monthly_payload = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )

        monthly_metrics = cast(dict[str, object], monthly_payload["monthly_metrics"])

        self.assertEqual(monthly_metrics["monthly_budget_usage_rate_percent"], 68.5)
        self.assertEqual(monthly_metrics["monthly_remaining_budget"], 94500)
        self.assertEqual(monthly_metrics["monthly_overspend_amount"], 0)
        self.assertEqual(monthly_metrics["monthly_income_usage_rate_percent"], 6.85)
        self.assertEqual(monthly_metrics["target_spending_to_income_rate_percent"], 10.0)
        self.assertEqual(monthly_metrics["estimated_saving_amount"], 2794500)
        self.assertEqual(monthly_metrics["estimated_saving_rate_percent"], 93.15)
        self.assertEqual(monthly_metrics["target_saving_amount"], 2700000)
        self.assertEqual(monthly_metrics["target_saving_rate_percent"], 90.0)
        self.assertAlmostEqual(
            float(monthly_metrics["fixed_cost_burden_rate_percent"]),
            2.1667,
            places=4,
        )
        self.assertEqual(monthly_metrics["spending_capacity"], 2880000)
        self.assertEqual(monthly_metrics["nonessential_spending_income_rate_percent"], 2.85)

    def test_monthly_spending_analysis_input_uses_monthly_json_and_profile(self) -> None:
        """월간 분석 JSON을 월간 해석 체인의 raw/indicator/profile 입력으로 변환하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            monthly_payload = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )

        monthly_data = parse_monthly_spending_data(monthly_payload)
        analysis_input = make_monthly_spending_analysis_input(
            monthly_data,
            user_profile={"user_id": 1, "saving_goal_text": "비상금"},
        )

        self.assertEqual(monthly_data.analysis_month, "2024-04")
        self.assertIn("monthly_summary.this_month_total", analysis_input["indicator_json"])
        self.assertIn("monthly_metrics", analysis_input["raw_json"])
        self.assertIn("monthly_comparisons", analysis_input["raw_json"])
        self.assertIn(
            "monthly_metrics.fixed_cost_burden_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "monthly_metrics.monthly_budget_usage_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "monthly_metrics.monthly_income_usage_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn(
            "monthly_metrics.target_spending_to_income_rate_percent",
            analysis_input["indicator_json"],
        )
        self.assertIn("목표 달성 시 저축률", analysis_input["indicator_json"])
        self.assertIn("최근 3개월 평균 대비 지출 증감률", analysis_input["indicator_json"])
        self.assertIn("구독료 합계", analysis_input["indicator_json"])
        self.assertIn("fixed_variable.fixed_total", analysis_input["indicator_json"])
        self.assertIn("high_spending.items", analysis_input["indicator_json"])
        self.assertIn("배달의민족", analysis_input["raw_json"])
        self.assertIn("비상금", analysis_input["user_profile_json"])

    def test_monthly_feedback_prompt_requires_monthly_inputs(self) -> None:
        """최종 월간 피드백 프롬프트가 월간 분석·해석·문서 근거를 입력으로 받는지 검증한다."""
        prompt = build_monthly_feedback_prompt()

        self.assertEqual(
            set(prompt.input_variables),
            {
                "monthly_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )

        rendered = prompt.invoke(
            {
                "monthly_json": '{"monthly_summary": {"this_month_total": 242500}}',
                "interpretation_json": '{"cause_result": {"intervention_targets": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "고정비 점검"}]',
                "user_profile_json": '{"saving_goal_text": "비상금"}',
                "memory_context_json": "{}",
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("월간 소비", content)
        self.assertIn("고정비 점검", content)
        self.assertIn("비상금", content)
        self.assertIn("JSON 수치 근거", content)
        self.assertIn("cause_result.intervention_targets는 RAG 검색용 중간 후보", content)
        self.assertIn("action_result가 포함된 경우에도 최종 행동이 아니므로", content)
        self.assertIn("RAG 문서 근거와 사용자 메모리", content)
        self.assertIn("그대로 복사하지 마라", content)
        self.assertIn("feedback_message", content)
        self.assertNotIn("scolding_message", content)
        self.assertIn("ratio_context_warning", content)
        self.assertIn("내부 판단에만 사용", content)
        self.assertIn("분모", content)
        self.assertIn("노출하지 마라", content)
        self.assertIn("결론은 반복 여부와 절대금액 점검 중심", content)
        self.assertIn("비중만으로 '급증', '습관 악화', '예산 초과'를 단정하지 마라", content)

    def test_make_monthly_feedback_input_serializes_contexts(self) -> None:
        """월간 피드백 체인 입력이 월간 분석, 해석, RAG, 프로필을 JSON 문자열로 직렬화하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            monthly_payload = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )
        monthly_data = parse_monthly_spending_data(monthly_payload)
        context = RetrievedAdviceContext(
            query="고정비 절약 방법",
            source="guide.pdf",
            content="자동이체 항목을 월 1회 점검한다.",
            page_number=4,
        )

        payload = make_monthly_feedback_input(
            monthly_data=monthly_data,
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
                "monthly_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )
        self.assertIn("monthly_summary", payload["monthly_json"])
        self.assertIn("cause_result", payload["interpretation_json"])
        self.assertIn("고정비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("비상금 300만원 만들기", payload["user_profile_json"])
        self.assertIn("{}", payload["memory_context_json"])

    def test_make_monthly_feedback_input_switches_compact_and_full_payloads(self) -> None:
        """최종 월간 피드백 입력을 토큰 절감 버전과 원본 전체 버전으로 전환할 수 있는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            monthly_payload = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )
        monthly_data = parse_monthly_spending_data(monthly_payload)
        advice_contexts = [
            RetrievedAdviceContext(
                query=f"절약 방법 {index}",
                source=f"guide-{index}.pdf",
                content="긴 월간 근거 문장 " * 120,
                page_number=index,
                document_kind="saving_tips",
                usefulness_score=1.0 - (index * 0.01),
                usefulness_reason="토큰 절약 입력에는 필요 없는 긴 유용성 설명",
            )
            for index in range(7)
        ]
        memory_context = {
            "user_id": 1,
            "period_type": "monthly",
            "memory_summary": "고정비와 식비 점검이 반복적으로 필요하다",
            "recent_sessions": [
                {
                    "analysis_date": "2024-03",
                    "analysis_result": '{"monthly_summary": {"this_month_total": 999999}}',
                    "feedback_reason": '[{"title": "전월 소비"}]',
                    "todo_tomorrow": "자동이체 항목을 점검한다.",
                }
            ],
        }

        compact_payload = make_monthly_feedback_input(
            monthly_data=monthly_data,
            interpretation_result={"cause_result": CauseAnalysisResult()},
            advice_contexts=advice_contexts,
            user_profile={
                "user_id": 1,
                "name": "김토스",
                "job": "개발자",
                "region": None,
                "saving_goal_text": "비상금",
            },
            memory_context=memory_context,
        )
        full_payload = make_monthly_feedback_input(
            monthly_data=monthly_data,
            interpretation_result={"cause_result": CauseAnalysisResult()},
            advice_contexts=advice_contexts,
            user_profile={
                "user_id": 1,
                "name": "김토스",
                "job": "개발자",
                "region": None,
                "saving_goal_text": "비상금",
            },
            memory_context=memory_context,
            compact_input=False,
        )

        compact_monthly = json.loads(compact_payload["monthly_json"])
        full_monthly = json.loads(full_payload["monthly_json"])
        compact_contexts = json.loads(compact_payload["retrieved_contexts"])
        full_contexts = json.loads(full_payload["retrieved_contexts"])
        compact_profile = json.loads(compact_payload["user_profile_json"])
        full_profile = json.loads(full_payload["user_profile_json"])
        compact_memory = json.loads(compact_payload["memory_context_json"])
        full_memory = json.loads(full_payload["memory_context_json"])

        self.assertNotIn("source_path", compact_monthly)
        self.assertNotIn("outlier_thresholds", compact_monthly)
        self.assertIn("source_path", full_monthly)
        self.assertIn("outlier_thresholds", full_monthly)
        self.assertNotIn(
            "category_monthly_spending_ratio",
            compact_monthly["monthly_metrics"],
        )
        self.assertIn("category_monthly_spending_ratio", full_monthly["monthly_metrics"])
        self.assertEqual(len(compact_contexts), 5)
        self.assertEqual(len(full_contexts), 7)
        self.assertLessEqual(len(compact_contexts[0]["content"]), 500)
        self.assertNotIn("usefulness_reason", compact_contexts[0])
        self.assertIn("usefulness_reason", full_contexts[0])
        self.assertNotIn("name", compact_profile)
        self.assertEqual(full_profile["name"], "김토스")
        self.assertNotIn("analysis_result", compact_memory["recent_sessions"][0])
        self.assertIn("analysis_result", full_memory["recent_sessions"][0])
        self.assertLess(len(compact_payload["monthly_json"]), len(full_payload["monthly_json"]))

    def test_generate_monthly_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """월간 분석, 해석, RAG 검색, 최종 피드백 체인이 순서대로 실행되는지 검증한다."""
        interpretation_result = {
            "cause_result": CauseAnalysisResult(
                intervention_targets=[
                    InterventionTarget(
                        target_type="fixed_transfer_review",
                        title="자동이체 항목 점검 타겟",
                        linked_cause="고정비 반복 지출 가능성",
                        target_json_path="fixed_variable.fixed_items[0].total_amount",
                        reason="고정비 항목이 월간 소비에 반복적으로 반영됨",
                        query_hint="자동이체 항목 점검",
                    )
                ]
            )
        }
        advice_context = RetrievedAdviceContext(
            query="자동이체 항목 점검 실천 방법",
            source="saving.pdf",
            content="월말에 고정비 목록을 확인한다.",
        )
        feedback_result = MonthlyFeedbackResult(
            summary_title="이번 달은 고정비와 배달비를 같이 봐야 합니다",
            feedback_message="고정비와 반복 식비가 월간 지출을 키웠습니다.",
            key_evidences=[
                MonthlyFeedbackEvidence(
                    evidence_type="spending_metric",
                    title="이번 달 총 소비",
                    detail="이번 달 총 소비가 242500원입니다.",
                    source_json_path="monthly_summary.this_month_total",
                )
            ],
            action_items=[
                MonthlyFeedbackAction(
                    title="자동이체 항목 점검",
                    detail="다음 달 시작 전 자동이체 목록을 확인합니다.",
                    target_json_path="fixed_variable.fixed_items",
                    urgency="this_month",
                )
            ],
            next_month_mission="다음 달 첫날 자동이체 목록을 정리합니다.",
        )
        interpretation_chain = MagicMock()
        interpretation_chain.invoke.return_value = interpretation_result
        feedback_chain = MagicMock()
        feedback_chain.invoke.return_value = feedback_result
        memory_summary_chain = MagicMock()
        memory_summary_chain.invoke.return_value = (
            "지난달 자동이체 점검과 이번 달 미션을 함께 요약합니다."
        )
        timing_records: list[FeedbackTimingRecord] = []

        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            ensure_user_database(settings=settings)
            with session_scope(settings) as db_session:
                db_session.add(
                    UserMemoryModel(
                        user_id=1,
                        period_type="monthly",
                        summary="지난달에는 자동이체와 식비 점검이 필요했습니다.",
                    )
                )
                db_session.add(
                    SessionModel(
                        user_id=1,
                        analysis_date="2024-03",
                        period_type="monthly",
                        analysis_result='{"monthly_summary": {"this_month_total": 110000}}',
                        feedback_reason='[{"title": "지난달 자동이체 부담"}]',
                        todo_tomorrow="지난달 자동이체 목록을 점검합니다.",
                    )
                )
            monthly_json = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )

            with (
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "build_monthly_consumption_analysis_json",
                    return_value=monthly_json,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "build_monthly_spending_analysis_chain",
                    return_value=interpretation_chain,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "retrieve_feedback_contexts",
                    return_value=[advice_context],
                ) as retrieve_contexts,
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "build_monthly_feedback_chain",
                    return_value=feedback_chain,
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "build_memory_summary_chain",
                    return_value=memory_summary_chain,
                ),
            ):
                result = generate_monthly_feedback(
                    member_id=1,
                    analysis_month="2024-04",
                    settings=settings,
                    compact_feedback_input=False,
                    timing_callback=timing_records.append,
                )

            with session_scope(settings) as db_session:
                saved_session = (
                    db_session.query(SessionModel)
                    .filter_by(
                        user_id=1,
                        analysis_date="2024-04",
                        period_type="monthly",
                    )
                    .one()
                )
                refreshed_memory = (
                    db_session.query(UserMemoryModel)
                    .filter_by(user_id=1, period_type="monthly")
                    .one()
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.feedback)
        assert result.feedback is not None
        self.assertEqual(
            result.feedback.next_month_mission, "다음 달 첫날 자동이체 목록을 정리합니다."
        )
        self.assertEqual(result.retrieved_contexts, [advice_context])
        self.assertGreaterEqual(len(result.retrieval_queries), 1)
        retrieve_contexts.assert_called_once()
        self.assertEqual(
            retrieve_contexts.call_args.kwargs["document_kinds"],
            (
                DocumentKind.USER_REPORT,
                DocumentKind.CATCHER_CONSUMPTION_BENCHMARK,
                DocumentKind.KCA_REPORT,
            ),
        )
        interpretation_payload = interpretation_chain.invoke.call_args.args[0]
        feedback_payload = feedback_chain.invoke.call_args.args[0]
        retrieval_query_text = "\n".join(result.retrieval_queries)
        self.assertIn(
            "monthly_summary.this_month_total",
            interpretation_payload["indicator_json"],
        )
        self.assertIn(
            "monthly_metrics.fixed_cost_burden_rate_percent",
            interpretation_payload["indicator_json"],
        )
        self.assertIn("비상금 300만원 만들기", interpretation_payload["user_profile_json"])
        self.assertIn("자동이체 항목 점검", retrieval_query_text)
        self.assertIn("monthly_json", feedback_payload)
        self.assertIn("interpretation_json", feedback_payload)
        self.assertIn("retrieved_contexts", feedback_payload)
        self.assertIn("user_profile_json", feedback_payload)
        self.assertIn("memory_context_json", feedback_payload)
        self.assertIn("outlier_thresholds", feedback_payload["monthly_json"])
        self.assertIsNotNone(result.memory_context)
        assert result.memory_context is not None
        self.assertEqual(
            result.memory_context.memory_summary,
            "지난달에는 자동이체와 식비 점검이 필요했습니다.",
        )
        self.assertIn("지난달 자동이체 목록을 점검합니다", feedback_payload["memory_context_json"])
        self.assertEqual(
            saved_session.feedback_message,
            "고정비와 반복 식비가 월간 지출을 키웠습니다.",
        )
        self.assertEqual(saved_session.todo_tomorrow, "다음 달 첫날 자동이체 목록을 정리합니다.")
        self.assertEqual(
            refreshed_memory.summary,
            "지난달 자동이체 점검과 이번 달 미션을 함께 요약합니다.",
        )
        self.assertEqual(
            [record.step_key for record in timing_records],
            [
                "monthly_analysis",
                "parse_monthly_analysis",
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

    def test_monthly_feedback_retrieval_queries_use_monthly_signals(self) -> None:
        """월간 분석 지표와 행동 미션에서 최종 피드백용 RAG 검색 질의를 생성하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
            monthly_payload = build_monthly_consumption_analysis_json(
                member_id=1,
                analysis_month="2024-04",
                settings=settings,
            )
        monthly_data = parse_monthly_spending_data(monthly_payload)
        queries = build_monthly_feedback_retrieval_queries(
            monthly_data,
            interpretation_result={
                "cause_result": CauseAnalysisResult(
                    intervention_targets=[
                        InterventionTarget(
                            target_type="cafe_micro_spending_review",
                            title="카페 소액 반복 결제 점검 타겟",
                            linked_cause="소액 반복 소비 가능성",
                            target_json_path="monthly_micro_spending.count",
                            reason="소액 결제 누적이 월간 소비에 영향을 줄 수 있음",
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
