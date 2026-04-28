from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.consumption_feedback import build_monthly_feedback_prompt
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
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


def _write_monthly_seed_csvs(csv_dir: Path) -> None:
    """월간 피드백 테스트에 사용할 사용자와 거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v1.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,saving_goal_text",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형,비상금 300만원 만들기",
            ]
        ),
        encoding="utf-8-sig",
    )
    (csv_dir / "transactions_v1.csv").write_text(
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
        self.assertIn(
            "monthly_metrics.fixed_cost_burden_rate_percent",
            analysis_input["indicator_json"],
        )
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
            },
        )

        rendered = prompt.invoke(
            {
                "monthly_json": '{"monthly_summary": {"this_month_total": 242500}}',
                "interpretation_json": '{"action_result": {"next_week_missions": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "고정비 점검"}]',
                "user_profile_json": '{"saving_goal_text": "비상금"}',
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("월간 소비", content)
        self.assertIn("고정비 점검", content)
        self.assertIn("비상금", content)
        self.assertIn("JSON 수치 근거", content)

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
            interpretation_result={"action_result": ActionAnalysisResult()},
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
            },
        )
        self.assertIn("monthly_summary", payload["monthly_json"])
        self.assertIn("action_result", payload["interpretation_json"])
        self.assertIn("고정비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("비상금 300만원 만들기", payload["user_profile_json"])

    def test_generate_monthly_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """월간 분석, 해석, RAG 검색, 최종 피드백 체인이 순서대로 실행되는지 검증한다."""
        interpretation_result = {
            "action_result": ActionAnalysisResult(
                next_week_missions=[
                    ActionMission(
                        action_type="monthly_cut",
                        title="자동이체 항목 점검",
                        detail="다음 달 시작 전 자동이체 항목을 점검한다.",
                        target_json_path="fixed_variable.fixed_items",
                        expected_effect="고정비 절감",
                        urgency="this_month",
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

        with TemporaryDirectory() as tmp_dir:
            settings = _make_monthly_settings(Path(tmp_dir))
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
                ),
                patch(
                    "catcher_llm.services.consumption_feedback.monthly_feedback."
                    "build_monthly_feedback_chain",
                    return_value=feedback_chain,
                ),
            ):
                result = generate_monthly_feedback(
                    member_id=1,
                    analysis_month="2024-04",
                    settings=settings,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.feedback)
        assert result.feedback is not None
        self.assertEqual(
            result.feedback.next_month_mission, "다음 달 첫날 자동이체 목록을 정리합니다."
        )
        self.assertEqual(result.retrieved_contexts, [advice_context])
        self.assertGreaterEqual(len(result.retrieval_queries), 1)
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
                "action_result": ActionAnalysisResult(
                    next_week_missions=[
                        ActionMission(
                            action_type="monthly_cut",
                            title="카페 결제 절반 줄이기",
                            detail="카페 결제를 줄인다.",
                            target_json_path="repeat_patterns.cafe",
                            expected_effect="소액 반복 소비 절감",
                            urgency="this_month",
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
        self.assertTrue(any("카페 결제 절반 줄이기" in query for query in queries))
        self.assertTrue(any("비상금" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
