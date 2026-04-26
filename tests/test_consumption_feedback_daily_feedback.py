from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.consumption_feedback import build_daily_feedback_prompt
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    ActionMission,
    DailyFeedbackAction,
    DailyFeedbackEvidence,
    DailyFeedbackResult,
    RetrievedAdviceContext,
)
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    build_feedback_retrieval_queries,
    generate_daily_feedback,
    make_daily_feedback_input,
    serialize_advice_contexts,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    load_user_spending_data,
)


class ConsumptionFeedbackDailyFeedbackTests(unittest.TestCase):
    def test_build_feedback_retrieval_queries_uses_spending_and_action_signals(self) -> None:
        """소비 분석과 해석 결과에서 RAG 검색에 쓸 소비 코칭 질의를 생성하는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        interpretation_result = {
            "action_result": ActionAnalysisResult(
                immediate_cuts=[
                    ActionMission(
                        action_type="fixed_cost_cut",
                        title="통신비 자동이체 점검",
                        detail="오늘 결제된 통신비 요금제를 확인한다.",
                        target_json_path="anomaly_detection.high_spending_items",
                        expected_effect="고정비 절감",
                        urgency="immediate",
                    )
                ]
            )
        }

        queries = build_feedback_retrieval_queries(
            user_data,
            interpretation_result=interpretation_result,
            max_queries=5,
        )

        self.assertGreaterEqual(len(queries), 3)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(any("생활" in query for query in queries))
        self.assertTrue(any("SKT통신비" in query for query in queries))
        self.assertTrue(any("통신비 자동이체 점검" in query for query in queries))

    def test_daily_feedback_prompt_requires_json_and_retrieved_contexts(self) -> None:
        """최종 일일 피드백 프롬프트가 분석 JSON, 해석 JSON, RAG 근거를 입력으로 받는지 검증한다."""
        prompt = build_daily_feedback_prompt()

        self.assertEqual(
            set(prompt.input_variables),
            {"daily_json", "interpretation_json", "retrieved_contexts"},
        )

        rendered = prompt.invoke(
            {
                "daily_json": '{"stable_metrics": {"today_total": 133044}}',
                "interpretation_json": '{"action_result": {"immediate_cuts": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "통신비 절약"}]',
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("일일 소비 잔소리", content)
        self.assertIn("통신비 절약", content)
        self.assertIn("JSON 수치 근거", content)

    def test_make_daily_feedback_input_serializes_contexts(self) -> None:
        """최종 피드백 체인 입력이 JSON 문자열 3종으로 직렬화되는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        context = RetrievedAdviceContext(
            query="통신비 절약 방법",
            source="guide.pdf",
            content="요금제를 점검한다.",
            page_number=3,
        )

        payload = make_daily_feedback_input(
            user_data=user_data,
            interpretation_result={"action_result": ActionAnalysisResult()},
            advice_contexts=[context],
        )

        self.assertEqual(set(payload), {"daily_json", "interpretation_json", "retrieved_contexts"})
        self.assertIn("stable_metrics", payload["daily_json"])
        self.assertIn("action_result", payload["interpretation_json"])
        self.assertIn("통신비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("guide.pdf", serialize_advice_contexts([context]))

    def test_generate_daily_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """일일 분석, 해석, RAG 조회, 최종 피드백 체인이 순서대로 연결되는지 검증한다."""
        settings = Settings(openai_api_key="test-key")
        daily_json = build_daily_consumption_analysis_json(
            member_id=1,
            analysis_date="2024-04-01",
            previous_date="2024-03-31",
        )
        interpretation_result = {"action_result": ActionAnalysisResult()}
        advice_context = RetrievedAdviceContext(
            query="생활 소비 절약 방법",
            source="saving.pdf",
            content="구독과 고정비를 점검한다.",
            page_number=None,
        )
        feedback_result = DailyFeedbackResult(
            summary_title="오늘은 고정비부터 확인하세요",
            scolding_message="SKT통신비가 오늘 소비를 크게 키웠으니 그냥 넘기면 안 됩니다.",
            key_evidences=[
                DailyFeedbackEvidence(
                    evidence_type="spending_metric",
                    title="오늘 총 지출",
                    detail="오늘 총 지출액이 133044원입니다.",
                    source_json_path="stable_metrics.today_total",
                )
            ],
            action_items=[
                DailyFeedbackAction(
                    title="통신비 요금제 확인",
                    detail="현재 요금제와 사용량을 비교합니다.",
                    target_json_path="anomaly_detection.high_spending_items",
                    urgency="immediate",
                )
            ],
            tomorrow_mission="내일 오전 고정비 결제 알림을 확인합니다.",
        )
        interpretation_chain = MagicMock()
        interpretation_chain.invoke.return_value = interpretation_result
        feedback_chain = MagicMock()
        feedback_chain.invoke.return_value = feedback_result

        with (
            patch(
                "catcher_llm.services.consumption_feedback.daily_feedback."
                "build_daily_consumption_analysis_json",
                return_value=daily_json,
            ),
            patch(
                "catcher_llm.services.consumption_feedback.daily_feedback."
                "build_spending_analysis_chain",
                return_value=interpretation_chain,
            ),
            patch(
                "catcher_llm.services.consumption_feedback.daily_feedback."
                "retrieve_feedback_contexts",
                return_value=[advice_context],
            ) as retrieve_contexts,
            patch(
                "catcher_llm.services.consumption_feedback.daily_feedback."
                "build_daily_feedback_chain",
                return_value=feedback_chain,
            ),
        ):
            result = generate_daily_feedback(
                member_id=1,
                analysis_date=date(2024, 4, 1),
                previous_date=date(2024, 3, 31),
                settings=settings,
            )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.feedback)
        assert result.feedback is not None
        self.assertEqual(result.feedback.summary_title, "오늘은 고정비부터 확인하세요")
        self.assertEqual(result.retrieved_contexts, [advice_context])
        self.assertGreaterEqual(len(result.retrieval_queries), 1)
        retrieve_contexts.assert_called_once()
        feedback_payload = feedback_chain.invoke.call_args.args[0]
        self.assertIn("daily_json", feedback_payload)
        self.assertIn("interpretation_json", feedback_payload)
        self.assertIn("retrieved_contexts", feedback_payload)


if __name__ == "__main__":
    unittest.main()
