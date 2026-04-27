from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import SessionModel, UserMemoryModel
from catcher_llm.db.session import session_scope
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
from catcher_llm.services.user_data_service import ensure_user_database


def _write_feedback_seed_csvs(csv_dir: Path) -> None:
    """일일 피드백 컨텍스트 테스트에 사용할 사용자와 거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "members_v1.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,saving_goal_text",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형,비상금 300만원 만들기",
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
    """일일 피드백 테스트가 격리된 SQLite DB를 쓰도록 설정을 만든다."""
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
        openai_api_key="test-key",
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
        """최종 일일 피드백 프롬프트가 사용자·메모리 컨텍스트까지 입력으로 받는지 검증한다."""
        prompt = build_daily_feedback_prompt()

        self.assertEqual(
            set(prompt.input_variables),
            {
                "daily_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )

        rendered = prompt.invoke(
            {
                "daily_json": '{"stable_metrics": {"today_total": 133044}}',
                "interpretation_json": '{"action_result": {"immediate_cuts": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "통신비 절약"}]',
                "user_profile_json": '{"job": "개발자", "saving_goal_text": "비상금"}',
                "memory_context_json": '{"memory_summary": "식비가 반복적으로 높다"}',
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("일일 소비 잔소리", content)
        self.assertIn("통신비 절약", content)
        self.assertIn("비상금", content)
        self.assertIn("식비가 반복적으로 높다", content)
        self.assertIn("JSON 수치 근거", content)

    def test_make_daily_feedback_input_serializes_contexts(self) -> None:
        """최종 피드백 체인 입력이 사용자·메모리 컨텍스트까지 직렬화되는지 검증한다."""
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
            user_profile={"user_id": 1, "job": "개발자", "saving_goal_text": "비상금"},
            memory_context={"memory_summary": "식비가 반복적으로 높다", "recent_sessions": []},
        )

        self.assertEqual(
            set(payload),
            {
                "daily_json",
                "interpretation_json",
                "retrieved_contexts",
                "user_profile_json",
                "memory_context_json",
            },
        )
        self.assertIn("stable_metrics", payload["daily_json"])
        self.assertIn("action_result", payload["interpretation_json"])
        self.assertIn("통신비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("비상금", payload["user_profile_json"])
        self.assertIn("식비가 반복적으로 높다", payload["memory_context_json"])
        self.assertIn("guide.pdf", serialize_advice_contexts([context]))

    def test_generate_daily_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """사용자·메모리·세션 컨텍스트를 최종 피드백에 넣고 실행 결과를 세션에 저장하는지 검증한다."""
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

        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))
            ensure_user_database(settings=settings)
            with session_scope(settings) as db_session:
                db_session.add(
                    UserMemoryModel(
                        user_id=1,
                        period_type="daily",
                        summary="최근 식비와 쇼핑 지출이 반복적으로 높다.",
                    )
                )
                db_session.add(
                    SessionModel(
                        user_id=1,
                        analysis_date="2024-03-31",
                        daily_analysis_result='{"stable_metrics": {"today_total": 2000}}',
                        feedback_reason='[{"title": "전일 소비"}]',
                        todo_tomorrow="간식 결제를 줄인다.",
                    )
                )

            daily_json = build_daily_consumption_analysis_json(
                member_id=1,
                analysis_date="2024-04-01",
                previous_date="2024-03-31",
                settings=settings,
            )

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

            with session_scope(settings) as db_session:
                saved_session = (
                    db_session.query(SessionModel)
                    .filter_by(
                        user_id=1,
                        analysis_date="2024-04-01",
                    )
                    .one()
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
        self.assertIn("user_profile_json", feedback_payload)
        self.assertIn("memory_context_json", feedback_payload)
        self.assertIn("비상금 300만원 만들기", feedback_payload["user_profile_json"])
        self.assertIn("최근 식비와 쇼핑 지출", feedback_payload["memory_context_json"])
        self.assertIn("간식 결제를 줄인다", feedback_payload["memory_context_json"])
        self.assertIsNotNone(saved_session.daily_analysis_result)
        self.assertIn("오늘 총 지출", saved_session.feedback_reason or "")
        self.assertEqual(saved_session.todo_tomorrow, "내일 오전 고정비 결제 알림을 확인합니다.")


if __name__ == "__main__":
    unittest.main()
