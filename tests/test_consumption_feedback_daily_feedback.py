from __future__ import annotations

import json
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
    CauseAnalysisResult,
    DailyFeedbackAction,
    DailyFeedbackEvidence,
    DailyFeedbackResult,
    InterventionTarget,
    RetrievedAdviceContext,
    UserProfileContext,
)
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    DailyFeedbackTimingRecord,
    build_feedback_retrieval_queries,
    generate_daily_feedback,
    load_user_profile_context,
    make_daily_feedback_input,
    retrieve_feedback_contexts,
    serialize_advice_contexts,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    load_user_spending_data,
)
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.user_data_service import ensure_user_database


def _write_feedback_seed_csvs(csv_dir: Path) -> None:
    """일일 피드백 컨텍스트 테스트에 사용할 v4 사용자와 거래 CSV를 작성한다."""
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
    def test_load_user_profile_context_includes_financial_target_fields(self) -> None:
        """사용자 프로필 컨텍스트가 연소득과 월 목표 소비 한도를 함께 제공하는지 검증한다."""
        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))
            user_profile = load_user_profile_context(member_id=1, settings=settings)

        self.assertEqual(user_profile.income, "36000000")
        self.assertEqual(user_profile.annual_income, 36_000_000)
        self.assertEqual(user_profile.monthly_income, 3_000_000)
        self.assertEqual(user_profile.target_max_spending_amount, 300_000)

    def test_build_feedback_retrieval_queries_uses_spending_and_intervention_targets(
        self,
    ) -> None:
        """소비 분석, 원인 개입 타겟, 사용자 프로필에서 RAG 검색 질의를 생성하는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        user_profile = UserProfileContext(
            user_id=1,
            job="개발자",
            persona="절약형",
            saving_goal_text="비상금 300만원 만들기",
        )
        interpretation_result = {
            "cause_result": CauseAnalysisResult(
                intervention_targets=[
                    InterventionTarget(
                        target_type="fixed_cost_review",
                        title="통신비 자동이체 점검 타겟",
                        linked_cause="고정비 고액 결제 점검 가능성",
                        target_json_path="anomaly_detection.high_spending_items[0].amount",
                        reason="통신비 결제가 고액 지출 항목으로 관찰됨",
                        query_hint="통신비 자동이체 점검",
                    )
                ]
            )
        }

        queries = build_feedback_retrieval_queries(
            user_data,
            interpretation_result=interpretation_result,
            user_profile=user_profile,
            max_queries=5,
        )

        self.assertGreaterEqual(len(queries), 3)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(any("생활" in query for query in queries))
        self.assertTrue(any("SKT통신비" in query for query in queries))
        self.assertTrue(any("통신비 자동이체 점검" in query for query in queries))
        self.assertTrue(any("비상금" in query for query in queries))

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
                "interpretation_json": '{"cause_result": {"intervention_targets": []}}',
                "retrieved_contexts": '[{"source": "guide.pdf", "content": "통신비 절약"}]',
                "user_profile_json": '{"job": "개발자", "saving_goal_text": "비상금"}',
                "memory_context_json": '{"memory_summary": "식비가 반복적으로 높다"}',
            }
        )
        content = str(rendered.messages[-1].content)

        self.assertIn("일일 소비 피드백", content)
        self.assertIn("비난, 조롱, 과장 표현은 피한다", content)
        self.assertIn("통신비 절약", content)
        self.assertIn("비상금", content)
        self.assertIn("식비가 반복적으로 높다", content)
        self.assertIn("JSON 수치 근거", content)
        self.assertIn("scolding_message", content)
        self.assertIn("비워두지 마라", content)
        self.assertIn("cause_result.intervention_targets는 RAG 검색용 중간 후보", content)
        self.assertIn("action_result가 포함된 경우에도 최종 행동이 아니므로", content)
        self.assertIn("RAG 문서 근거와 사용자 메모리", content)
        self.assertIn("그대로 복사하지 마라", content)
        self.assertIn("ratio_context_warning", content)
        self.assertIn("내부 판단에만 사용", content)
        self.assertIn("분모", content)
        self.assertIn("노출하지 마라", content)
        self.assertIn("결론은 반복 여부와 절대금액 점검 중심", content)
        self.assertIn("비중만으로 '급증', '습관 악화', '예산 초과'를 단정하지 마라", content)

    def test_daily_feedback_result_expands_blank_scolding_message(self) -> None:
        """최종 일일 피드백 본문이 비어 있으면 근거와 미션으로 저장 가능한 본문을 보강하는지 검증한다."""
        feedback = DailyFeedbackResult(
            summary_title="오늘 생활비 지출 점검",
            scolding_message=" ",
            key_evidences=[
                DailyFeedbackEvidence(
                    evidence_type="spending_metric",
                    title="생활비 비중 증가",
                    detail="생활 카테고리가 오늘 지출의 77.12%를 차지했습니다.",
                    source_json_path="stable_metrics.category_ratio_changes[0]",
                )
            ],
            action_items=[
                DailyFeedbackAction(
                    title="생활비 영수증 확인",
                    detail="오늘 결제한 생활비 항목을 한 번 정리합니다.",
                    target_json_path="stable_metrics.category_ratio_changes[0]",
                    urgency="immediate",
                )
            ],
            tomorrow_mission="내일은 생활비 결제 전에 필요한 항목인지 먼저 확인해보세요.",
        )

        self.assertGreaterEqual(len(feedback.scolding_message), 80)
        self.assertIn("생활 카테고리", feedback.scolding_message)
        self.assertIn("내일은 생활비", feedback.scolding_message)

    def test_daily_feedback_result_expands_too_short_scolding_message(self) -> None:
        """최종 일일 피드백 본문이 지나치게 짧으면 기존 문장을 살려 근거 설명을 덧붙이는지 검증한다."""
        feedback = DailyFeedbackResult(
            summary_title="오늘 소비가 좋아요",
            scolding_message="좋은 흐름이에요.",
            key_evidences=[
                DailyFeedbackEvidence(
                    evidence_type="spending_metric",
                    title="무소비일",
                    detail="오늘 총 지출액은 0원입니다.",
                    source_json_path="stable_metrics.today_total",
                )
            ],
            action_items=[],
            tomorrow_mission="내일은 가벼운 산책으로 좋은 리듬을 이어가세요.",
        )

        self.assertGreaterEqual(len(feedback.scolding_message), 80)
        self.assertTrue(feedback.scolding_message.startswith("좋은 흐름이에요."))
        self.assertIn("오늘 총 지출액은 0원", feedback.scolding_message)

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
            interpretation_result={"cause_result": CauseAnalysisResult()},
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
        self.assertIn("cause_result", payload["interpretation_json"])
        self.assertIn("통신비 절약 방법", payload["retrieved_contexts"])
        self.assertIn("비상금", payload["user_profile_json"])
        self.assertIn("식비가 반복적으로 높다", payload["memory_context_json"])
        self.assertIn("guide.pdf", serialize_advice_contexts([context]))

    def test_make_daily_feedback_input_switches_compact_and_full_payloads(self) -> None:
        """최종 피드백 입력을 토큰 절감 버전과 원본 전체 버전으로 전환할 수 있는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        advice_contexts = [
            RetrievedAdviceContext(
                query=f"절약 방법 {index}",
                source=f"guide-{index}.pdf",
                content="긴 근거 문장 " * 120,
                page_number=index,
                document_kind="saving_tips",
                usefulness_score=1.0 - (index * 0.01),
                usefulness_reason="토큰 절약 입력에는 필요 없는 긴 유용성 설명",
            )
            for index in range(7)
        ]
        memory_context = {
            "user_id": 1,
            "period_type": "daily",
            "memory_summary": "식비가 반복적으로 높다",
            "recent_sessions": [
                {
                    "analysis_date": "2024-03-31",
                    "analysis_result": '{"stable_metrics": {"today_total": 999999}}',
                    "feedback_reason": '[{"title": "전일 소비"}]',
                    "todo_tomorrow": "간식 결제를 줄인다.",
                }
            ],
        }

        compact_payload = make_daily_feedback_input(
            user_data=user_data,
            interpretation_result={"cause_result": CauseAnalysisResult()},
            advice_contexts=advice_contexts,
            user_profile={
                "user_id": 1,
                "name": None,
                "job": "개발자",
                "region": None,
                "saving_goal_text": "비상금",
            },
            memory_context=memory_context,
        )
        full_payload = make_daily_feedback_input(
            user_data=user_data,
            interpretation_result={"cause_result": CauseAnalysisResult()},
            advice_contexts=advice_contexts,
            user_profile={
                "user_id": 1,
                "name": None,
                "job": "개발자",
                "region": None,
                "saving_goal_text": "비상금",
            },
            memory_context=memory_context,
            compact_input=False,
        )

        compact_daily = json.loads(compact_payload["daily_json"])
        full_daily = json.loads(full_payload["daily_json"])
        compact_contexts = json.loads(compact_payload["retrieved_contexts"])
        full_contexts = json.loads(full_payload["retrieved_contexts"])
        compact_profile = json.loads(compact_payload["user_profile_json"])
        full_profile = json.loads(full_payload["user_profile_json"])
        compact_memory = json.loads(compact_payload["memory_context_json"])
        full_memory = json.loads(full_payload["memory_context_json"])

        self.assertNotIn("source_paths", compact_daily)
        self.assertNotIn("outlier_thresholds", compact_daily)
        self.assertIn("source_paths", full_daily)
        self.assertIn("outlier_thresholds", full_daily)
        self.assertEqual(len(compact_contexts), 5)
        self.assertEqual(len(full_contexts), 7)
        self.assertLessEqual(len(compact_contexts[0]["content"]), 500)
        self.assertNotIn("usefulness_reason", compact_contexts[0])
        self.assertIn("usefulness_reason", full_contexts[0])
        self.assertNotIn("name", compact_profile)
        self.assertIsNone(full_profile["name"])
        self.assertNotIn("analysis_result", compact_memory["recent_sessions"][0])
        self.assertIn("analysis_result", full_memory["recent_sessions"][0])

    def test_retrieve_feedback_contexts_searches_each_document_kind_and_filters_useful_records(
        self,
    ) -> None:
        """문서 종류별 벡터스토어를 각각 검색하고 유용한 청크만 피드백 근거로 남기는지 검증한다."""

        def fake_retrieve_context_records(
            question: str,
            chunk_size: int,
            chunk_overlap: int,
            top_k: int,
            *,
            raw_data_dir: Path | str | None = None,
            source_files: object | None = None,
            settings: Settings | None = None,
        ) -> list[dict[str, str | int | None]]:
            """테스트용 검색 함수로 문서 종류별 검색 결과를 고정한다."""
            if raw_data_dir is None:
                return []
            raw_dir = Path(raw_data_dir)
            if raw_dir.name == "saving_tips":
                return [
                    {
                        "source": "saving.pdf",
                        "content": "배달 주문 횟수를 정하고 주간 예산을 먼저 잠그면 지출을 줄일 수 있다.",
                        "page_number": 2,
                    }
                ]
            if raw_dir.name == "welfare":
                return [
                    {
                        "source": "welfare.pdf",
                        "content": "청년 주거 지원 정책은 보증금과 월세 신청 조건을 확인해야 한다.",
                        "page_number": 7,
                    }
                ]
            if raw_dir.name == "catcher_consumption_benchmark":
                return [
                    {
                        "source": "benchmark.pdf",
                        "content": "Catcher 소비 벤치마크는 온라인쇼핑과 정기결제 흐름을 비교한다.",
                        "page_number": 3,
                    }
                ]
            return []

        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))

            with patch(
                "catcher_llm.services.consumption_feedback.daily_feedback.retrieve_context_records",
                side_effect=fake_retrieve_context_records,
            ) as retrieve_records:
                contexts = retrieve_feedback_contexts(
                    ["배달 주문 지출 줄이는 방법"],
                    top_k=1,
                    document_kinds=[
                        DocumentKind.SAVING_TIPS,
                        DocumentKind.CATCHER_CONSUMPTION_BENCHMARK,
                        DocumentKind.WELFARE,
                    ],
                    usefulness_threshold=0.3,
                    settings=settings,
                )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].document_kind, "saving_tips")
        self.assertGreaterEqual(contexts[0].usefulness_score or 0.0, 0.3)
        self.assertIn("saving_tips", contexts[0].query)
        self.assertEqual(retrieve_records.call_count, 3)
        searched_dirs = [
            Path(call.kwargs["raw_data_dir"]).relative_to(settings.raw_data_dir)
            for call in retrieve_records.call_args_list
        ]
        self.assertEqual(
            searched_dirs,
            [
                Path("pdf") / "saving_tips",
                Path("pdf") / "catcher_consumption_benchmark",
                Path("pdf") / "welfare",
            ],
        )

    def test_retrieve_feedback_contexts_keeps_best_fallback_when_all_scores_are_low(
        self,
    ) -> None:
        """모든 문서 청크가 기준 미만이면 최고 점수 후보를 fallback 근거로 남기는지 검증한다."""

        def fake_retrieve_context_records(
            question: str,
            chunk_size: int,
            chunk_overlap: int,
            top_k: int,
            *,
            raw_data_dir: Path | str | None = None,
            source_files: object | None = None,
            settings: Settings | None = None,
        ) -> list[dict[str, str | int | None]]:
            """테스트용 검색 함수로 기준 미만 후보를 반환한다."""
            return [
                {
                    "source": "saving.pdf",
                    "content": "배달 주문은 한 번 더 생각하고 결정한다.",
                    "page_number": 1,
                }
            ]

        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))

            with patch(
                "catcher_llm.services.consumption_feedback.daily_feedback.retrieve_context_records",
                side_effect=fake_retrieve_context_records,
            ):
                contexts = retrieve_feedback_contexts(
                    ["배달 주문 지출 줄이는 방법"],
                    top_k=1,
                    document_kinds=[DocumentKind.SAVING_TIPS],
                    usefulness_threshold=0.95,
                    settings=settings,
                )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].document_kind, "saving_tips")
        self.assertLess(contexts[0].usefulness_score or 0.0, 0.95)
        self.assertIn("fallback", contexts[0].usefulness_reason or "")

    def test_retrieve_feedback_contexts_searches_each_query_without_concatenating(
        self,
    ) -> None:
        """여러 RAG 질의를 하나로 합치지 않고 문서 종류별로 각각 검색하는지 검증한다."""

        def fake_retrieve_context_records(
            question: str,
            chunk_size: int,
            chunk_overlap: int,
            top_k: int,
            *,
            raw_data_dir: Path | str | None = None,
            source_files: object | None = None,
            settings: Settings | None = None,
        ) -> list[dict[str, str | int | None]]:
            """테스트용 검색 함수로 검색 질의별 청크를 반환한다."""
            return [
                {
                    "source": "welfare.pdf",
                    "content": f"{question} 청년 지원 혜택 신청 조건",
                    "page_number": 1,
                }
            ]

        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))

            with patch(
                "catcher_llm.services.consumption_feedback.daily_feedback.retrieve_context_records",
                side_effect=fake_retrieve_context_records,
            ) as retrieve_records:
                contexts = retrieve_feedback_contexts(
                    ["청년 주거 지원", "통신비 복지 혜택"],
                    top_k=4,
                    document_kinds=[DocumentKind.WELFARE],
                    usefulness_threshold=0.1,
                    settings=settings,
                )

        searched_questions = [call.args[0] for call in retrieve_records.call_args_list]
        self.assertEqual(retrieve_records.call_count, 2)
        self.assertIn("청년 주거 지원", searched_questions[0])
        self.assertIn("통신비 복지 혜택", searched_questions[1])
        self.assertTrue(all(" / " not in question for question in searched_questions))
        self.assertEqual(len(contexts), 2)

    def test_generate_daily_feedback_orchestrates_analysis_interpretation_rag_and_feedback(
        self,
    ) -> None:
        """사용자·메모리·세션 컨텍스트를 최종 피드백에 넣고 실행 결과를 세션에 저장하는지 검증한다."""
        interpretation_result = {"cause_result": CauseAnalysisResult()}
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
        memory_summary_chain = MagicMock()
        memory_summary_chain.invoke.side_effect = RuntimeError("fallback summary")
        timing_records: list[DailyFeedbackTimingRecord] = []

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
                patch(
                    "catcher_llm.services.consumption_feedback.daily_feedback."
                    "build_memory_summary_chain",
                    return_value=memory_summary_chain,
                ),
            ):
                result = generate_daily_feedback(
                    member_id=1,
                    analysis_date=date(2024, 4, 1),
                    previous_date=date(2024, 3, 31),
                    settings=settings,
                    timing_callback=timing_records.append,
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
                refreshed_memory = (
                    db_session.query(UserMemoryModel)
                    .filter_by(
                        user_id=1,
                        period_type="daily",
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
        self.assertEqual(
            retrieve_contexts.call_args.kwargs["document_kinds"],
            (DocumentKind.WELFARE,),
        )
        feedback_payload = feedback_chain.invoke.call_args.args[0]
        interpretation_payload = interpretation_chain.invoke.call_args.args[0]
        retrieval_query_text = "\n".join(result.retrieval_queries)
        self.assertIn("user_profile_json", interpretation_payload)
        self.assertIn("비상금 300만원 만들기", interpretation_payload["user_profile_json"])
        self.assertIn("비상금 300만원 만들기", retrieval_query_text)
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
        self.assertIn("2024-03-31", refreshed_memory.summary)
        self.assertIn("2024-04-01", refreshed_memory.summary)
        self.assertIn("간식 결제를 줄인다", refreshed_memory.summary)
        self.assertIn("내일 오전 고정비 결제 알림을 확인합니다.", refreshed_memory.summary)
        self.assertEqual(
            [record.step_key for record in timing_records],
            [
                "daily_analysis",
                "parse_daily_analysis",
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

    def test_generate_daily_feedback_selects_interpretation_mode_builder(self) -> None:
        """일일 피드백 생성 서비스가 요청한 해석 모드에 맞는 체인 빌더를 사용하는지 검증한다."""
        interpretation_result = {"cause_result": CauseAnalysisResult()}
        advice_context = RetrievedAdviceContext(
            query="생활 소비 절약 방법",
            source="saving.pdf",
            content="구독과 고정비를 점검한다.",
            page_number=None,
        )
        feedback_result = DailyFeedbackResult(
            summary_title="오늘은 고정비부터 확인하세요",
            scolding_message="고정비가 오늘 소비를 크게 키웠습니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="내일 오전 고정비 결제 알림을 확인합니다.",
        )
        interpretation_chain = MagicMock()
        interpretation_chain.invoke.return_value = interpretation_result
        feedback_chain = MagicMock()
        feedback_chain.invoke.return_value = feedback_result

        with TemporaryDirectory() as tmp_dir:
            settings = _make_feedback_settings(Path(tmp_dir))
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
                ) as split_builder,
                patch(
                    "catcher_llm.services.consumption_feedback.daily_feedback."
                    "build_balanced_spending_analysis_chain",
                    return_value=interpretation_chain,
                ) as balanced_builder,
                patch(
                    "catcher_llm.services.consumption_feedback.daily_feedback."
                    "build_unified_spending_analysis_chain",
                ) as unified_builder,
                patch(
                    "catcher_llm.services.consumption_feedback.daily_feedback."
                    "retrieve_feedback_contexts",
                    return_value=[advice_context],
                ),
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
                    interpretation_mode="balanced",
                    compact_feedback_input=False,
                )

        self.assertIsNone(result.error)
        balanced_builder.assert_called_once()
        split_builder.assert_not_called()
        unified_builder.assert_not_called()
        feedback_payload = feedback_chain.invoke.call_args.args[0]
        self.assertIn("source_paths", feedback_payload["daily_json"])
        self.assertIn("outlier_thresholds", feedback_payload["daily_json"])


if __name__ == "__main__":
    unittest.main()
