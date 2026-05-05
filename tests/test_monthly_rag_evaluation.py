from __future__ import annotations

from unittest.mock import patch

from catcher_llm.config.settings import Settings
from catcher_llm.evaluation.monthly_rag import (
    build_monthly_langsmith_examples,
    build_monthly_llm_judge_prompt,
    build_monthly_rag_query_specs,
    evaluate_monthly_feedback_rules,
    format_monthly_feedback_response,
    monthly_rag_answer_nonempty,
    monthly_rag_context_recall,
    monthly_rag_langsmith_target,
    run_monthly_langsmith_evaluation,
    run_monthly_rag_query_generation,
)
from catcher_llm.schemas.consumption_feedback import (
    MonthlyFeedbackAction,
    MonthlyFeedbackEvidence,
    MonthlyFeedbackResult,
    MonthlyFeedbackServiceResult,
    RetrievedAdviceContext,
)
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk


def _make_monthly_result(
    *,
    feedback: MonthlyFeedbackResult,
    contexts: list[RetrievedAdviceContext],
) -> MonthlyFeedbackServiceResult:
    """월간 RAG 평가 테스트에 사용할 서비스 결과 모델을 생성한다."""
    return MonthlyFeedbackServiceResult(
        member_id=1,
        analysis_month="2026-04",
        feedback=feedback,
        retrieval_queries=["[user_report] 2026-04 식비 전체 사용자 월간 소비 비중"],
        retrieved_contexts=contexts,
    )


def _make_valid_feedback() -> MonthlyFeedbackResult:
    """프로젝트 월간 피드백 규칙을 만족하는 테스트용 피드백을 만든다."""
    return MonthlyFeedbackResult(
        summary_title="식비 증가 점검",
        feedback_message=(
            "이번 달 식비 지출은 전월보다 커졌고, 사용자 보고서의 전체 사용자 식비 비중도 "
            "함께 확인하면 개인 증가폭이 더 두드러집니다."
        ),
        key_evidences=[
            MonthlyFeedbackEvidence(
                evidence_type="spending_metric",
                title="식비 전월 대비 증가",
                detail="식비가 전월 대비 42,000원 증가했습니다.",
                source_json_path="category_deep[0].diff_amount",
            ),
            MonthlyFeedbackEvidence(
                evidence_type="document",
                title="전체 사용자 식비 흐름",
                detail="USER_REPORT에서 전체 사용자 식비 비중은 완만한 증가 흐름입니다.",
                source="users_report/monthly/2026-04.md",
            ),
        ],
        action_items=[
            MonthlyFeedbackAction(
                title="결제 전 식사 일정 확인",
                detail="평일 저녁 외식 결제 전 집에 있는 식재료와 다음 날 일정을 먼저 확인합니다.",
                target_json_path="category_deep[0]",
                urgency="this_month",
                related_source="users_report/monthly/2026-04.md",
            )
        ],
        next_month_mission="평일 저녁 외식 결제 전 냉장고 식재료와 다음 날 일정을 1분 동안 확인하세요.",
    )


def test_build_monthly_rag_query_specs_extracts_query_level_reference() -> None:
    """월간 RAGAS 평가 대상이 최종 피드백이 아니라 검색 쿼리별 문서 근거인지 검증한다."""
    context = RetrievedAdviceContext(
        query="[user_report] 2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content=(
            "### evidence_card: monthly_2026_04_category_change_식비\n"
            "- usable_claim: 2026-04 전체 앱 사용자 식비 소비는 전월 대비 완만하게 증가했다.\n"
            "- caution: 개인의 특정 가맹점 지출 원인을 직접 설명하지 않는다."
        ),
        document_kind="user_report",
    )
    result = _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])

    specs = build_monthly_rag_query_specs(result)

    assert len(specs) == 1
    assert specs[0].query == context.query
    assert specs[0].document_kind == "user_report"
    assert "전체 앱 사용자 식비 소비" in specs[0].reference


def test_run_monthly_rag_query_generation_uses_generate_rag_reply_per_query() -> None:
    """쿼리별 RAGAS 레코드 생성이 generate_rag_reply의 문서 RAG 답변을 사용하도록 검증한다."""
    context = RetrievedAdviceContext(
        query="[user_report] 2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content="- usable_claim: 2026-04 전체 앱 사용자 식비 소비는 전월 대비 증가했다.",
        document_kind="user_report",
    )
    result = _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])
    specs = build_monthly_rag_query_specs(result)
    rag_response = RAGResponse(
        answer="2026-04 전체 앱 사용자 식비 소비는 전월 대비 증가했습니다.",
        contexts=[
            RetrievedChunk(
                source="users_report/monthly/2026-04.md",
                content="2026-04 전체 앱 사용자 식비 소비는 전월 대비 증가했다.",
            )
        ],
        sources=["users_report/monthly/2026-04.md"],
    )

    with (
        patch("catcher_llm.evaluation.monthly_rag.get_rag_pipeline_config") as get_config,
        patch("catcher_llm.evaluation.monthly_rag.generate_rag_reply") as generate_reply,
    ):
        get_config.return_value.raw_data_dir = "raw"
        get_config.return_value.source_files = ["users_report.md"]
        get_config.return_value.prompt = object()
        generate_reply.return_value = rag_response

        records = run_monthly_rag_query_generation(specs, settings=None)

    generate_reply.assert_called_once()
    assert generate_reply.call_args.args[0] == context.query
    assert generate_reply.call_args.kwargs["raw_data_dir"] == "raw"
    assert generate_reply.call_args.kwargs["source_files"] == ["users_report.md"]
    assert records[0].user_input == context.query
    assert records[0].response == rag_response.answer
    assert records[0].retrieved_contexts == [rag_response.contexts[0].content]
    assert "전체 앱 사용자 식비 소비" in records[0].reference


def test_build_monthly_langsmith_examples_uses_query_inputs_and_reference_outputs() -> None:
    """LangSmith 평가 예제가 쿼리 입력과 기준 답변 출력을 분리해 구성되는지 검증한다."""
    context = RetrievedAdviceContext(
        query="[user_report] 2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content="- usable_claim: 2026-04 전체 앱 사용자 식비 소비는 전월 대비 증가했다.",
        document_kind="user_report",
    )
    specs = build_monthly_rag_query_specs(
        _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])
    )

    examples = build_monthly_langsmith_examples(specs)

    assert examples[0]["inputs"]["query"] == context.query
    assert examples[0]["inputs"]["document_kind"] == "user_report"
    assert "전체 앱 사용자 식비 소비" in examples[0]["outputs"]["reference"]
    assert examples[0]["metadata"]["source_count"] == 1


def test_monthly_rag_langsmith_target_runs_document_specific_rag_reply() -> None:
    """LangSmith target이 document_kind에 맞는 RAG 파이프라인으로 답변을 생성하는지 검증한다."""
    rag_response = RAGResponse(
        answer="식비 소비는 전월 대비 증가했습니다.",
        contexts=[
            RetrievedChunk(
                source="users_report/monthly/2026-04.md",
                content="식비 소비는 전월 대비 증가했다.",
            )
        ],
        sources=["users_report/monthly/2026-04.md"],
    )

    with (
        patch("catcher_llm.evaluation.monthly_rag.get_rag_pipeline_config") as get_config,
        patch("catcher_llm.evaluation.monthly_rag.generate_rag_reply") as generate_reply,
    ):
        get_config.return_value.raw_data_dir = "raw"
        get_config.return_value.source_files = ["users_report.md"]
        get_config.return_value.prompt = object()
        generate_reply.return_value = rag_response

        output = monthly_rag_langsmith_target(
            {
                "query": "[user_report] 2026-04 식비 전체 사용자 월간 소비 비중",
                "document_kind": "user_report",
            },
            settings=None,
        )

    assert output["answer"] == rag_response.answer
    assert output["sources"] == rag_response.sources
    assert output["contexts"] == [
        {
            "source": "users_report/monthly/2026-04.md",
            "content": "식비 소비는 전월 대비 증가했다.",
            "page_number": None,
        }
    ]
    assert output["document_kind"] == "user_report"


def test_monthly_langsmith_code_evaluators_score_basic_rag_outputs() -> None:
    """LangSmith code evaluator가 응답 존재와 reference-context 재현 여부를 채점하는지 검증한다."""
    nonempty = monthly_rag_answer_nonempty(
        {"query": "질문"},
        {"answer": "답변", "contexts": [{"content": "기준 답변"}]},
    )
    recall = monthly_rag_context_recall(
        {"query": "질문"},
        {"answer": "답변", "contexts": [{"content": "기준 답변"}]},
        {"reference": "기준 답변"},
    )

    assert nonempty.key == "answer_nonempty"
    assert nonempty.score is True
    assert recall.key == "reference_in_contexts"
    assert recall.score is True


def test_run_monthly_langsmith_evaluation_uploads_dataset_and_starts_experiment() -> None:
    """LangSmith 평가 실행 함수가 dataset 예제를 업로드하고 evaluate를 호출하는지 검증한다."""
    context = RetrievedAdviceContext(
        query="[user_report] 2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content="- usable_claim: 2026-04 전체 앱 사용자 식비 소비는 전월 대비 증가했다.",
        document_kind="user_report",
    )
    specs = build_monthly_rag_query_specs(
        _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])
    )

    class _FakeDataset:
        """LangSmith dataset 반환값을 대신하는 테스트용 객체다."""

        id = "dataset-id"
        name = "monthly-rag-dataset"

    class _FakeClient:
        """LangSmith Client의 dataset/example 메서드 호출을 기록하는 대역이다."""

        def __init__(self) -> None:
            """업로드된 예제를 확인할 수 있도록 내부 목록을 초기화한다."""
            self.created_examples: list[dict[str, object]] = []

        def read_dataset(self, *, dataset_name: str) -> _FakeDataset:
            """요청한 이름의 dataset이 이미 있다고 가정한다."""
            assert dataset_name == "monthly-rag-dataset"
            return _FakeDataset()

        def create_examples(
            self,
            *,
            dataset_id: str,
            examples: list[dict[str, object]],
        ) -> None:
            """LangSmith에 업로드될 예제를 기록한다."""
            assert dataset_id == "dataset-id"
            self.created_examples = examples

    class _FakeExperiment:
        """LangSmith evaluate 반환값의 주요 표시 필드를 대신한다."""

        experiment_name = "monthly-rag-exp"
        url = "https://smith.langchain.com/o/project/datasets/example"

    fake_client = _FakeClient()
    with (
        patch("catcher_llm.evaluation.monthly_rag.build_langsmith_client") as build_client,
        patch("catcher_llm.evaluation.monthly_rag.langsmith_evaluate") as evaluate,
    ):
        build_client.return_value = fake_client
        evaluate.return_value = _FakeExperiment()

        result = run_monthly_langsmith_evaluation(
            specs,
            settings=Settings(langsmith_api_key="test-key"),
            dataset_name="monthly-rag-dataset",
            experiment_prefix="monthly-rag",
        )

    assert len(fake_client.created_examples) == 1
    assert evaluate.call_args.kwargs["data"] == "monthly-rag-dataset"
    assert evaluate.call_args.kwargs["experiment_prefix"] == "monthly-rag"
    assert result.dataset_name == "monthly-rag-dataset"
    assert result.experiment_name == "monthly-rag-exp"
    assert result.experiment_url == "https://smith.langchain.com/o/project/datasets/example"


def test_evaluate_monthly_feedback_rules_passes_valid_project_rules() -> None:
    """규칙 기반 평가가 문서 근거 포함, 실행형 미션, 검색 문서 사용을 통과 처리하는지 검증한다."""
    context = RetrievedAdviceContext(
        query="2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content="전체 사용자 식비 비중은 전월 대비 완만하게 증가했습니다.",
        document_kind="user_report",
    )
    result = _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])

    summary = evaluate_monthly_feedback_rules(result)
    checks = {item.key: item for item in summary.rule_evaluations}

    assert summary.pass_rate == 1.0
    assert checks["has_document_evidence"].passed
    assert checks["no_saving_tips_context"].passed
    assert checks["mission_is_actionable"].passed
    assert checks["message_mission_not_repeated"].passed


def test_evaluate_monthly_feedback_rules_detects_monthly_rag_violations() -> None:
    """규칙 기반 평가가 문서 근거 누락, saving tips 사용, 목표형 미션을 실패로 표시하는지 검증한다."""
    feedback = MonthlyFeedbackResult(
        summary_title="식비 관리",
        feedback_message="식비를 20% 줄이세요.",
        key_evidences=[
            MonthlyFeedbackEvidence(
                evidence_type="spending_metric",
                title="식비 증가",
                detail="식비가 증가했습니다.",
                source_json_path="category_deep[0].diff_amount",
            )
        ],
        action_items=[],
        next_month_mission="식비를 20% 줄이세요.",
    )
    context = RetrievedAdviceContext(
        query="식비 절약 방법",
        source="saving_tips/food.md",
        content="식비 절약 팁입니다.",
        document_kind="saving_tips",
    )
    result = _make_monthly_result(feedback=feedback, contexts=[context])

    checks = {item.key: item for item in evaluate_monthly_feedback_rules(result).rule_evaluations}

    assert not checks["has_document_evidence"].passed
    assert not checks["no_saving_tips_context"].passed
    assert not checks["mission_is_actionable"].passed
    assert not checks["message_mission_not_repeated"].passed


def test_format_monthly_feedback_response_includes_core_fields() -> None:
    """최종 피드백 포맷터가 제목, 본문, 미션, 근거를 RAGAS 응답 문자열에 포함하는지 검증한다."""
    response = format_monthly_feedback_response(_make_valid_feedback())

    assert "summary_title: 식비 증가 점검" in response
    assert "feedback_message:" in response
    assert "next_month_mission:" in response
    assert "key_evidences:" in response
    assert "전체 사용자 식비 흐름" in response


def test_build_monthly_llm_judge_prompt_mentions_project_specific_criteria() -> None:
    """LLM Judge 프롬프트가 월간 피드백 RAG 전용 평가 기준을 포함하는지 검증한다."""
    context = RetrievedAdviceContext(
        query="2026-04 식비 전체 사용자 월간 소비 비중",
        source="users_report/monthly/2026-04.md",
        content="전체 사용자 식비 비중은 전월 대비 완만하게 증가했습니다.",
        document_kind="user_report",
    )
    result = _make_monthly_result(feedback=_make_valid_feedback(), contexts=[context])

    prompt = build_monthly_llm_judge_prompt(result)

    assert "월간 피드백 RAG 평가자" in prompt
    assert "evidence_type=document" in prompt
    assert "saving tips" in prompt
    assert "납부" in prompt
    assert "USER_REPORT" in prompt
