from __future__ import annotations

import unittest
from unittest.mock import patch

from datasets import Dataset

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.ragas.ragas_kca_eval import (
    build_kca_ragas_dataset,
    run_kca_ragas_eval,
)


class RagasKcaEvalTests(unittest.TestCase):
    def test_build_kca_ragas_dataset_converts_rag_response(self) -> None:
        """RAG 응답을 ragas 평가용 데이터셋 컬럼 구조로 변환하는지 검증한다."""
        rag_response = RAGResponse(
            answer="절약 습관을 점검하세요.",
            contexts=[
                RetrievedChunk(
                    source="kca-report.pdf",
                    content="불필요한 소비를 구분하고 지출 내역을 점검해야 합니다.",
                )
            ],
            sources=["kca-report.pdf"],
        )

        with (
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.get_kca_eval_questions",
                return_value=["질문"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.get_kca_ground_truths",
                return_value=["정답"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.generate_kca_report_rag_reply",
                return_value=rag_response,
            ) as generate_reply,
        ):
            dataset = build_kca_ragas_dataset()

        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset["user_input"], ["질문"])
        self.assertEqual(dataset["response"], ["절약 습관을 점검하세요."])
        self.assertEqual(
            dataset["retrieved_contexts"],
            [["불필요한 소비를 구분하고 지출 내역을 점검해야 합니다."]],
        )
        self.assertEqual(dataset["reference"], ["정답"])
        generate_reply.assert_called_once_with("질문", settings=None)

    def test_run_kca_ragas_eval_uses_project_models(self) -> None:
        """ragas 평가가 프로젝트 설정 기반 LLM과 임베딩 모델을 명시적으로 사용한다."""
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="test-anthropic-key",
            anthropic_model="claude-test",
            embedding_provider="ollama",
            ollama_embedding_model="nomic-test",
            ollama_base_url="http://localhost:11434",
        )
        dataset = Dataset.from_dict(
            {
                "user_input": ["질문"],
                "response": ["응답"],
                "retrieved_contexts": [["근거 문맥"]],
                "reference": ["정답"],
            }
        )
        fake_llm = object()
        fake_embeddings = object()
        fake_result = object()

        with (
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.build_kca_ragas_dataset",
                return_value=dataset,
            ),
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.get_chat_model",
                return_value=fake_llm,
            ) as get_chat_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.get_embeddings_model",
                return_value=fake_embeddings,
            ) as get_embeddings_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_kca_eval.evaluate",
                return_value=fake_result,
            ) as evaluate_mock,
        ):
            result = run_kca_ragas_eval(settings=settings)

        self.assertIs(result, fake_result)
        get_chat_model_mock.assert_called_once_with(settings)
        get_embeddings_model_mock.assert_called_once_with(settings)
        evaluate_mock.assert_called_once()
        evaluate_kwargs = evaluate_mock.call_args.kwargs
        self.assertIs(evaluate_kwargs["dataset"], dataset)
        self.assertIs(evaluate_kwargs["llm"], fake_llm)
        self.assertIs(evaluate_kwargs["embeddings"], fake_embeddings)
        metric_names = [metric.name for metric in evaluate_kwargs["metrics"]]
        self.assertEqual(
            metric_names,
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )
        answer_relevancy_metric = evaluate_kwargs["metrics"][1]
        self.assertEqual(answer_relevancy_metric.strictness, 1)


if __name__ == "__main__":
    unittest.main()
