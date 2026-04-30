from __future__ import annotations

import unittest
from unittest.mock import patch

from datasets import Dataset

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval import (
    build_catcher_consumption_benchmark_ragas_dataset,
    run_catcher_consumption_benchmark_ragas_eval,
)


class RagasCatcherConsumptionBenchmarkEvalTests(unittest.TestCase):
    def test_build_catcher_consumption_benchmark_ragas_dataset_converts_rag_response(self) -> None:
        """Catcher 소비 벤치마크 리포트 RAG 응답을 ragas 평가 데이터셋 구조로 변환하는지 검증한다."""
        rag_response = RAGResponse(
            answer="고정지출과 변동지출을 함께 점검해야 한다.",
            contexts=[
                RetrievedChunk(
                    source="catcher-consumption-benchmark.pdf",
                    content="지출 항목을 고정지출과 변동지출로 나누어 점검한다.",
                )
            ],
            sources=["catcher-consumption-benchmark.pdf"],
        )

        with (
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.get_catcher_consumption_benchmark_eval_questions",
                return_value=["질문"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.get_catcher_consumption_benchmark_ground_truths",
                return_value=["정답"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.generate_catcher_consumption_benchmark_rag_reply",
                return_value=rag_response,
            ) as generate_reply,
        ):
            dataset = build_catcher_consumption_benchmark_ragas_dataset()

        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset["user_input"], ["질문"])
        self.assertEqual(dataset["response"], ["고정지출과 변동지출을 함께 점검해야 한다."])
        self.assertEqual(
            dataset["retrieved_contexts"],
            [["지출 항목을 고정지출과 변동지출로 나누어 점검한다."]],
        )
        self.assertEqual(dataset["reference"], ["정답"])
        generate_reply.assert_called_once_with("질문", settings=None)

    def test_run_catcher_consumption_benchmark_ragas_eval_uses_project_models(self) -> None:
        """Catcher 소비 벤치마크 리포트 ragas 평가가 프로젝트 설정 기반 모델을 사용하는지 검증한다."""
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
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.build_catcher_consumption_benchmark_ragas_dataset",
                return_value=dataset,
            ),
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.get_chat_model",
                return_value=fake_llm,
            ) as get_chat_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.get_embeddings_model",
                return_value=fake_embeddings,
            ) as get_embeddings_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_catcher_consumption_benchmark_eval.evaluate",
                return_value=fake_result,
            ) as evaluate_mock,
        ):
            result = run_catcher_consumption_benchmark_ragas_eval(settings=settings)

        self.assertIs(result, fake_result)
        get_chat_model_mock.assert_called_once_with(settings)
        get_embeddings_model_mock.assert_called_once_with(settings)
        evaluate_mock.assert_called_once()
        evaluate_kwargs = evaluate_mock.call_args.kwargs
        self.assertIs(evaluate_kwargs["dataset"], dataset)
        self.assertIs(evaluate_kwargs["llm"], fake_llm)
        self.assertIs(evaluate_kwargs["embeddings"], fake_embeddings)


if __name__ == "__main__":
    unittest.main()
