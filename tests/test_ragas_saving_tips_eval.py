from __future__ import annotations

import unittest
from unittest.mock import patch

from datasets import Dataset

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.ragas.ragas_saving_tips_eval import (
    build_saving_tips_ragas_dataset,
    run_saving_tips_ragas_eval,
)


class RagasSavingTipsEvalTests(unittest.TestCase):
    def test_build_saving_tips_ragas_dataset_converts_rag_response(self) -> None:
        """절약 팁 RAG 응답을 ragas 평가 데이터셋 구조로 변환하는지 검증한다."""
        rag_response = RAGResponse(
            answer="체크카드는 과소비를 줄이는 데 도움이 된다.",
            contexts=[
                RetrievedChunk(
                    source="saving-tips.pdf",
                    content="체크카드는 지출 통제에 도움이 된다.",
                )
            ],
            sources=["saving-tips.pdf"],
        )

        with (
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.get_saving_tips_eval_questions",
                return_value=["질문"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.get_saving_tips_ground_truths",
                return_value=["정답"],
            ),
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.generate_saving_tips_rag_reply",
                return_value=rag_response,
            ) as generate_reply,
        ):
            dataset = build_saving_tips_ragas_dataset()

        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset["user_input"], ["질문"])
        self.assertEqual(dataset["response"], ["체크카드는 과소비를 줄이는 데 도움이 된다."])
        self.assertEqual(dataset["retrieved_contexts"], [["체크카드는 지출 통제에 도움이 된다."]])
        self.assertEqual(dataset["reference"], ["정답"])
        generate_reply.assert_called_once_with("질문", settings=None)

    def test_run_saving_tips_ragas_eval_uses_project_models(self) -> None:
        """절약 팁 ragas 평가가 프로젝트 설정 기반 모델을 사용하는지 검증한다."""
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
                "catcher_llm.services.ragas.ragas_saving_tips_eval.build_saving_tips_ragas_dataset",
                return_value=dataset,
            ),
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.get_chat_model",
                return_value=fake_llm,
            ) as get_chat_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.get_embeddings_model",
                return_value=fake_embeddings,
            ) as get_embeddings_model_mock,
            patch(
                "catcher_llm.services.ragas.ragas_saving_tips_eval.evaluate",
                return_value=fake_result,
            ) as evaluate_mock,
        ):
            result = run_saving_tips_ragas_eval(settings=settings)

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
