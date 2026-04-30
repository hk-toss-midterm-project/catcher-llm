from __future__ import annotations

import unittest
from unittest.mock import patch

from catcher_llm.prompts.rag_prompt import get_catcher_consumption_benchmark_prompt
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.services.rag.catcher_consumption_benchmark import (
    generate_catcher_consumption_benchmark_rag_reply,
)


class CatcherConsumptionBenchmarkRagServiceTests(unittest.TestCase):
    def test_generate_catcher_consumption_benchmark_rag_reply_normalizes_report_fact_answers(
        self,
    ) -> None:
        """Catcher 소비 벤치마크 리포트 RAG 서비스가 보고서 수치와 순위를 직접 답변으로 정규화하는지 검증한다."""
        rag_response = RAGResponse(
            answer="설명형 답변",
            contexts=[
                RetrievedChunk(
                    source="catcher-consumption-benchmark.pdf",
                    content=(
                        "201812 회원월당 소비성 금액은 44.7만 원이다. "
                        "201812 변동비 누수 후보 비율은 87.1%이다. "
                        "절약 타깃은 온라인쇼핑, 사교활동, 정기결제 순이다. "
                        "소비후잔액부담지수는 9.73로 6개월 최고치이다."
                    ),
                )
            ],
            sources=["catcher-consumption-benchmark.pdf"],
        )

        with (
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.get_rag_pipeline_config"
            ) as get_config,
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.get_catcher_consumption_benchmark_prompt"
            ),
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.generate_rag_reply",
                side_effect=lambda *args, **kwargs: RAGResponse(
                    answer=rag_response.answer,
                    contexts=list(rag_response.contexts),
                    sources=list(rag_response.sources),
                ),
            ),
        ):
            get_config.return_value.raw_data_dir = "raw"
            get_config.return_value.source_files = ["doc.pdf"]

            result_amount = generate_catcher_consumption_benchmark_rag_reply(
                "201812 회원월당 소비성 금액은 얼마인가?"
            )
            result_ratio = generate_catcher_consumption_benchmark_rag_reply(
                "201812 변동비 누수 후보 비율은 얼마인가?"
            )
            result_priority = generate_catcher_consumption_benchmark_rag_reply(
                "절약 타깃은 어떤 순서인가?"
            )
            result_burden = generate_catcher_consumption_benchmark_rag_reply(
                "소비후잔액부담지수는 어떤 수준인가?"
            )

        self.assertEqual(result_amount.answer, "201812 회원월당 소비성 금액은 44.7만 원이다.")
        self.assertEqual(result_ratio.answer, "201812 변동비 누수 후보 비율은 87.1%이다.")
        self.assertEqual(
            result_priority.answer,
            "절약 타깃은 온라인쇼핑, 사교활동, 정기결제 순이다.",
        )
        self.assertEqual(result_burden.answer, "소비후잔액부담지수는 9.73로 6개월 최고치이다.")

    def test_get_catcher_consumption_benchmark_prompt_requests_fact_style_answer(self) -> None:
        """Catcher 소비 벤치마크 리포트 프롬프트가 보고서 기반 한 문장 사실형 답변을 요구하는지 검증한다."""
        prompt = get_catcher_consumption_benchmark_prompt()
        messages = prompt.format_messages(
            question="201812 회원월당 소비성 금액은 얼마인가?",
            context="벤치마크 문맥",
        )

        self.assertIn("Catcher 소비 벤치마크 리포트", messages[0].content)
        self.assertIn("한국인 300만 명", messages[0].content)
        self.assertIn("2018년 7월부터 12월까지", messages[0].content)
        self.assertIn("답변은 한 문장만 쓴다", messages[0].content)
        self.assertIn("201812 회원월당 소비성 금액은 얼마인가?", messages[1].content)

    def test_generate_catcher_consumption_benchmark_rag_reply_uses_tighter_default_top_k_and_prompt(
        self,
    ) -> None:
        """Catcher 소비 벤치마크 리포트 RAG 서비스가 전용 프롬프트와 기본 top_k=2를 사용하는지 검증한다."""
        with (
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.get_rag_pipeline_config"
            ) as get_config,
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.get_catcher_consumption_benchmark_prompt"
            ) as get_prompt,
            patch(
                "catcher_llm.services.rag.catcher_consumption_benchmark.generate_rag_reply"
            ) as generate_reply,
        ):
            get_config.return_value.raw_data_dir = "raw"
            get_config.return_value.source_files = ["doc.pdf"]
            get_prompt.return_value = object()
            generate_reply.return_value = RAGResponse(
                answer="원본 답변",
                contexts=[],
                sources=[],
            )

            generate_catcher_consumption_benchmark_rag_reply("질문")

        generate_reply.assert_called_once()
        self.assertEqual(generate_reply.call_args.kwargs["top_k"], 2)
        self.assertIs(generate_reply.call_args.kwargs["prompt"], get_prompt.return_value)


if __name__ == "__main__":
    unittest.main()
