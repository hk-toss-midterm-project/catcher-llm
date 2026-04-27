from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from catcher_llm.prompts.rag_prompt import get_saving_tips_prompt
from catcher_llm.schemas.rag import RAGResponse
from catcher_llm.services.rag.saving_tips import generate_saving_tips_rag_reply


class SavingTipsRagServiceTests(unittest.TestCase):
    def test_generate_saving_tips_rag_reply_restates_question_in_answer(self) -> None:
        """절약 팁 RAG 서비스가 답변을 질문 재진술형 문장으로 정규화하는지 검증한다."""
        with (
            patch("catcher_llm.services.rag.saving_tips.get_rag_pipeline_config") as get_config,
            patch("catcher_llm.services.rag.saving_tips.get_saving_tips_prompt"),
            patch(
                "catcher_llm.services.rag.saving_tips._get_saving_tips_source_files",
                return_value=[],
            ),
            patch("catcher_llm.services.rag.saving_tips.generate_rag_reply") as generate_reply,
        ):
            get_config.return_value.raw_data_dir = Path("raw")
            get_config.return_value.source_files = ["doc.pdf"]
            generate_reply.return_value = RAGResponse(
                answer="텀블러 사용, 편의점 커피 대체, 주간 카페 예산 설정, 홈카페 이용이다.",
                contexts=[],
                sources=[],
            )

            result = generate_saving_tips_rag_reply(
                "카페 소비가 잦을 경우 실천할 수 있는 절약 행동은 무엇인가?"
            )

        self.assertEqual(
            result.answer,
            "카페 소비가 잦을 경우 실천할 수 있는 절약 행동은 텀블러 사용, 편의점 커피 대체, 주간 카페 예산 설정, 홈카페 이용이다.",
        )

    def test_get_saving_tips_prompt_requests_question_restatement_style(self) -> None:
        """절약 팁 프롬프트가 질문 핵심어를 답변 첫머리에 다시 반영하도록 지시하는지 검증한다."""
        prompt = get_saving_tips_prompt()
        messages = prompt.format_messages(
            question="카페 소비가 잦을 경우 실천할 수 있는 절약 행동은 무엇인가?",
            context="절약 팁 문맥",
        )

        self.assertIn("질문의 핵심 명사를 답변 첫머리에 다시 포함한다", messages[0].content)
        self.assertIn("카페 소비가 잦을 경우", messages[1].content)

    def test_generate_saving_tips_rag_reply_uses_tighter_default_top_k_prompt_and_text_source(
        self,
    ) -> None:
        """절약 팁 RAG 서비스가 전용 프롬프트와 가이드 텍스트 소스를 사용하는지 검증한다."""
        with (
            patch("catcher_llm.services.rag.saving_tips.get_rag_pipeline_config") as get_config,
            patch("catcher_llm.services.rag.saving_tips.get_saving_tips_prompt") as get_prompt,
            patch(
                "catcher_llm.services.rag.saving_tips._get_saving_tips_source_files"
            ) as get_sources,
            patch("catcher_llm.services.rag.saving_tips.generate_rag_reply") as generate_reply,
        ):
            get_config.return_value.raw_data_dir = Path("raw")
            get_config.return_value.source_files = ["doc.pdf"]
            get_prompt.return_value = object()
            get_sources.return_value = [Path("raw") / "txt" / "saving_guides.txt"]
            generate_reply.return_value = RAGResponse(
                answer="원본 답변이다.",
                contexts=[],
                sources=[],
            )

            generate_saving_tips_rag_reply("질문")

        generate_reply.assert_called_once()
        generate_kwargs = generate_reply.call_args.kwargs
        self.assertEqual(generate_kwargs["top_k"], 2)
        self.assertIs(generate_kwargs["prompt"], get_prompt.return_value)
        self.assertEqual(
            generate_kwargs["source_files"],
            [Path("raw") / "txt" / "saving_guides.txt"],
        )


if __name__ == "__main__":
    unittest.main()
