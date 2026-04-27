from __future__ import annotations

import unittest
from unittest.mock import patch

from catcher_llm.prompts.rag_prompt import get_welfare_prompt
from catcher_llm.services.rag.welfare import generate_welfare_rag_reply


class WelfareRagServiceTests(unittest.TestCase):
    def test_get_welfare_prompt_requests_fact_style_answer(self) -> None:
        """복지 정책 프롬프트가 한 문장 사실형 답변을 요구하는지 검증한다."""
        prompt = get_welfare_prompt()
        messages = prompt.format_messages(
            question="여성청소년 생리용품 지원의 월 지원금은 얼마인가?",
            context="복지 문맥",
        )

        self.assertIn("복지 정책 문서", messages[0].content)
        self.assertIn("답변은 한 문장만 쓴다", messages[0].content)
        self.assertIn("여성청소년 생리용품 지원의 월 지원금은 얼마인가?", messages[1].content)

    def test_generate_welfare_rag_reply_uses_tighter_default_top_k_and_prompt(self) -> None:
        """복지 정책 RAG 서비스가 전용 프롬프트와 기본 top_k=2를 사용하는지 검증한다."""
        with (
            patch("catcher_llm.services.rag.welfare.get_rag_pipeline_config") as get_config,
            patch("catcher_llm.services.rag.welfare.get_welfare_prompt") as get_prompt,
            patch("catcher_llm.services.rag.welfare.generate_rag_reply") as generate_reply,
        ):
            get_config.return_value.raw_data_dir = "raw"
            get_config.return_value.source_files = ["doc.pdf"]
            get_prompt.return_value = object()
            generate_reply.return_value = object()

            generate_welfare_rag_reply("질문")

        generate_reply.assert_called_once()
        self.assertEqual(generate_reply.call_args.kwargs["top_k"], 2)
        self.assertIs(generate_reply.call_args.kwargs["prompt"], get_prompt.return_value)


if __name__ == "__main__":
    unittest.main()
