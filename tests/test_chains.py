from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from catcher_llm.chains.chat_chain import build_chat_chain
from catcher_llm.chains.consumption_feedback import build_memory_summary_chain
from catcher_llm.chains.rag_chain import build_rag_chain
from catcher_llm.chains.router_chain import route_request
from catcher_llm.chains.summary_chain import build_summary_chain
from catcher_llm.config.settings import Settings
from catcher_llm.prompts.chat_prompt import build_chat_prompt
from catcher_llm.prompts.rag_prompt import build_rag_prompt

logger = logging.getLogger(__name__)


class ChainTests(unittest.TestCase):
    def test_route_request_uses_summary_keywords(self) -> None:
        self.assertEqual(route_request("이 문서를 요약해줘"), "summary")

    def test_route_request_uses_rag_keywords(self) -> None:
        self.assertEqual(route_request("문서에서 관련 내용을 찾아줘"), "rag")

    def test_chat_prompt_exposes_expected_inputs(self) -> None:
        prompt = build_chat_prompt()
        self.assertEqual(set(prompt.input_variables), {"history", "input"})

    def test_rag_prompt_uses_korean_instructions(self) -> None:
        """기본 RAG 프롬프트가 한국어 지시문과 입력 라벨을 사용하는지 검증한다."""
        prompt = build_rag_prompt()
        messages = prompt.format_messages(
            history="이전 대화",
            context="검색 컨텍스트",
            question="질문",
        )

        self.assertIn("검색된 문맥", messages[0].content)
        self.assertIn("문맥이 부족하면", messages[0].content)
        self.assertIn("이전 대화", messages[1].content)
        self.assertIn("검색된 문맥", messages[1].content)
        self.assertIn("질문", messages[1].content)

    def test_build_chat_chain_invokes_llm_with_fake_model(self) -> None:
        chain = build_chat_chain(llm=FakeListChatModel(responses=["stubbed reply"]))

        result = chain.invoke({"history": "user: 안녕", "input": "간단히 답해줘"})

        self.assertEqual(result, "stubbed reply")

    def test_build_chat_chain_passes_temperature_to_model_factory(self) -> None:
        """채팅 체인이 호출 옵션 temperature를 모델 팩토리에 전달하는지 검증한다."""
        settings = Settings(openai_api_key="test-key")
        fake_llm = FakeListChatModel(responses=["stubbed reply"])

        with patch(
            "catcher_llm.chains.chat_chain.get_chat_model", return_value=fake_llm
        ) as factory:
            chain = build_chat_chain(settings=settings, temperature=0.7)

        self.assertEqual(chain.invoke({"history": "", "input": "hello"}), "stubbed reply")
        factory.assert_called_once_with(settings, temperature=0.7)

    def test_build_summary_chain_passes_temperature_to_model_factory(self) -> None:
        """요약 체인이 호출 옵션 temperature를 모델 팩토리에 전달하는지 검증한다."""
        settings = Settings(openai_api_key="test-key")
        fake_llm = FakeListChatModel(responses=["summary"])

        with patch(
            "catcher_llm.chains.summary_chain.get_chat_model",
            return_value=fake_llm,
        ) as factory:
            chain = build_summary_chain(settings=settings, temperature=0.2)

        self.assertEqual(chain.invoke({"text": "긴 글"}), "summary")
        factory.assert_called_once_with(settings, temperature=0.2)

    def test_build_memory_summary_chain_uses_period_prompt_and_string_output(self) -> None:
        """메모리 요약 체인이 기간 라벨 프롬프트와 문자열 출력을 연결하는지 검증한다."""
        chain = build_memory_summary_chain(
            "일일",
            llm=FakeListChatModel(responses=["최근 식비 반복과 개선 흐름이 보입니다."]),
        )

        result = chain.invoke({"session_list": "- 2024-04-01: 핵심근거 식비"})

        self.assertEqual(result, "최근 식비 반복과 개선 흐름이 보입니다.")

    def test_build_rag_chain_passes_temperature_to_model_factory(self) -> None:
        """RAG 체인이 호출 옵션 temperature를 모델 팩토리에 전달하는지 검증한다."""
        settings = Settings(openai_api_key="test-key")
        fake_llm = FakeListChatModel(responses=["rag answer"])

        with patch("catcher_llm.chains.rag_chain.get_chat_model", return_value=fake_llm) as factory:
            chain = build_rag_chain(settings=settings, temperature=0.0)

        result = chain.invoke({"history": "", "context": "context", "question": "question"})
        self.assertEqual(result, "rag answer")
        factory.assert_called_once_with(settings, temperature=0.0)

    def test_route_request_logs_selected_chain(self) -> None:
        user_input = "이 문서를 요약해줘"
        logger.info("calling route_request with input=%s", user_input)

        result = route_request(user_input)

        logger.info("route_request returned route=%s", result)
        self.assertEqual(result, "summary")


if __name__ == "__main__":
    unittest.main()
