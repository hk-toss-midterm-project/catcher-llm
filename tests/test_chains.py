from __future__ import annotations

import logging
import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from catcher_llm.chains.chat_chain import build_chat_chain
from catcher_llm.chains.router_chain import route_request
from catcher_llm.prompts.chat_prompt import build_chat_prompt

logger = logging.getLogger(__name__)


class ChainTests(unittest.TestCase):
    def test_route_request_uses_summary_keywords(self) -> None:
        self.assertEqual(route_request("이 문서를 요약해줘"), "summary")

    def test_route_request_uses_rag_keywords(self) -> None:
        self.assertEqual(route_request("문서에서 관련 내용을 찾아줘"), "rag")

    def test_chat_prompt_exposes_expected_inputs(self) -> None:
        prompt = build_chat_prompt()
        self.assertEqual(set(prompt.input_variables), {"history", "input"})

    def test_build_chat_chain_invokes_llm_with_fake_model(self) -> None:
        chain = build_chat_chain(llm=FakeListChatModel(responses=["stubbed reply"]))

        result = chain.invoke({"history": "user: 안녕", "input": "간단히 답해줘"})

        self.assertEqual(result, "stubbed reply")

    def test_route_request_logs_selected_chain(self) -> None:
        user_input = "이 문서를 요약해줘"
        logger.info("calling route_request with input=%s", user_input)

        result = route_request(user_input)

        logger.info("route_request returned route=%s", result)
        self.assertEqual(result, "summary")


if __name__ == "__main__":
    unittest.main()
