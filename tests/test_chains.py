from __future__ import annotations

import unittest

from catcher_llm.chains.router_chain import route_request
from catcher_llm.prompts.chat_prompt import build_chat_prompt


class ChainTests(unittest.TestCase):
    def test_route_request_uses_summary_keywords(self) -> None:
        self.assertEqual(route_request("이 문서를 요약해줘"), "summary")

    def test_route_request_uses_rag_keywords(self) -> None:
        self.assertEqual(route_request("문서에서 관련 내용을 찾아줘"), "rag")

    def test_chat_prompt_exposes_expected_inputs(self) -> None:
        prompt = build_chat_prompt()
        self.assertEqual(set(prompt.input_variables), {"history", "input"})


if __name__ == "__main__":
    unittest.main()
