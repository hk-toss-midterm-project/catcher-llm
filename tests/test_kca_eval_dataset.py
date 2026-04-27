from __future__ import annotations

import unittest

from catcher_llm.services.ragas.kca_eval_dataset import (
    get_kca_eval_questions,
    get_kca_ground_truths,
)


class KcaEvalDatasetTests(unittest.TestCase):
    def test_kca_eval_dataset_uses_corpus_grounded_questions_and_answers(self) -> None:
        """KCA 평가셋이 문서에 직접 등장하는 사실형 질문과 정답으로 구성되는지 검증한다."""
        questions = get_kca_eval_questions()
        ground_truths = get_kca_ground_truths()

        self.assertEqual(
            questions[0],
            "2023년 소비자문제 경험률이 가장 높았던 품목은 무엇인가?",
        )
        self.assertEqual(
            ground_truths[0],
            "소비자문제 경험률이 가장 높았던 품목은 식품(농수축산물·가공식품, 건강식품)이다.",
        )
        self.assertIn("소비자문제유형", questions[2])
        self.assertIn("상품·서비스 품질불량", ground_truths[2])
        self.assertIn("안전한 결제시스템 도입", ground_truths[3])
        self.assertIn("식품·외식 분야", ground_truths[4])


if __name__ == "__main__":
    unittest.main()
