from __future__ import annotations

import unittest

from catcher_llm.services.ragas.welfare_eval_dataset import (
    get_welfare_eval_questions,
    get_welfare_ground_truths,
)


class WelfareEvalDatasetTests(unittest.TestCase):
    def test_welfare_eval_dataset_uses_document_grounded_questions_and_answers(self) -> None:
        """복지 정책 평가셋이 실제 문서에 나온 지원 대상과 지원금 기준에 직접 맞는지 검증한다."""
        questions = get_welfare_eval_questions()
        ground_truths = get_welfare_ground_truths()

        self.assertEqual(len(questions), len(ground_truths))
        self.assertEqual(
            questions[0],
            "여성청소년 생리용품 지원의 월 지원금은 얼마인가?",
        )
        self.assertEqual(
            ground_truths[0],
            "여성청소년 생리용품 지원의 월 지원금은 1만 4,000원이다.",
        )
        self.assertIn("13만 원", ground_truths[1])
        self.assertIn("25만 원", ground_truths[2])
        self.assertIn("10만 원", ground_truths[3])


if __name__ == "__main__":
    unittest.main()
