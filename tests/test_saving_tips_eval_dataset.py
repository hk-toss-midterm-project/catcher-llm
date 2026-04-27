from __future__ import annotations

import unittest

from catcher_llm.services.ragas.saving_tips_eval_dataset import (
    get_saving_tips_eval_questions,
    get_saving_tips_ground_truths,
)


class SavingTipsEvalDatasetTests(unittest.TestCase):
    def test_saving_tips_eval_dataset_uses_corpus_grounded_questions_and_answers(self) -> None:
        """절약 팁 평가셋이 문서에 직접 있는 절약 행동과 정책 질문으로 구성되는지 검증한다."""
        questions = get_saving_tips_eval_questions()
        ground_truths = get_saving_tips_ground_truths()

        self.assertEqual(len(questions), len(ground_truths))
        self.assertEqual(
            questions[0],
            "배달 소비가 반복될 경우 실천할 수 있는 절약 행동은 무엇인가?",
        )
        self.assertEqual(
            ground_truths[0],
            "주간 횟수 제한, 식재료 사전 구매, 밀키트 대체, 냉동식품 활용이 배달 소비 절약 행동이다.",
        )
        self.assertIn("텀블러 사용", ground_truths[1])
        self.assertIn("결제 전 10분 대기", ground_truths[2])
        self.assertIn("청년도약계좌", ground_truths[3])


if __name__ == "__main__":
    unittest.main()
