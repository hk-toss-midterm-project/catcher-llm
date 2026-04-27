from __future__ import annotations

import unittest

from catcher_llm.services.ragas.self_report_eval_dataset import (
    get_self_report_eval_questions,
    get_self_report_ground_truths,
)


class SelfReportEvalDatasetTests(unittest.TestCase):
    def test_self_report_eval_dataset_uses_report_grounded_questions_and_answers(self) -> None:
        """소비 자기진단 리포트 평가셋이 보고서 핵심 진단 문장에 직접 맞는지 검증한다."""
        questions = get_self_report_eval_questions()
        ground_truths = get_self_report_ground_truths()

        self.assertEqual(len(questions), len(ground_truths))
        self.assertEqual(
            questions[0],
            "201812 회원월당 소비성 금액은 얼마인가?",
        )
        self.assertEqual(
            ground_truths[0],
            "201812 회원월당 소비성 금액은 44.7만 원이다.",
        )
        self.assertIn("87.1%", ground_truths[1])
        self.assertEqual(
            questions[2],
            "절약 타깃은 어떤 순서인가?",
        )
        self.assertEqual(
            ground_truths[2],
            "절약 타깃은 온라인쇼핑, 사교활동, 정기결제 순이다.",
        )
        self.assertEqual(
            questions[3],
            "소비후잔액부담지수는 어떤 수준인가?",
        )
        self.assertEqual(
            ground_truths[3],
            "소비후잔액부담지수는 9.73로 6개월 최고치이다.",
        )


if __name__ == "__main__":
    unittest.main()
