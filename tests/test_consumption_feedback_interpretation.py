from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from catcher_llm.analysis.user_daily_analysis import build_daily_consumption_analysis_from_frames
from catcher_llm.prompts.consumption_feedback import (
    build_consumption_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_spending_indicators,
    load_user_spending_data,
    make_spending_analysis_input,
    parse_user_spending_data,
)

_USER_PROFILE = {"user_id": 1, "job": "개발자", "persona": "절약형", "saving_goal_text": "비상금"}


class ConsumptionFeedbackInterpretationTests(unittest.TestCase):
    def test_extract_spending_indicators_matches_notebook_contract(self) -> None:
        """노트북 검증 셀의 JSON 지표 추출 계약을 제품 코드에서 재현하는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        spending_indicators = extract_spending_indicators(user_data)
        analysis_input = make_spending_analysis_input(user_data, user_profile=_USER_PROFILE)
        metric_by_name = {metric.name: metric for metric in spending_indicators.metrics}

        self.assertEqual(user_data.member_id, 1)
        self.assertEqual(user_data.analysis_date, "2024-04-01")
        self.assertEqual(metric_by_name["오늘 총 지출액"].value, 133044)
        self.assertEqual(
            metric_by_name["평소 대비 지출 증가율"].source_json_path,
            "stable_metrics.increase_rate_percent",
        )
        self.assertIs(metric_by_name["소비 급증 여부"].value, False)
        self.assertEqual(metric_by_name["고액 지출 건수"].value, 1)
        largest_increase = spending_indicators.largest_category_increase
        largest_decrease = spending_indicators.largest_category_decrease
        self.assertIsNotNone(largest_increase)
        self.assertIsNotNone(largest_decrease)
        assert largest_increase is not None
        assert largest_decrease is not None
        self.assertEqual(largest_increase.category, "생활")
        self.assertEqual(largest_decrease.category, "식비")
        self.assertEqual(spending_indicators.high_spending_items[0].description, "SKT통신비")
        self.assertEqual(spending_indicators.main_category_shift.previous_category, "식비")
        self.assertEqual(spending_indicators.main_category_shift.current_category, "생활")
        self.assertEqual(spending_indicators.time_slot_diffs[0].time_slot, "2.오전(06-11)")
        self.assertEqual(metric_by_name["마찰력 없는 지출 비중"].value, 0.0)
        self.assertEqual(
            metric_by_name["마찰력 없는 지출 비중"].source_json_path,
            "payment_behavior_analysis.frictionless_spending.ratio_percent",
        )
        self.assertIn("indicator_json", analysis_input)
        self.assertIn("raw_json", analysis_input)
        self.assertIn("user_profile_json", analysis_input)
        self.assertNotIn("report_text", analysis_input)
        self.assertIn("stable_metrics.today_total", analysis_input["indicator_json"])
        self.assertIn("비상금", analysis_input["user_profile_json"])
        self.assertFalse(analysis_input["indicator_json"].lstrip().startswith("#"))

    def test_consumption_feedback_prompts_keep_json_inputs(self) -> None:
        """소비 피드백 프롬프트가 JSON 입력만 요구하고 노트북 검증 문구를 유지하는지 검증한다."""
        user_data = load_user_spending_data(Path("notebook/team02/02_Layer4/user_data.json"))
        analysis_input = make_spending_analysis_input(user_data, user_profile=_USER_PROFILE)
        pattern_prompt = build_consumption_pattern_prompt()
        problem_prompt = build_consumption_problem_prompt()
        cause_prompt = build_consumption_cause_prompt()
        action_prompt = build_consumption_action_prompt()

        self.assertEqual(
            set(pattern_prompt.input_variables),
            {"raw_json", "indicator_json", "user_profile_json"},
        )
        self.assertEqual(
            set(problem_prompt.input_variables),
            {"raw_json", "indicator_json", "user_profile_json"},
        )
        self.assertEqual(
            set(cause_prompt.input_variables),
            {
                "raw_json",
                "indicator_json",
                "user_profile_json",
                "pattern_text",
                "problem_text",
            },
        )
        self.assertEqual(
            set(action_prompt.input_variables),
            {
                "raw_json",
                "indicator_json",
                "user_profile_json",
                "pattern_text",
                "problem_text",
                "cause_text",
            },
        )

        rendered_pattern = pattern_prompt.invoke(analysis_input)
        rendered_pattern_content = str(rendered_pattern.messages[-1].content)
        self.assertIn("stable_metrics.today_total", rendered_pattern_content)
        self.assertIn("SKT통신비", rendered_pattern_content)
        self.assertIn("비상금", rendered_pattern_content)
        self.assertNotIn("# 멤버", rendered_pattern_content)
        self.assertNotIn("마크다운 소비 보고서", rendered_pattern_content)

        rendered_cause = cause_prompt.invoke(
            {
                "raw_json": analysis_input["raw_json"],
                "indicator_json": analysis_input["indicator_json"],
                "user_profile_json": analysis_input["user_profile_json"],
                "pattern_text": "sample pattern",
                "problem_text": "sample problem",
            }
        )
        rendered_cause_content = str(rendered_cause.messages[-1].content)
        self.assertIn("sample pattern", rendered_cause_content)
        self.assertIn("sample problem", rendered_cause_content)
        self.assertIn("추출 지표 JSON", rendered_cause_content)

    def test_parse_user_spending_data_accepts_daily_analysis_json_shape(self) -> None:
        """일일 분석 서비스 JSON을 추가 변환 없이 해석 입력 모델로 읽을 수 있는지 검증한다."""
        past_frame = pd.read_csv("data/raw/csv/consumption_v1.csv", encoding="utf-8-sig")
        today_frame = pd.read_csv(
            "notebook/team02/data_pre/data_input_month.csv",
            encoding="utf-8-sig",
        )
        raw_result = build_daily_consumption_analysis_from_frames(
            past_frame,
            today_frame,
            member_id=1,
            analysis_date="2024-04-01",
            previous_date="2024-03-31",
            past_source_path="data/raw/csv/consumption_v1.csv",
            today_source_path="notebook/team02/data_pre/data_input_month.csv",
        )

        user_data = parse_user_spending_data(raw_result)
        analysis_input = make_spending_analysis_input(user_data)

        self.assertEqual(user_data.source_paths.past_source, "data/raw/csv/consumption_v1.csv")
        self.assertEqual(user_data.stable_metrics.today_total, 133044)
        self.assertEqual(
            user_data.payment_behavior_analysis.frictionless_spending.total_amount,
            1486,
        )
        self.assertIn("anomaly_detection.high_spending_items", analysis_input["indicator_json"])
        self.assertIn("payment_behavior_analysis", analysis_input["raw_json"])
        self.assertIn("마찰력 없는 지출 비중", analysis_input["indicator_json"])
        self.assertEqual(analysis_input["user_profile_json"], "{}")


if __name__ == "__main__":
    unittest.main()
