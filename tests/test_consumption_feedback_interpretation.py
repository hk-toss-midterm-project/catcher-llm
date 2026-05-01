from __future__ import annotations

import re
import unittest
from pathlib import Path

import pandas as pd
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from catcher_llm.analysis.user_daily_analysis import build_daily_consumption_analysis_from_frames
from catcher_llm.chains.consumption_feedback.analysis import (
    build_balanced_spending_analysis_chain,
    build_spending_analysis_chain,
    build_unified_spending_analysis_chain,
)
from catcher_llm.chains.consumption_feedback.monthly import build_monthly_spending_analysis_chain
from catcher_llm.chains.consumption_feedback.sanitization import sanitize_spending_analysis_payload
from catcher_llm.chains.consumption_feedback.weekly import build_weekly_spending_analysis_chain
from catcher_llm.prompts.consumption_feedback import (
    build_consumption_action_prompt,
    build_consumption_cause_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
    build_consumption_unified_analysis_prompt,
    build_monthly_consumption_action_prompt,
    build_monthly_consumption_cause_prompt,
    build_monthly_consumption_pattern_prompt,
    build_monthly_consumption_problem_prompt,
    build_weekly_consumption_action_prompt,
    build_weekly_consumption_cause_prompt,
    build_weekly_consumption_pattern_prompt,
    build_weekly_consumption_problem_prompt,
)
from catcher_llm.schemas.consumption_feedback import (
    ActionMission,
    CauseActionAnalysisResult,
    CauseAnalysisResult,
    EvidenceItem,
    GroupCompetitionMetric,
    InterventionTarget,
    PatternAnalysisResult,
    ProblemAnalysisResult,
    SpendingAnalysisResult,
    SpendingFinding,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_spending_indicators,
    load_user_spending_data,
    make_spending_analysis_input,
    parse_user_spending_data,
)

_USER_PROFILE = {"user_id": 1, "job": "개발자", "persona": "절약형", "saving_goal_text": "비상금"}


class StructuredOutputFakeChatModel:
    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda:
        """구조화 출력 체인이 기본 Pydantic 모델을 반환하도록 만드는 테스트 더블이다."""
        return RunnableLambda(lambda _input: schema())


class SanitizingFakeChatModel:
    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda:
        """후처리 회귀 테스트용 구조화 출력 모델을 반환하는 테스트 더블이다."""
        return RunnableLambda(lambda _input: self._build_output(schema))

    def _build_output(self, schema: type[BaseModel]) -> BaseModel:
        """요청 스키마별로 원본 JSON과 맞지 않는 반복 소비 결과를 만든다."""
        invalid_pattern = PatternAnalysisResult(
            repeated_consumption=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="1회 결제를 반복 소비로 오분류",
                    detail="11번가 1회 결제를 반복 소비로 잘못 분류한 결과",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="repeat_patterns.top_merchants[0]",
                            supporting_value='{"merchant":"11번가","visit_count":1}',
                            reason="방문 횟수 1회인 상위 가맹점",
                        )
                    ],
                )
            ]
        )
        if schema is PatternAnalysisResult:
            return invalid_pattern
        if schema is ProblemAnalysisResult:
            return ProblemAnalysisResult()
        if schema is CauseAnalysisResult:
            return CauseAnalysisResult()
        if schema is CauseActionAnalysisResult:
            return CauseActionAnalysisResult()
        if schema is SpendingAnalysisResult:
            return SpendingAnalysisResult(pattern_result=invalid_pattern)
        return schema()


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
        cause_action_prompt = build_consumption_cause_action_prompt()
        unified_prompt = build_consumption_unified_analysis_prompt()

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
        self.assertEqual(
            set(cause_action_prompt.input_variables),
            {
                "raw_json",
                "indicator_json",
                "user_profile_json",
                "pattern_text",
                "problem_text",
            },
        )
        self.assertEqual(
            set(unified_prompt.input_variables),
            {"raw_json", "indicator_json", "user_profile_json"},
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

        rendered_unified = unified_prompt.invoke(analysis_input)
        rendered_unified_content = str(rendered_unified.messages[-1].content)
        self.assertIn("패턴, 문제 소비, 원인, 개선 후보·개입 타겟", rendered_unified_content)
        self.assertIn("RAG 검색과 최종 피드백 판단", rendered_unified_content)
        self.assertIn("stable_metrics.today_total", rendered_unified_content)

    def test_action_prompts_treat_outputs_as_intervention_candidates(self) -> None:
        """행동 결과 프롬프트가 최종 미션이 아닌 개입 후보를 만들도록 안내하는지 검증한다."""
        base_input = {
            "raw_json": "{}",
            "indicator_json": "{}",
            "user_profile_json": "{}",
            "pattern_text": "{}",
            "problem_text": "{}",
            "cause_text": "{}",
        }
        cause_action_input = {
            key: value for key, value in base_input.items() if key != "cause_text"
        }

        daily_action = build_consumption_action_prompt().invoke(base_input)
        daily_cause_action = build_consumption_cause_action_prompt().invoke(cause_action_input)
        daily_unified = build_consumption_unified_analysis_prompt().invoke(
            {
                "raw_json": "{}",
                "indicator_json": "{}",
                "user_profile_json": "{}",
            }
        )
        weekly_action = build_weekly_consumption_action_prompt().invoke(base_input)
        monthly_action = build_monthly_consumption_action_prompt().invoke(base_input)

        action_content = "\n".join(
            str(message.content)
            for message in [
                *daily_action.messages,
                *daily_cause_action.messages,
                *daily_unified.messages,
                *weekly_action.messages,
                *monthly_action.messages,
            ]
        )

        self.assertIn("개입 타겟", action_content)
        self.assertIn("개선 후보", action_content)
        self.assertIn("RAG 검색", action_content)
        self.assertIn("최종 미션으로 확정하지 마라", action_content)
        self.assertIn("금지 표현", action_content)
        self.assertIn("줄여보세요", action_content)
        self.assertIn("재조정하세요", action_content)
        self.assertIn("계획하세요", action_content)
        self.assertIn("절약 가능", action_content)
        self.assertIn("근거 없는 가맹점명", action_content)
        self.assertIn("프로필만으로 생활 제안", action_content)
        self.assertIn("generic immediate/substitution/mission", action_content)
        self.assertIn("_candidate", action_content)
        self.assertIn("배열 인덱스만 단독으로", action_content)
        self.assertNotIn("category_summary[4]", action_content)
        self.assertIn("총액 같은 결과 지표", action_content)
        self.assertIn("행동 기반 지표", action_content)

    def test_action_output_schema_describes_candidate_safety_constraints(self) -> None:
        """행동 후보 스키마가 최종 지시와 근거 없는 후보를 막는 기준을 설명하는지 검증한다."""
        action_schema = ActionMission.model_json_schema()
        group_schema = GroupCompetitionMetric.model_json_schema()
        intervention_schema = InterventionTarget.model_json_schema()

        action_schema_text = str(action_schema)
        group_schema_text = str(group_schema)
        intervention_schema_text = str(intervention_schema)

        self.assertIn("구체적인 snake_case", action_schema_text)
        self.assertIn("_candidate", action_schema_text)
        self.assertIn("generic immediate/substitution/mission 금지", action_schema_text)
        self.assertIn("명령형 표현 금지", action_schema_text)
        self.assertIn("줄여보세요", action_schema_text)
        self.assertIn("확정 절약액 금지", action_schema_text)
        self.assertIn("배열 인덱스만 단독으로 쓰지 말고", action_schema_text)
        self.assertNotIn("category_summary[4]", action_schema_text)
        self.assertIn("행동 기반", group_schema_text)
        self.assertIn("총액·비율 같은 결과 지표만 제안하지 마라", group_schema_text)
        self.assertIn("RAG 검색 타겟", intervention_schema_text)
        self.assertIn("최종 행동 지시가 아닌", intervention_schema_text)
        self.assertIn("연결된 원인 제목", intervention_schema_text)

    def test_cause_output_schema_supports_event_and_timing_causes(self) -> None:
        """원인 해석 스키마가 심리 단정 없이 이벤트·납부 타이밍·기간 집중 원인을 담는지 검증한다."""
        cause_result = CauseAnalysisResult()
        cause_schema_text = str(CauseAnalysisResult.model_json_schema())

        self.assertEqual(cause_result.one_off_high_spending_causes, [])
        self.assertEqual(cause_result.fixed_cost_timing_causes, [])
        self.assertEqual(cause_result.period_concentration_causes, [])
        self.assertIn("단일 고액 결제", cause_schema_text)
        self.assertIn("고정비 납부 타이밍", cause_schema_text)
        self.assertIn("요일·기간 집중", cause_schema_text)

    def test_period_prompts_use_period_specific_json_path_rules(self) -> None:
        """일·주·월 해석 프롬프트가 각 기간의 원본 JSON 경로 규칙만 안내하는지 검증한다."""
        base_input = {
            "raw_json": "{}",
            "indicator_json": "{}",
            "user_profile_json": "{}",
        }
        cause_input = {
            **base_input,
            "pattern_text": "{}",
            "problem_text": "{}",
        }

        daily_content = "\n".join(
            str(message.content)
            for rendered_prompt in [
                build_consumption_pattern_prompt().invoke(base_input),
                build_consumption_problem_prompt().invoke(base_input),
                build_consumption_cause_prompt().invoke(cause_input),
                build_consumption_unified_analysis_prompt().invoke(base_input),
            ]
            for message in rendered_prompt.messages
        )
        weekly_content = "\n".join(
            str(message.content)
            for rendered_prompt in [
                build_weekly_consumption_pattern_prompt().invoke(base_input),
                build_weekly_consumption_problem_prompt().invoke(base_input),
                build_weekly_consumption_cause_prompt().invoke(cause_input),
            ]
            for message in rendered_prompt.messages
        )
        monthly_content = "\n".join(
            str(message.content)
            for rendered_prompt in [
                build_monthly_consumption_pattern_prompt().invoke(base_input),
                build_monthly_consumption_problem_prompt().invoke(base_input),
                build_monthly_consumption_cause_prompt().invoke(cause_input),
            ]
            for message in rendered_prompt.messages
        )

        self.assertIn("anomaly_detection.high_spending_items[n].amount", daily_content)
        self.assertIn("stable_metrics.category_ratio_changes[n].diff_point", daily_content)
        self.assertNotIn("waste_detection.high_spending.items[n].amount", daily_content)
        self.assertNotIn("category_summary[n].diff_amount", daily_content)

        self.assertIn("waste_detection.high_spending.items[n].amount", weekly_content)
        self.assertIn("category_summary[n].diff_amount", weekly_content)

        self.assertIn("high_spending.items[n].amount", monthly_content)
        self.assertIn("category_deep[n].diff_amount", monthly_content)
        self.assertNotIn("waste_detection.high_spending.items[n].amount", monthly_content)
        self.assertNotIn("category_summary[n].diff_amount", monthly_content)

    def test_default_interpretation_chains_stop_at_cause_intervention_targets(self) -> None:
        """기본 해석 체인이 독립 action_result 없이 원인별 개입 타겟까지만 생성하는지 검증한다."""
        fake_llm = StructuredOutputFakeChatModel()
        chain_input = {
            "raw_json": "{}",
            "indicator_json": "{}",
            "user_profile_json": "{}",
        }

        for chain_builder in (
            build_spending_analysis_chain,
            build_weekly_spending_analysis_chain,
            build_monthly_spending_analysis_chain,
        ):
            result = chain_builder(llm=fake_llm).invoke(chain_input)

            self.assertIn("pattern_result", result)
            self.assertIn("problem_result", result)
            self.assertIn("cause_result", result)
            self.assertNotIn("action_result", result)
            self.assertIsInstance(result["cause_result"], CauseAnalysisResult)
            cause_result = result["cause_result"]
            assert isinstance(cause_result, CauseAnalysisResult)
            self.assertEqual(cause_result.intervention_targets, [])

    def test_compact_interpretation_chains_apply_sanitization(self) -> None:
        """balanced/unified 해석 체인도 원본 JSON으로 반증되는 결과를 후처리하는지 검증한다."""
        raw_json = """
        {
          "weekly_summary": {"this_week_total": 1000},
          "category_summary": [],
          "repeat_patterns": {
            "top_merchants": [
              {"merchant": "11번가", "visit_count": 1, "total_amount": 1000, "main_category": "쇼핑"}
            ]
          },
          "waste_detection": {"high_spending": {"items": []}}
        }
        """
        chain_input = {
            "raw_json": raw_json,
            "indicator_json": "{}",
            "user_profile_json": "{}",
        }
        fake_llm = SanitizingFakeChatModel()

        for chain_builder in (
            build_balanced_spending_analysis_chain,
            build_unified_spending_analysis_chain,
        ):
            result = chain_builder(llm=fake_llm).invoke(chain_input)
            pattern_result = result["pattern_result"]
            assert isinstance(pattern_result, PatternAnalysisResult)
            self.assertEqual(pattern_result.repeated_consumption, [])

    def test_cause_prompts_create_intervention_targets_instead_of_final_actions(self) -> None:
        """원인 해석 프롬프트가 최종 행동 대신 RAG 검색 타겟 후보를 만들도록 안내하는지 검증한다."""
        cause_input = {
            "raw_json": "{}",
            "indicator_json": "{}",
            "user_profile_json": "{}",
            "pattern_text": "{}",
            "problem_text": "{}",
        }

        daily_cause = build_consumption_cause_prompt().invoke(cause_input)
        weekly_cause = build_weekly_consumption_cause_prompt().invoke(cause_input)
        monthly_cause = build_monthly_consumption_cause_prompt().invoke(cause_input)

        cause_content = "\n".join(
            str(message.content)
            for message in [
                *daily_cause.messages,
                *weekly_cause.messages,
                *monthly_cause.messages,
            ]
        )

        self.assertIn("intervention_targets", cause_content)
        self.assertIn("RAG 검색 타겟", cause_content)
        self.assertIn("최종 행동 지시", cause_content)
        self.assertIn("action_result를 별도 생성하지 않는다", cause_content)
        self.assertIn("one_off_high_spending_causes", cause_content)
        self.assertIn("fixed_cost_timing_causes", cause_content)
        self.assertIn("period_concentration_causes", cause_content)
        self.assertIn("단일 고액", cause_content)
        self.assertIn("고정비 납부 타이밍", cause_content)
        self.assertIn("요일·기간 집중", cause_content)
        self.assertIn("고정비 항목은 one_off_high_spending_causes에 중복", cause_content)
        self.assertIn("items[n].amount", cause_content)
        self.assertIn("최대 소비 요일의 실제 금액", cause_content)

    def test_sanitize_spending_analysis_payload_removes_unsupported_weekly_findings(
        self,
    ) -> None:
        """주간 원본 JSON으로 반증되는 반복·문제·원인 오분류를 제거하고 개입 타겟을 정리한다."""
        raw_json = """
        {
          "weekly_summary": {"this_week_total": 1078800},
          "category_summary": [
            {"category": "교육", "total_amount": 604400, "prev_week_amount": 36500, "diff_amount": 567900, "diff_rate_percent": 1555.8904},
            {"category": "납부", "total_amount": 189100, "prev_week_amount": 72800, "diff_amount": 116300, "diff_rate_percent": 159.7527},
            {"category": "식비", "total_amount": 107200, "prev_week_amount": 256100, "diff_amount": -148900, "diff_rate_percent": -58.1414},
            {"category": "쇼핑", "total_amount": 84600, "prev_week_amount": 951200, "diff_amount": -866600, "diff_rate_percent": -91.106},
            {"category": "여가", "total_amount": 54100, "prev_week_amount": 0, "diff_amount": 54100, "diff_rate_percent": 0.0, "ratio_percent": 5.0148}
          ],
          "weekday_pattern": {
            "peak_weekday": "화",
            "weekday_breakdown": [
              {"weekday": "화", "weekday_num": 1, "total_amount": 742000, "transaction_count": 6}
            ]
          },
          "repeat_patterns": {
            "top_merchants": [
              {"merchant": "11번가", "visit_count": 1, "total_amount": 41500, "main_category": "쇼핑"},
              {"merchant": "넷플릭스", "visit_count": 1, "total_amount": 30400, "main_category": "여가"}
            ],
            "consecutive_merchants": [],
            "delivery": {"count": 1, "total_amount": 84400}
          },
          "waste_detection": {
            "micro_spending": {"count": 2, "total_amount": 18600},
            "high_spending": {
              "items": [
                {"merchant": "온라인강의", "amount": 566000, "category": "교육"},
                {"merchant": "아파트관리비", "amount": 189100, "category": "납부"}
              ]
            }
          },
          "saving_potential": {"delivery_save_per_skip": 84400},
          "weekly_metrics": {
            "weekday_spending_ratio_percent": 80.2836,
            "special_metrics": {"weekday_concentration_ratio_percent": 68.7801}
          }
        }
        """
        pattern_result = PatternAnalysisResult(
            repeated_consumption=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="11번가",
                    detail="1회 방문, 총 41500원 소비",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="repeat_patterns.top_merchants[0]",
                            supporting_value="11번가 1회",
                            reason="반복적으로 소비한 가맹점",
                        )
                    ],
                )
            ],
            impulse_patterns=[
                SpendingFinding(
                    subcategory="여가",
                    title="넷플릭스",
                    detail="30400원 소비, 여가 관련 충동 소비",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="repeat_patterns.top_merchants[1]",
                            supporting_value="넷플릭스 1회",
                            reason="여가 관련 소비가 확인됨",
                        )
                    ],
                )
            ],
            contextual_patterns=[
                SpendingFinding(
                    subcategory="요일 소비 패턴",
                    title="화요일 소비 집중",
                    detail="이번 주 소비는 화요일에 742000원으로 가장 많았으며, 전체 소비의 68.78%가 주중에 발생했습니다.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekday_pattern.peak_weekday",
                            supporting_value="화",
                            reason="화요일에 소비가 집중됨.",
                        )
                    ],
                )
            ],
        )
        problem_result = ProblemAnalysisResult(
            variable_cost_issues=[
                SpendingFinding(
                    subcategory="식비",
                    title="식비 감소",
                    detail="식비 지출 감소",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[2]",
                            supporting_value="식비 -148900",
                            reason="전주 대비 식비 지출이 크게 감소함",
                        )
                    ],
                )
            ],
            short_term_problem_spending=[
                SpendingFinding(
                    subcategory="여가",
                    title="여가 지출 증가",
                    detail="여가 지출이 54100원으로 증가함.",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[4]",
                            supporting_value="여가 54100",
                            reason="여가 지출이 새롭게 발생하여 단기 소비에 영향을 줄 수 있음.",
                        )
                    ],
                )
            ],
            long_term_problem_spending=[
                SpendingFinding(
                    subcategory="교육",
                    title="교육 지출 증가",
                    detail="교육 지출이 지속적으로 증가",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[0]",
                            supporting_value="교육 604400",
                            reason="교육 카테고리에서 지출이 지속적으로 증가함",
                        )
                    ],
                )
            ],
        )
        cause_result = CauseAnalysisResult(
            habitual_causes=[
                SpendingFinding(
                    subcategory="식비",
                    title="배달의민족",
                    detail="배달 음식 소비가 높음",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="repeat_patterns.delivery.total_amount",
                            supporting_value="84400",
                            reason="배달 음식 소비가 높음",
                        )
                    ],
                )
            ],
            reward_causes=[
                SpendingFinding(
                    subcategory="교육",
                    title="온라인강의",
                    detail="고액 결제 발생",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[0]",
                            supporting_value="566000",
                            reason="교육 카테고리 고액 결제",
                        )
                    ],
                )
            ],
            stress_causes=[
                SpendingFinding(
                    subcategory="납부",
                    title="아파트관리비",
                    detail="고정비 지출 증가",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[1]",
                            supporting_value="189100",
                            reason="납부 지출 증가",
                        )
                    ],
                )
            ],
            intervention_targets=[
                InterventionTarget(
                    target_type="high_spending",
                    title="온라인강의",
                    linked_cause="reward_causes",
                    target_json_path="waste_detection.high_spending.items[0]",
                    reason="고액 결제 항목으로 확인됨.",
                ),
                InterventionTarget(
                    target_type="fixed_cost",
                    title="아파트관리비",
                    linked_cause="stress_causes",
                    target_json_path="category_summary[1]",
                    reason="전주 대비 납부 지출이 증가함.",
                ),
                InterventionTarget(
                    target_type="habitual_spending",
                    title="배달의민족",
                    linked_cause="habitual_causes",
                    target_json_path="repeat_patterns.delivery.total_amount",
                    reason="배달 음식 소비가 높음.",
                ),
                InterventionTarget(
                    target_type="impulse_spending",
                    title="넷플릭스",
                    linked_cause="habitual_causes",
                    target_json_path="repeat_patterns.top_merchants[1]",
                    reason="여가 관련 소비가 확인됨.",
                ),
                InterventionTarget(
                    target_type="education_expense",
                    title="교육비 급증",
                    linked_cause="long_term_problem_spending",
                    target_json_path="category_summary[0]",
                    reason="교육 관련 지출의 지속적인 증가가 장기적으로 예산에 부담을 줄 수 있음.",
                    query_hint="교육비 지출 내역 검토",
                ),
                InterventionTarget(
                    target_type="payment_increase",
                    title="납부 지출 증가",
                    linked_cause="saving_blockers",
                    target_json_path="category_summary[1]",
                    reason="납부 항목의 지출이 증가하여 고정비 부담이 커질 수 있음.",
                    query_hint="납부 항목 세부 내역 확인",
                ),
                InterventionTarget(
                    target_type="food_expense",
                    title="식비 변동성",
                    linked_cause="variable_cost_issues",
                    target_json_path="category_summary[2]",
                    reason="식비의 급격한 감소가 가정의 식사 패턴에 영향을 미칠 수 있음.",
                    query_hint="식비 지출 패턴 분석",
                ),
                InterventionTarget(
                    target_type="shopping_expense",
                    title="쇼핑 변동성",
                    linked_cause="variable_cost_issues",
                    target_json_path="category_summary[3]",
                    reason="쇼핑 지출의 급격한 감소가 소비 패턴에 변화가 생길 수 있음.",
                    query_hint="쇼핑 지출 내역 검토",
                ),
                InterventionTarget(
                    target_type="leisure_expense",
                    title="여가 지출 증가",
                    linked_cause="short_term_problem_spending",
                    target_json_path="category_summary[4]",
                    reason="여가 지출이 새롭게 발생하여 단기적으로 소비 패턴에 영향을 미칠 수 있음.",
                    query_hint="여가 지출 내역 확인",
                ),
            ],
        )

        sanitized = sanitize_spending_analysis_payload(
            {
                "raw_json": raw_json,
                "indicator_json": "{}",
                "pattern_result": pattern_result,
                "problem_result": problem_result,
                "cause_result": cause_result,
            }
        )

        sanitized_pattern = sanitized["pattern_result"]
        sanitized_problem = sanitized["problem_result"]
        sanitized_cause = sanitized["cause_result"]
        assert isinstance(sanitized_pattern, PatternAnalysisResult)
        assert isinstance(sanitized_problem, ProblemAnalysisResult)
        assert isinstance(sanitized_cause, CauseAnalysisResult)

        self.assertEqual(sanitized_pattern.repeated_consumption, [])
        self.assertEqual(sanitized_pattern.impulse_patterns, [])
        self.assertEqual(len(sanitized_pattern.contextual_patterns), 1)
        self.assertIn("화요일 742000원", sanitized_pattern.contextual_patterns[0].detail)
        self.assertIn("화요일 비중 68.78%", sanitized_pattern.contextual_patterns[0].detail)
        self.assertIn("주중 소비 비중 80.28%", sanitized_pattern.contextual_patterns[0].detail)
        self.assertNotIn("68.78%가 주중", sanitized_pattern.contextual_patterns[0].detail)
        self.assertEqual(sanitized_problem.variable_cost_issues, [])
        self.assertEqual(sanitized_problem.short_term_problem_spending, [])
        self.assertEqual(sanitized_problem.long_term_problem_spending, [])
        self.assertEqual(sanitized_cause.habitual_causes, [])
        self.assertEqual(sanitized_cause.reward_causes, [])
        self.assertEqual(sanitized_cause.stress_causes, [])

        target_titles = [target.title for target in sanitized_cause.intervention_targets]
        self.assertIn("온라인강의 교육 고액 단일 결제 점검 타겟", target_titles)
        self.assertIn("아파트관리비 고정비 점검 타겟", target_titles)
        self.assertIn("배달 단일 결제 점검 타겟", target_titles)
        self.assertNotIn("넷플릭스", target_titles)
        self.assertNotIn("식비 변동성", target_titles)
        self.assertNotIn("쇼핑 변동성", target_titles)
        self.assertNotIn("여가 지출 증가", target_titles)
        self.assertTrue(
            all(
                target.linked_cause
                not in {
                    "reward_causes",
                    "stress_causes",
                    "habitual_causes",
                    "long_term_problem_spending",
                    "saving_blockers",
                    "variable_cost_issues",
                    "short_term_problem_spending",
                }
                for target in sanitized_cause.intervention_targets
            )
        )
        self.assertTrue(
            all(
                not re.fullmatch(r"category_summary\[\d+\]", target.target_json_path)
                for target in sanitized_cause.intervention_targets
            )
        )

    def test_sanitize_spending_analysis_payload_preserves_supported_weekly_causes(
        self,
    ) -> None:
        """주간 후처리가 원인 해석을 개입 타겟만 남기는 빈 결과로 축소하지 않는지 검증한다."""
        raw_json = """
        {
          "weekly_summary": {"this_week_total": 1078800},
          "category_summary": [
            {"category": "식비", "total_amount": 160000, "prev_week_amount": 90000, "diff_amount": 70000, "diff_rate_percent": 77.7778}
          ],
          "repeat_patterns": {
            "delivery": {"count": 2, "total_amount": 84400},
            "top_merchants": []
          },
          "waste_detection": {
            "micro_spending": {"count": 4, "total_amount": 18600},
            "high_spending": {"items": []}
          }
        }
        """
        cause_result = CauseAnalysisResult(
            convenience_causes=[
                SpendingFinding(
                    subcategory="식비",
                    title="배달 결제 편의성",
                    detail="배달 결제 2회가 확인되어 편의성 기반 소비 후보로 해석합니다.",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="repeat_patterns.delivery.total_amount",
                            supporting_value="84400",
                            reason="배달 결제 금액",
                        )
                    ],
                )
            ],
            small_accumulation_causes=[
                SpendingFinding(
                    subcategory="소액",
                    title="소액 결제 누적",
                    detail="소액 결제 4회가 확인되어 누적형 소비 후보로 해석합니다.",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.micro_spending.count",
                            supporting_value="4",
                            reason="소액 결제 건수",
                        )
                    ],
                )
            ],
        )

        sanitized = sanitize_spending_analysis_payload(
            {
                "raw_json": raw_json,
                "indicator_json": "{}",
                "pattern_result": PatternAnalysisResult(),
                "problem_result": ProblemAnalysisResult(),
                "cause_result": cause_result,
            }
        )

        sanitized_cause = sanitized["cause_result"]
        assert isinstance(sanitized_cause, CauseAnalysisResult)
        self.assertEqual(len(sanitized_cause.convenience_causes), 1)
        self.assertEqual(len(sanitized_cause.small_accumulation_causes), 1)
        self.assertTrue(sanitized_cause.intervention_targets)

    def test_sanitize_spending_analysis_payload_reanchors_high_spending_evidence(
        self,
    ) -> None:
        """고액 결제·카테고리 문제 근거를 실제 판단 필드로 보정하고 핵심 개입 타겟을 보강한다."""
        raw_json = """
        {
          "weekly_summary": {"this_week_total": 1671800, "amount_diff": 886400},
          "category_summary": [
            {"category": "쇼핑", "total_amount": 1173400, "prev_week_amount": 59200, "diff_amount": 1114200, "diff_rate_percent": 1882.0946},
            {"category": "납부", "total_amount": 398400, "prev_week_amount": 656600, "diff_amount": -258200, "diff_rate_percent": -39.3238},
            {"category": "여가", "total_amount": 16200, "prev_week_amount": 41200, "diff_amount": -25000, "diff_rate_percent": -60.6796}
          ],
          "weekly_comparisons": {
            "recent_4week_average": {"amount_diff": 1231775.0}
          },
          "weekly_metrics": {"weekend_spending_ratio_percent": 95.3284},
          "waste_detection": {
            "late_night": {"total_amount": 1227200},
            "high_spending": {
              "items": [
                {"merchant": "하이마트", "amount": 1164600, "category": "쇼핑"},
                {"merchant": "도시가스", "amount": 278400, "category": "납부"},
                {"merchant": "유선방송요금", "amount": 120000, "category": "납부"}
              ]
            }
          }
        }
        """
        pattern_result = PatternAnalysisResult(
            overspending_windows=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="고액 소비",
                    detail="하이마트에서 1164600원 소비",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_summary.this_week_total",
                            supporting_value="1671800",
                            reason="이번 주 총 지출액",
                        )
                    ],
                ),
                SpendingFinding(
                    subcategory="납부",
                    title="고액 소비",
                    detail="도시가스에서 278400원 소비",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_summary.this_week_total",
                            supporting_value="1671800",
                            reason="이번 주 총 지출액",
                        )
                    ],
                ),
            ]
        )
        problem_result = ProblemAnalysisResult(
            money_leaks=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="쇼핑 지출 급증",
                    detail="이번 주 쇼핑 지출이 1173400원으로 전주 대비 1114200원 증가함.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_summary.amount_diff",
                            supporting_value="886400",
                            reason="전주 대비 지출 증감액",
                        )
                    ],
                )
            ],
            long_term_problem_spending=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="지속적인 쇼핑 지출 증가",
                    detail="쇼핑 지출이 최근 4주 평균 대비 증가하여 장기적인 소비 패턴에 문제가 있음.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_comparisons.recent_4week_average.amount_diff",
                            supporting_value="1231775.0",
                            reason="최근 4주 평균 대비 지출 증감액",
                        )
                    ],
                )
            ],
            fixed_cost_issues=[
                SpendingFinding(
                    subcategory="납부",
                    title="납부 지출 감소",
                    detail="납부 카테고리에서 398400원 지출이 발생했으나 전주 대비 258200원 감소함.",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[1].diff_amount",
                            supporting_value="-258200",
                            reason="납부 지출이 감소하였음.",
                        )
                    ],
                )
            ],
            variable_cost_issues=[
                SpendingFinding(
                    subcategory="여가",
                    title="여가 지출 감소",
                    detail="여가 카테고리에서 16200원 지출이 발생했으나 전주 대비 25000원 감소함.",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[2].diff_amount",
                            supporting_value="-25000",
                            reason="여가 지출이 감소하였음.",
                        )
                    ],
                )
            ],
        )
        cause_result = CauseAnalysisResult(
            one_off_high_spending_causes=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="하이마트 단일 고액 결제",
                    detail="하이마트 1164600원 결제가 이번 주 쇼핑 증가를 설명하는 단일 이벤트 후보입니다.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[0].amount",
                            supporting_value="1164600",
                            reason="하이마트 고액 결제 금액",
                        )
                    ],
                )
            ],
            fixed_cost_timing_causes=[
                SpendingFinding(
                    subcategory="납부",
                    title="주간 고정비 납부 집중",
                    detail="도시가스와 유선방송요금 납부가 같은 주에 발생해 총액을 밀어 올린 타이밍 후보입니다.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[1].amount",
                            supporting_value="278400",
                            reason="도시가스 고액 납부 금액",
                        ),
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[2].amount",
                            supporting_value="120000",
                            reason="유선방송요금 고액 납부 금액",
                        ),
                    ],
                )
            ],
            period_concentration_causes=[
                SpendingFinding(
                    subcategory="요일 패턴",
                    title="일요일 소비 집중",
                    detail="일요일 1572400원 결제가 발생해 이번 주 지출이 특정 요일에 집중된 후보입니다.",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
                            supporting_value="94.0543",
                            reason="최대 소비 요일 집중도",
                        )
                    ],
                )
            ],
            stress_causes=[
                SpendingFinding(
                    subcategory="여가",
                    title="여가 지출 감소",
                    detail="여가 지출이 16200원으로 전주 대비 25000원 감소",
                    confidence="medium",
                    evidences=[
                        EvidenceItem(
                            json_path="category_summary[2].diff_amount",
                            supporting_value="-25000",
                            reason="여가 지출 감소가 스트레스 해소에 부정적 영향을 미칠 수 있음.",
                        )
                    ],
                )
            ],
            intervention_targets=[
                InterventionTarget(
                    target_type="fixed_cost_review",
                    title="도시가스 고정비 점검 타겟",
                    linked_cause="납부 카테고리에서 도시가스 278400원 결제가 발생함",
                    target_json_path="waste_detection.high_spending.items[1].amount",
                    reason="고정비 성격의 도시가스 결제가 주간 지출에 반영됨",
                    query_hint="도시가스 고정비 점검",
                ),
                InterventionTarget(
                    target_type="shopping_increase",
                    title="쇼핑 지출 증가 점검",
                    linked_cause="쇼핑 지출 장기 증가",
                    target_json_path="weekly_comparisons.recent_4week_average.amount_diff",
                    reason="최근 4주 평균 대비 쇼핑 지출이 급증하였음.",
                    query_hint="쇼핑 카테고리 지출 내역 분석",
                ),
                InterventionTarget(
                    target_type="weekend_spending",
                    title="주말 소비 패턴 점검",
                    linked_cause="주말 소비 비중",
                    target_json_path="weekly_metrics.weekend_spending_ratio_percent",
                    reason="주말 소비가 전체 소비의 대부분을 차지함.",
                    query_hint="주말 소비 내역 분석",
                ),
                InterventionTarget(
                    target_type="late_night_spending",
                    title="야간 소비 점검",
                    linked_cause="야간 소비 증가",
                    target_json_path="waste_detection.late_night.total_amount",
                    reason="야간 소비가 전체 소비에서 큰 비중을 차지함.",
                    query_hint="야간 소비 내역 검토",
                ),
            ],
        )

        sanitized = sanitize_spending_analysis_payload(
            {
                "raw_json": raw_json,
                "indicator_json": "{}",
                "pattern_result": pattern_result,
                "problem_result": problem_result,
                "cause_result": cause_result,
            }
        )

        sanitized_pattern = sanitized["pattern_result"]
        sanitized_problem = sanitized["problem_result"]
        sanitized_cause = sanitized["cause_result"]
        assert isinstance(sanitized_pattern, PatternAnalysisResult)
        assert isinstance(sanitized_problem, ProblemAnalysisResult)
        assert isinstance(sanitized_cause, CauseAnalysisResult)

        self.assertEqual(
            sanitized_pattern.overspending_windows[0].evidences[0].json_path,
            "waste_detection.high_spending.items[0].amount",
        )
        self.assertEqual(
            sanitized_pattern.overspending_windows[1].evidences[0].json_path,
            "waste_detection.high_spending.items[1].amount",
        )
        self.assertEqual(
            sanitized_problem.money_leaks[0].evidences[0].json_path,
            "waste_detection.high_spending.items[0].amount",
        )
        self.assertEqual(
            sanitized_problem.saving_blockers[0].evidences[0].json_path,
            "category_summary[0].diff_amount",
        )
        self.assertEqual(sanitized_problem.long_term_problem_spending, [])
        self.assertEqual(len(sanitized_problem.fixed_cost_issues), 1)
        self.assertEqual(
            [evidence.json_path for evidence in sanitized_problem.fixed_cost_issues[0].evidences],
            [
                "waste_detection.high_spending.items[1].amount",
                "waste_detection.high_spending.items[2].amount",
            ],
        )
        self.assertEqual(sanitized_problem.variable_cost_issues, [])
        self.assertEqual(sanitized_problem.short_term_problem_spending, [])
        self.assertEqual(sanitized_cause.stress_causes, [])
        self.assertEqual(len(sanitized_cause.one_off_high_spending_causes), 1)
        self.assertEqual(len(sanitized_cause.fixed_cost_timing_causes), 1)
        self.assertEqual(len(sanitized_cause.period_concentration_causes), 1)

        target_titles = [target.title for target in sanitized_cause.intervention_targets]
        self.assertIn("하이마트 쇼핑 고액 단일 결제 점검 타겟", target_titles)
        self.assertIn("도시가스 고정비 점검 타겟", target_titles)
        self.assertIn("유선방송요금 고정비 점검 타겟", target_titles)
        self.assertNotIn("쇼핑 지출 증가 점검", target_titles)
        self.assertNotIn("주말 소비 패턴 점검", target_titles)
        self.assertNotIn("야간 소비 점검", target_titles)
        self.assertEqual(len(sanitized_cause.intervention_targets), 3)
        high_target = next(
            target
            for target in sanitized_cause.intervention_targets
            if target.title == "하이마트 쇼핑 고액 단일 결제 점검 타겟"
        )
        self.assertEqual(
            high_target.target_json_path,
            "waste_detection.high_spending.items[0].amount",
        )
        self.assertEqual(high_target.query_hint, "가전 쇼핑 고액 결제 예산 점검")

    def test_sanitize_weekly_cause_result_dedupes_and_contextualizes_causes(
        self,
    ) -> None:
        """주간 원인 결과에서 고정비 중복을 제거하고 요일·납부 맥락을 구체화하는지 검증한다."""
        raw_json = """
        {
          "weekly_summary": {
            "this_week_total": 1671800,
            "max_day_date": "2026-04-05",
            "max_day_amount": 1572400
          },
          "category_summary": [
            {"category": "쇼핑", "total_amount": 1173400, "prev_week_amount": 59200, "diff_amount": 1114200, "diff_rate_percent": 1882.0946},
            {"category": "납부", "total_amount": 398400, "prev_week_amount": 656600, "diff_amount": -258200, "diff_rate_percent": -39.3238}
          ],
          "weekday_pattern": {
            "peak_weekday": "일",
            "weekday_breakdown": [
              {"weekday": "수", "weekday_num": 2, "total_amount": 71300, "transaction_count": 2},
              {"weekday": "목", "weekday_num": 3, "total_amount": 6800, "transaction_count": 1},
              {"weekday": "토", "weekday_num": 5, "total_amount": 21300, "transaction_count": 2},
              {"weekday": "일", "weekday_num": 6, "total_amount": 1572400, "transaction_count": 4}
            ]
          },
          "weekly_metrics": {
            "weekend_spending_ratio_percent": 95.3284,
            "special_metrics": {"weekday_concentration_ratio_percent": 94.0543}
          },
          "waste_detection": {
            "high_spending": {
              "items": [
                {"used_at": "2026-04-05 21:24:57", "merchant": "하이마트", "amount": 1164600, "category": "쇼핑"},
                {"used_at": "2026-04-05 09:20:29", "merchant": "도시가스", "amount": 278400, "category": "납부"},
                {"used_at": "2026-04-05 11:21:05", "merchant": "유선방송요금", "amount": 120000, "category": "납부"}
              ]
            }
          }
        }
        """
        cause_result = CauseAnalysisResult(
            one_off_high_spending_causes=[
                SpendingFinding(
                    subcategory="쇼핑",
                    title="하이마트 고액 결제",
                    detail="하이마트에서 1164600원 결제",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[0]",
                            supporting_value="1164600",
                            reason="고액 결제",
                        )
                    ],
                ),
                SpendingFinding(
                    subcategory="납부",
                    title="도시가스 고액 결제",
                    detail="도시가스에서 278400원 결제",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[1]",
                            supporting_value="278400",
                            reason="고액 결제",
                        )
                    ],
                ),
                SpendingFinding(
                    subcategory="납부",
                    title="유선방송요금 고액 결제",
                    detail="유선방송요금에서 120000원 결제",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[2]",
                            supporting_value="120000",
                            reason="고액 결제",
                        )
                    ],
                ),
            ],
            fixed_cost_timing_causes=[
                SpendingFinding(
                    subcategory="납부",
                    title="도시가스 고액 결제",
                    detail="도시가스에서 278400원 결제",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[1]",
                            supporting_value="278400",
                            reason="고정비 지출이 증가하여 재정적 압박을 초래할 수 있음.",
                        )
                    ],
                ),
                SpendingFinding(
                    subcategory="납부",
                    title="유선방송요금 고액 결제",
                    detail="유선방송요금에서 120000원 결제",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="waste_detection.high_spending.items[2]",
                            supporting_value="120000",
                            reason="고정비 지출이 증가하여 재정적 압박을 초래할 수 있음.",
                        )
                    ],
                ),
            ],
            period_concentration_causes=[
                SpendingFinding(
                    subcategory="요일 소비 패턴",
                    title="주말 소비 집중",
                    detail="주말에 95.33%의 소비가 발생하여 주중 소비와 큰 차이를 보임",
                    confidence="high",
                    evidences=[
                        EvidenceItem(
                            json_path="weekly_metrics.weekend_spending_ratio_percent",
                            supporting_value="95.3284",
                            reason="주말 소비 비중",
                        )
                    ],
                )
            ],
            intervention_targets=[
                InterventionTarget(
                    target_type="spending_pattern",
                    title="일요일 소비 패턴 점검",
                    linked_cause="period_concentration_causes",
                    target_json_path="weekday_pattern.peak_weekday",
                    reason="일요일이 피크 요일로 소비가 집중됨.",
                )
            ],
        )

        sanitized = sanitize_spending_analysis_payload(
            {
                "raw_json": raw_json,
                "indicator_json": "{}",
                "cause_result": cause_result,
            }
        )

        sanitized_cause = sanitized["cause_result"]
        assert isinstance(sanitized_cause, CauseAnalysisResult)

        self.assertEqual(len(sanitized_cause.one_off_high_spending_causes), 1)
        self.assertEqual(
            sanitized_cause.one_off_high_spending_causes[0].title, "하이마트 고액 결제"
        )
        self.assertEqual(
            sanitized_cause.one_off_high_spending_causes[0].evidences[0].json_path,
            "waste_detection.high_spending.items[0].amount",
        )
        self.assertNotIn(
            "납부",
            {finding.subcategory for finding in sanitized_cause.one_off_high_spending_causes},
        )

        self.assertEqual(len(sanitized_cause.fixed_cost_timing_causes), 1)
        fixed_cost_cause = sanitized_cause.fixed_cost_timing_causes[0]
        self.assertIn("일요일", fixed_cost_cause.detail)
        self.assertIn("도시가스 278400원", fixed_cost_cause.detail)
        self.assertIn("유선방송요금 120000원", fixed_cost_cause.detail)
        self.assertNotIn("재정적 압박", fixed_cost_cause.evidences[0].reason)
        self.assertEqual(
            [evidence.json_path for evidence in fixed_cost_cause.evidences],
            [
                "waste_detection.high_spending.items[1].amount",
                "waste_detection.high_spending.items[2].amount",
            ],
        )

        self.assertEqual(len(sanitized_cause.period_concentration_causes), 1)
        period_cause = sanitized_cause.period_concentration_causes[0]
        self.assertIn("일요일 1572400원", period_cause.detail)
        self.assertIn("요일 편중도 94.05%", period_cause.detail)
        self.assertIn("주말 소비 비중 95.33%", period_cause.detail)
        self.assertIn("하이마트 1164600원", period_cause.detail)
        self.assertIn("21:24", period_cause.detail)
        self.assertIn(
            "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
            [evidence.json_path for evidence in period_cause.evidences],
        )

        period_target = next(
            target
            for target in sanitized_cause.intervention_targets
            if target.target_type == "period_concentration_review"
        )
        self.assertEqual(period_target.title, "일요일 고액 결제 집중 점검 타겟")
        self.assertEqual(
            period_target.linked_cause,
            "일요일 단일 고액 결제 중심의 주말 편중",
        )
        self.assertEqual(
            period_target.target_json_path,
            "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
        )
        self.assertIn("하이마트 1164600원", period_target.reason)
        self.assertEqual(period_target.query_hint, "일요일 고액 결제 집중 점검")
        self.assertNotIn(
            "period_concentration_causes",
            [target.linked_cause for target in sanitized_cause.intervention_targets],
        )

    def test_period_analysis_prompts_name_extended_comparison_baselines(self) -> None:
        """일·주·월 해석 프롬프트가 새 비교 기준을 명시적으로 안내하는지 검증한다."""
        base_input = {
            "raw_json": "{}",
            "indicator_json": "{}",
            "user_profile_json": "{}",
        }

        daily_pattern = build_consumption_pattern_prompt().invoke(base_input)
        daily_problem = build_consumption_problem_prompt().invoke(base_input)
        weekly_pattern = build_weekly_consumption_pattern_prompt().invoke(base_input)
        weekly_problem = build_weekly_consumption_problem_prompt().invoke(base_input)
        monthly_pattern = build_monthly_consumption_pattern_prompt().invoke(base_input)
        monthly_problem = build_monthly_consumption_problem_prompt().invoke(base_input)

        daily_content = "\n".join(
            str(message.content) for message in [*daily_pattern.messages, *daily_problem.messages]
        )
        weekly_content = "\n".join(
            str(message.content) for message in [*weekly_pattern.messages, *weekly_problem.messages]
        )
        monthly_content = "\n".join(
            str(message.content)
            for message in [*monthly_pattern.messages, *monthly_problem.messages]
        )

        self.assertIn("지난주 같은 요일", daily_content)
        self.assertIn("최근 4주 같은 요일 평균", daily_content)
        self.assertIn("최근 4주 평균", weekly_content)
        self.assertIn("지난달 같은 주차", weekly_content)
        self.assertIn("최근 3개월 평균", monthly_content)

    def test_parse_user_spending_data_accepts_daily_analysis_json_shape(self) -> None:
        """일일 분석 서비스 JSON을 추가 변환 없이 해석 입력 모델로 읽을 수 있는지 검증한다."""
        past_frame = pd.read_csv("data/raw/csv/transactions_v1.csv", encoding="utf-8-sig")
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
            past_source_path="data/raw/csv/transactions_v1.csv",
            today_source_path="notebook/team02/data_pre/data_input_month.csv",
        )

        user_data = parse_user_spending_data(raw_result)
        analysis_input = make_spending_analysis_input(user_data)

        self.assertEqual(user_data.source_paths.past_source, "data/raw/csv/transactions_v1.csv")
        self.assertEqual(user_data.stable_metrics.today_total, 133044)
        self.assertEqual(
            user_data.payment_behavior_analysis.frictionless_spending.total_amount,
            1486,
        )
        self.assertIn("anomaly_detection.high_spending_items", analysis_input["indicator_json"])
        self.assertIn("daily_metrics", analysis_input["raw_json"])
        self.assertIn("daily_comparisons", analysis_input["raw_json"])
        self.assertIn("daily_metrics.late_night_ratio_percent", analysis_input["indicator_json"])
        self.assertIn("지난주 같은 요일 대비 지출 증감률", analysis_input["indicator_json"])
        self.assertIn("최근 4주 같은 요일 평균 대비 지출 증감률", analysis_input["indicator_json"])
        self.assertIn("충동소비 점수", analysis_input["indicator_json"])
        self.assertIn("payment_behavior_analysis", analysis_input["raw_json"])
        self.assertIn("마찰력 없는 지출 비중", analysis_input["indicator_json"])
        self.assertEqual(analysis_input["user_profile_json"], "{}")


if __name__ == "__main__":
    unittest.main()
