from __future__ import annotations

from catcher_llm.prompts.consumption_feedback.analysis import (
    build_consumption_action_prompt,
    build_consumption_cause_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
    build_consumption_unified_analysis_prompt,
)
from catcher_llm.prompts.consumption_feedback.feedback import (
    build_daily_feedback_prompt,
    build_monthly_feedback_prompt,
    build_weekly_feedback_prompt,
)
from catcher_llm.prompts.consumption_feedback.memory import (
    build_feedback_memory_rank_prompt,
    build_memory_summary_prompt,
)
from catcher_llm.prompts.consumption_feedback.monthly import (
    build_monthly_consumption_action_prompt,
    build_monthly_consumption_cause_prompt,
    build_monthly_consumption_pattern_prompt,
    build_monthly_consumption_problem_prompt,
)
from catcher_llm.prompts.consumption_feedback.weekly import (
    build_weekly_consumption_action_prompt,
    build_weekly_consumption_cause_prompt,
    build_weekly_consumption_pattern_prompt,
    build_weekly_consumption_problem_prompt,
)

__all__ = [
    "build_consumption_pattern_prompt",
    "build_consumption_problem_prompt",
    "build_consumption_cause_prompt",
    "build_consumption_action_prompt",
    "build_consumption_cause_action_prompt",
    "build_consumption_unified_analysis_prompt",
    "build_weekly_consumption_pattern_prompt",
    "build_weekly_consumption_problem_prompt",
    "build_weekly_consumption_cause_prompt",
    "build_weekly_consumption_action_prompt",
    "build_monthly_consumption_pattern_prompt",
    "build_monthly_consumption_problem_prompt",
    "build_monthly_consumption_cause_prompt",
    "build_monthly_consumption_action_prompt",
    "build_daily_feedback_prompt",
    "build_weekly_feedback_prompt",
    "build_monthly_feedback_prompt",
    "build_memory_summary_prompt",
    "build_feedback_memory_rank_prompt",
]
