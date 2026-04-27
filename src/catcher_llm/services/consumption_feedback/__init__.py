from __future__ import annotations

from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    build_feedback_retrieval_queries,
    generate_daily_feedback,
    make_daily_feedback_input,
    retrieve_feedback_contexts,
    serialize_advice_contexts,
    serialize_interpretation_result,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_spending_indicators,
    load_user_spending_data,
    make_spending_analysis_input,
    parse_user_spending_data,
)
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.monthly_feedback import (
    build_monthly_feedback_retrieval_queries,
    extract_monthly_spending_indicators,
    generate_monthly_feedback,
    make_monthly_feedback_input,
    make_monthly_spending_analysis_input,
    parse_monthly_spending_data,
)
from catcher_llm.services.consumption_feedback.weekly_analysis import (
    build_weekly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.weekly_feedback import (
    build_weekly_feedback_retrieval_queries,
    extract_weekly_spending_indicators,
    generate_weekly_feedback,
    make_weekly_feedback_input,
    make_weekly_spending_analysis_input,
    parse_weekly_spending_data,
)

__all__ = [
    "build_daily_consumption_analysis_json",
    "build_feedback_retrieval_queries",
    "build_monthly_consumption_analysis_json",
    "build_monthly_feedback_retrieval_queries",
    "build_weekly_consumption_analysis_json",
    "build_weekly_feedback_retrieval_queries",
    "extract_monthly_spending_indicators",
    "extract_spending_indicators",
    "extract_weekly_spending_indicators",
    "generate_daily_feedback",
    "generate_monthly_feedback",
    "generate_weekly_feedback",
    "load_user_spending_data",
    "make_daily_feedback_input",
    "make_monthly_feedback_input",
    "make_monthly_spending_analysis_input",
    "make_spending_analysis_input",
    "make_weekly_feedback_input",
    "make_weekly_spending_analysis_input",
    "parse_monthly_spending_data",
    "parse_weekly_spending_data",
    "parse_user_spending_data",
    "retrieve_feedback_contexts",
    "serialize_advice_contexts",
    "serialize_interpretation_result",
]
