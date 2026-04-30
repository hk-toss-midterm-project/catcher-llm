from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback.feedback import (
    build_daily_feedback_prompt,
    build_monthly_feedback_prompt,
    build_weekly_feedback_prompt,
)
from catcher_llm.prompts.persona_prompt import PERSONAS
from catcher_llm.schemas.consumption_feedback.daily import DailyFeedbackResult
from catcher_llm.schemas.consumption_feedback.monthly import MonthlyFeedbackResult
from catcher_llm.schemas.consumption_feedback.weekly import WeeklyFeedbackResult


def build_daily_feedback_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
    persona_key: str | None = None,
) -> Runnable[dict[str, str], DailyFeedbackResult]:
    """일일 소비 분석 JSON, 해석 JSON, RAG 근거로 최종 피드백을 생성하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    persona_override = (
        PERSONAS[persona_key]["prompt"] if persona_key and persona_key in PERSONAS else None
    )
    return build_daily_feedback_prompt(
        persona_override=persona_override
    ) | chat_model.with_structured_output(DailyFeedbackResult)  # type: ignore[return-value]


def build_weekly_feedback_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
    persona_key: str | None = None,
) -> Runnable[dict[str, str], WeeklyFeedbackResult]:
    """주간 소비 분석 JSON, 해석 JSON, RAG 근거로 최종 피드백을 생성하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    persona_override = (
        PERSONAS[persona_key]["prompt"] if persona_key and persona_key in PERSONAS else None
    )
    return build_weekly_feedback_prompt(
        persona_override=persona_override
    ) | chat_model.with_structured_output(WeeklyFeedbackResult)  # type: ignore[return-value]


def build_monthly_feedback_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
    persona_key: str | None = None,
) -> Runnable[dict[str, str], MonthlyFeedbackResult]:
    """월간 소비 분석 JSON, 해석 JSON, RAG 근거로 최종 피드백을 생성하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    persona_override = (
        PERSONAS[persona_key]["prompt"] if persona_key and persona_key in PERSONAS else None
    )
    return build_monthly_feedback_prompt(
        persona_override=persona_override
    ) | chat_model.with_structured_output(MonthlyFeedbackResult)  # type: ignore[return-value]


__all__ = [
    "build_daily_feedback_chain",
    "build_weekly_feedback_chain",
    "build_monthly_feedback_chain",
]
