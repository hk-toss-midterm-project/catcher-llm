from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from catcher_llm.chains.consumption_feedback.analysis import (
    _extract_indicator_json,
    _extract_raw_json,
    _extract_user_profile_json,
    prepare_cause_payload,
)
from catcher_llm.chains.consumption_feedback.sanitization import sanitize_spending_analysis_payload
from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback.monthly import (
    build_monthly_consumption_action_prompt,
    build_monthly_consumption_cause_prompt,
    build_monthly_consumption_pattern_prompt,
    build_monthly_consumption_problem_prompt,
)
from catcher_llm.schemas.consumption_feedback.analysis_outputs import (
    ActionAnalysisResult,
    CauseAnalysisResult,
    PatternAnalysisResult,
    ProblemAnalysisResult,
)


def build_monthly_consumption_pattern_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], PatternAnalysisResult]:
    """월간 소비 패턴 탐지 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_monthly_consumption_pattern_prompt() | llm.with_structured_output(
        PatternAnalysisResult
    )  # type: ignore[return-value]


def build_monthly_consumption_problem_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ProblemAnalysisResult]:
    """월간 문제 소비 식별 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_monthly_consumption_problem_prompt() | llm.with_structured_output(
        ProblemAnalysisResult
    )  # type: ignore[return-value]


def build_monthly_consumption_cause_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], CauseAnalysisResult]:
    """월간 소비 원인 해석 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_monthly_consumption_cause_prompt() | llm.with_structured_output(
        CauseAnalysisResult
    )  # type: ignore[return-value]


def build_monthly_consumption_action_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ActionAnalysisResult]:
    """월간 개선 후보·개입 타겟 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_monthly_consumption_action_prompt() | llm.with_structured_output(
        ActionAnalysisResult
    )  # type: ignore[return-value]


def build_monthly_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """월간 JSON 지표 기반 소비 패턴, 문제 소비, 원인·개입 타겟 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    diagnosis_chain = RunnableParallel(
        raw_json=RunnableLambda(_extract_raw_json),
        indicator_json=RunnableLambda(_extract_indicator_json),
        user_profile_json=RunnableLambda(_extract_user_profile_json),
        pattern_result=build_monthly_consumption_pattern_chain(chat_model),
        problem_result=build_monthly_consumption_problem_chain(chat_model),
    )
    return (
        diagnosis_chain
        | RunnablePassthrough.assign(
            cause_result=RunnableLambda(prepare_cause_payload)
            | build_monthly_consumption_cause_chain(chat_model)
        )
        | RunnableLambda(sanitize_spending_analysis_payload)
    )


__all__ = [
    "build_monthly_consumption_pattern_chain",
    "build_monthly_consumption_problem_chain",
    "build_monthly_consumption_cause_chain",
    "build_monthly_consumption_action_chain",
    "build_monthly_spending_analysis_chain",
]
