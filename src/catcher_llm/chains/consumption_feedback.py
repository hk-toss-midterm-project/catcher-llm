from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback import (
    build_consumption_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
)
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    CauseAnalysisResult,
    PatternAnalysisResult,
    ProblemAnalysisResult,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    prepare_action_payload,
    prepare_cause_payload,
)


def _extract_raw_json(payload: dict[str, str]) -> str:
    """체인 병렬 실행 중 입력 페이로드의 원본 JSON 문자열을 전달한다."""
    return payload["raw_json"]


def _extract_indicator_json(payload: dict[str, str]) -> str:
    """체인 병렬 실행 중 입력 페이로드의 지표 JSON 문자열을 전달한다."""
    return payload["indicator_json"]


def build_consumption_pattern_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], PatternAnalysisResult]:
    """소비 패턴 탐지 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_consumption_pattern_prompt() | llm.with_structured_output(PatternAnalysisResult)  # type: ignore[return-value]


def build_consumption_problem_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ProblemAnalysisResult]:
    """문제 소비 식별 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_consumption_problem_prompt() | llm.with_structured_output(ProblemAnalysisResult)  # type: ignore[return-value]


def build_consumption_cause_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], CauseAnalysisResult]:
    """소비 원인 해석 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_consumption_cause_prompt() | llm.with_structured_output(CauseAnalysisResult)  # type: ignore[return-value]


def build_consumption_action_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ActionAnalysisResult]:
    """행동 개선 포인트 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_consumption_action_prompt() | llm.with_structured_output(ActionAnalysisResult)  # type: ignore[return-value]


def build_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """JSON 지표 기반 소비 패턴, 문제 소비, 원인, 행동 포인트 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    diagnosis_chain = RunnableParallel(
        raw_json=RunnableLambda(_extract_raw_json),
        indicator_json=RunnableLambda(_extract_indicator_json),
        pattern_result=build_consumption_pattern_chain(chat_model),
        problem_result=build_consumption_problem_chain(chat_model),
    )
    return (
        diagnosis_chain
        | RunnablePassthrough.assign(
            cause_result=RunnableLambda(prepare_cause_payload)
            | build_consumption_cause_chain(chat_model)
        )
        | RunnablePassthrough.assign(
            action_result=RunnableLambda(prepare_action_payload)
            | build_consumption_action_chain(chat_model)
        )
    )
