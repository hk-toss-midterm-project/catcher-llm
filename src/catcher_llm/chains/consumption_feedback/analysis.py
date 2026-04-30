from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback.analysis import (
    build_consumption_action_prompt,
    build_consumption_cause_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
    build_consumption_unified_analysis_prompt,
)
from catcher_llm.schemas.consumption_feedback.analysis_outputs import (
    ActionAnalysisResult,
    CauseActionAnalysisResult,
    CauseAnalysisResult,
    PatternAnalysisResult,
    ProblemAnalysisResult,
    SpendingAnalysisResult,
)


def _extract_raw_json(payload: dict[str, str]) -> str:
    """체인 병렬 실행 중 입력 페이로드의 원본 JSON 문자열을 전달한다."""
    return payload["raw_json"]


def _extract_indicator_json(payload: dict[str, str]) -> str:
    """체인 병렬 실행 중 입력 페이로드의 지표 JSON 문자열을 전달한다."""
    return payload["indicator_json"]


def _extract_user_profile_json(payload: dict[str, str]) -> str:
    """체인 병렬 실행 중 입력 페이로드의 사용자 프로필 JSON 문자열을 전달한다."""
    return payload["user_profile_json"]


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


def build_consumption_cause_action_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], CauseActionAnalysisResult]:
    """소비 원인과 행동 개선 포인트를 하나의 구조화 출력으로 생성하는 체인을 생성한다."""
    return build_consumption_cause_action_prompt() | llm.with_structured_output(
        CauseActionAnalysisResult
    )  # type: ignore[return-value]


def build_consumption_unified_analysis_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], SpendingAnalysisResult]:
    """일일 소비 해석 전체 결과를 하나의 구조화 출력으로 생성하는 체인을 생성한다."""
    return build_consumption_unified_analysis_prompt() | llm.with_structured_output(
        SpendingAnalysisResult
    )  # type: ignore[return-value]


def prepare_cause_payload(payload: dict[str, object]) -> dict[str, str]:
    """원인 해석 체인에 필요한 JSON 입력 페이로드를 생성한다."""
    pattern_result = payload["pattern_result"]
    problem_result = payload["problem_result"]
    if not isinstance(pattern_result, PatternAnalysisResult):
        raise TypeError("pattern_result must be PatternAnalysisResult")
    if not isinstance(problem_result, ProblemAnalysisResult):
        raise TypeError("problem_result must be ProblemAnalysisResult")
    return {
        "raw_json": str(payload["raw_json"]),
        "indicator_json": str(payload["indicator_json"]),
        "user_profile_json": str(payload["user_profile_json"]),
        "pattern_text": pattern_result.model_dump_json(indent=2),
        "problem_text": problem_result.model_dump_json(indent=2),
    }


def prepare_action_payload(payload: dict[str, object]) -> dict[str, str]:
    """행동 개선 체인에 필요한 JSON 입력 페이로드를 생성한다."""
    pattern_result = payload["pattern_result"]
    problem_result = payload["problem_result"]
    cause_result = payload["cause_result"]
    if not isinstance(pattern_result, PatternAnalysisResult):
        raise TypeError("pattern_result must be PatternAnalysisResult")
    if not isinstance(problem_result, ProblemAnalysisResult):
        raise TypeError("problem_result must be ProblemAnalysisResult")
    if not isinstance(cause_result, CauseAnalysisResult):
        raise TypeError("cause_result must be CauseAnalysisResult")
    return {
        "raw_json": str(payload["raw_json"]),
        "indicator_json": str(payload["indicator_json"]),
        "user_profile_json": str(payload["user_profile_json"]),
        "pattern_text": pattern_result.model_dump_json(indent=2),
        "problem_text": problem_result.model_dump_json(indent=2),
        "cause_text": cause_result.model_dump_json(indent=2),
    }


def prepare_cause_action_payload(payload: dict[str, object]) -> dict[str, str]:
    """원인·행동 통합 체인에 필요한 JSON 입력 페이로드를 생성한다."""
    pattern_result = payload["pattern_result"]
    problem_result = payload["problem_result"]
    if not isinstance(pattern_result, PatternAnalysisResult):
        raise TypeError("pattern_result must be PatternAnalysisResult")
    if not isinstance(problem_result, ProblemAnalysisResult):
        raise TypeError("problem_result must be ProblemAnalysisResult")
    return {
        "raw_json": str(payload["raw_json"]),
        "indicator_json": str(payload["indicator_json"]),
        "user_profile_json": str(payload["user_profile_json"]),
        "pattern_text": pattern_result.model_dump_json(indent=2),
        "problem_text": problem_result.model_dump_json(indent=2),
    }


def flatten_cause_action_payload(payload: dict[str, object]) -> dict[str, object]:
    """원인·행동 통합 출력 모델을 기존 해석 결과 키 구조로 펼친다."""
    cause_action_result = payload["cause_action_result"]
    if not isinstance(cause_action_result, CauseActionAnalysisResult):
        cause_action_result = CauseActionAnalysisResult.model_validate(cause_action_result)
    return {
        "raw_json": payload["raw_json"],
        "indicator_json": payload["indicator_json"],
        "user_profile_json": payload["user_profile_json"],
        "pattern_result": payload["pattern_result"],
        "problem_result": payload["problem_result"],
        "cause_result": cause_action_result.cause_result,
        "action_result": cause_action_result.action_result,
    }


def flatten_spending_analysis_result(result: object) -> dict[str, object]:
    """통합 해석 출력 모델을 기존 해석 결과 키 구조로 변환한다."""
    analysis_result = (
        result
        if isinstance(result, SpendingAnalysisResult)
        else SpendingAnalysisResult.model_validate(result)
    )
    return {
        "pattern_result": analysis_result.pattern_result,
        "problem_result": analysis_result.problem_result,
        "cause_result": analysis_result.cause_result,
        "action_result": analysis_result.action_result,
    }


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
        user_profile_json=RunnableLambda(_extract_user_profile_json),
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


def build_balanced_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """패턴·문제는 분리하고 원인·행동은 통합한 일일 소비 해석 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    diagnosis_chain = RunnableParallel(
        raw_json=RunnableLambda(_extract_raw_json),
        indicator_json=RunnableLambda(_extract_indicator_json),
        user_profile_json=RunnableLambda(_extract_user_profile_json),
        pattern_result=build_consumption_pattern_chain(chat_model),
        problem_result=build_consumption_problem_chain(chat_model),
    )
    return (
        diagnosis_chain
        | RunnablePassthrough.assign(
            cause_action_result=RunnableLambda(prepare_cause_action_payload)
            | build_consumption_cause_action_chain(chat_model)
        )
        | RunnableLambda(flatten_cause_action_payload)
    )


def build_unified_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """패턴·문제·원인·행동을 한 번의 구조화 출력으로 생성하는 일일 소비 해석 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_consumption_unified_analysis_chain(chat_model) | RunnableLambda(
        flatten_spending_analysis_result
    )


__all__ = [
    "build_consumption_pattern_chain",
    "build_consumption_problem_chain",
    "build_consumption_cause_chain",
    "build_consumption_action_chain",
    "build_consumption_cause_action_chain",
    "build_consumption_unified_analysis_chain",
    "prepare_cause_payload",
    "prepare_action_payload",
    "prepare_cause_action_payload",
    "flatten_cause_action_payload",
    "flatten_spending_analysis_result",
    "build_spending_analysis_chain",
    "build_balanced_spending_analysis_chain",
    "build_unified_spending_analysis_chain",
]
