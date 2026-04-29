from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback import (
    build_consumption_action_prompt,
    build_consumption_cause_prompt,
    build_consumption_pattern_prompt,
    build_consumption_problem_prompt,
    build_daily_feedback_prompt,
    build_memory_summary_prompt,
    build_monthly_consumption_action_prompt,
    build_monthly_consumption_cause_prompt,
    build_monthly_consumption_pattern_prompt,
    build_monthly_consumption_problem_prompt,
    build_monthly_feedback_prompt,
    build_weekly_consumption_action_prompt,
    build_weekly_consumption_cause_prompt,
    build_weekly_consumption_pattern_prompt,
    build_weekly_consumption_problem_prompt,
    build_weekly_feedback_prompt,
)
from catcher_llm.prompts.persona_prompt import PERSONAS
from catcher_llm.schemas.consumption_feedback import (
    ActionAnalysisResult,
    CauseAnalysisResult,
    DailyFeedbackResult,
    MonthlyFeedbackResult,
    PatternAnalysisResult,
    ProblemAnalysisResult,
    WeeklyFeedbackResult,
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


def build_weekly_consumption_pattern_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], PatternAnalysisResult]:
    """주간 소비 패턴 탐지 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_weekly_consumption_pattern_prompt() | llm.with_structured_output(
        PatternAnalysisResult
    )  # type: ignore[return-value]


def build_weekly_consumption_problem_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ProblemAnalysisResult]:
    """주간 문제 소비 식별 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_weekly_consumption_problem_prompt() | llm.with_structured_output(
        ProblemAnalysisResult
    )  # type: ignore[return-value]


def build_weekly_consumption_cause_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], CauseAnalysisResult]:
    """주간 소비 원인 해석 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_weekly_consumption_cause_prompt() | llm.with_structured_output(CauseAnalysisResult)  # type: ignore[return-value]


def build_weekly_consumption_action_chain(
    llm: BaseChatModel,
) -> Runnable[dict[str, str], ActionAnalysisResult]:
    """주간 행동 개선 포인트 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_weekly_consumption_action_prompt() | llm.with_structured_output(
        ActionAnalysisResult
    )  # type: ignore[return-value]


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
    """월간 행동 개선 포인트 프롬프트와 구조화 출력 모델을 연결한 체인을 생성한다."""
    return build_monthly_consumption_action_prompt() | llm.with_structured_output(
        ActionAnalysisResult
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


def build_weekly_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """주간 JSON 지표 기반 소비 패턴, 문제 소비, 원인, 행동 포인트 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    diagnosis_chain = RunnableParallel(
        raw_json=RunnableLambda(_extract_raw_json),
        indicator_json=RunnableLambda(_extract_indicator_json),
        user_profile_json=RunnableLambda(_extract_user_profile_json),
        pattern_result=build_weekly_consumption_pattern_chain(chat_model),
        problem_result=build_weekly_consumption_problem_chain(chat_model),
    )
    return (
        diagnosis_chain
        | RunnablePassthrough.assign(
            cause_result=RunnableLambda(prepare_cause_payload)
            | build_weekly_consumption_cause_chain(chat_model)
        )
        | RunnablePassthrough.assign(
            action_result=RunnableLambda(prepare_action_payload)
            | build_weekly_consumption_action_chain(chat_model)
        )
    )


def build_monthly_spending_analysis_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], dict[str, object]]:
    """월간 JSON 지표 기반 소비 패턴, 문제 소비, 원인, 행동 포인트 체인을 생성한다."""
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
        | RunnablePassthrough.assign(
            action_result=RunnableLambda(prepare_action_payload)
            | build_monthly_consumption_action_chain(chat_model)
        )
    )


def build_memory_summary_chain(
    period_label: str,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], str]:
    """기간별 피드백 세션 목록을 장기 메모리 요약 문자열로 변환하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_memory_summary_prompt(period_label) | chat_model | StrOutputParser()


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
