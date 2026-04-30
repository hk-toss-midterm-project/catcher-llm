from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_memory_summary_prompt(period_label: str) -> ChatPromptTemplate:
    """누적 세션 기록을 하나의 통합 요약문으로 합성하는 메모리 요약 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 소비 피드백 세션 기록들을 읽고, "
                "핵심 소비 패턴과 반복 문제, 개선 흐름을 간결하게 통합 요약하는 메모리 관리자다. "
                "각 세션을 개별 나열하지 말고, 전체를 아우르는 하나의 흐름으로 합쳐서 서술하라. "
                "소비 금액 절대값보다는 패턴과 습관의 변화에 집중하라. "
                "모든 응답은 한국어로 작성하라.",
            ),
            (
                "human",
                f"아래는 사용자의 최근 {period_label} 소비 피드백 세션 목록이다.\n"
                "각 세션을 하나씩 나열하지 말고, 전체를 아우르는 통합 요약을 작성하라.\n"
                "반복되는 소비 문제, 개선된 점, 주의해야 할 점을 3~4문장으로 요약하라.\n\n"
                "세션 목록:\n{session_list}",
            ),
        ]
    )


def build_feedback_memory_rank_prompt() -> ChatPromptTemplate:
    """사용자 거부 이유 목록을 의미 기반 구체성 순으로 재정렬하는 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자가 남긴 피드백 거부 이유 텍스트 목록을 구체성 기준으로 재정렬하는 분류기다.\n"
                "각 줄은 사용자가 직접 작성한 거부 이유 텍스트다.\n\n"
                "구체성 판단 기준:\n"
                "- 높은 구체성: 거부 이유에 구체적 상황·맥락·이유가 담겨 있음\n"
                "  예) '내가 대인기피증이 있어서 버스를 못 타 대중교통에 대해서는 간섭하지 말아줘'\n"
                "  예) '주 3일 야근으로 저녁에 요리할 시간이 없어'\n"
                "- 낮은 구체성: 단순 감정 표현, 단어 하나, 이유 없는 거부\n"
                "  예) '싫어', '하기 싫음', '몰라'\n\n"
                "규칙:\n"
                "- 구체성이 높은 줄을 위에, 낮은 줄을 아래에 배치해 재정렬하라.\n"
                "- 각 줄의 텍스트는 절대 수정하지 말고 순서만 바꿔라.\n"
                "- 출력은 재정렬된 텍스트를 한 줄씩만 출력하라.\n"
                "- 번호, 머리말, 설명 등 다른 텍스트는 절대 포함하지 마라.",
            ),
            (
                "human",
                "아래 거부 이유 목록을 구체성이 높은 순서대로 재정렬해서 출력하라.\n\n{entries}",
            ),
        ]
    )


__all__ = [
    "build_memory_summary_prompt",
    "build_feedback_memory_rank_prompt",
]
