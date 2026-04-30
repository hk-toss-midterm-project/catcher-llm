from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_weekly_consumption_pattern_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 소비 패턴 탐지용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 주간 카드 소비 JSON 데이터를 해석하는 금융 코치다. 제공된 원본 주간 JSON과 "
                "추출 지표 JSON만 사용해 판단하고, 사용자 프로필 JSON은 맥락화와 우선순위 판단에만 "
                "사용하라. 전주 대비 변화, 최근 4주 평균 대비 변화, 지난달 같은 주차 대비 변화, "
                "반복 가맹점, 요일 패턴, 야간·소액·고액 소비를 구분해 "
                "해석하라. 모든 응답은 한국어로 작성하고 evidences.json_path에는 실제 JSON 경로를 적어라.",
            ),
            (
                "human",
                "아래 주간 JSON 데이터를 바탕으로 소비 패턴을 탐지하라.\n"
                "반드시 반복 소비, 과소비 구간, 충동소비 의심 패턴, 요일/상황별 소비 패턴을 각각 채워라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 주간 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}",
            ),
        ]
    )


def build_weekly_consumption_problem_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 문제 소비 식별용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 주간 소비에서 절약을 방해하는 문제를 JSON 지표로 분리해서 설명하는 소비 분석가다. "
                "제공된 JSON 값만 사용하고 추측성 서술은 최소화하라. 전주 대비, 최근 4주 평균 대비, "
                "지난달 같은 주차 대비 변화와 반복 결제, 야간 소비, 소액 누적, 고액 결제를 서로 분리해 판단하라.",
            ),
            (
                "human",
                "아래 주간 JSON 데이터를 바탕으로 문제 소비를 식별하라.\n"
                "새는 돈 포인트, 절약 방해 요소, 고정비 문제, 변동비 문제, 단기 문제 소비, 장기 문제 소비를 채워라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 주간 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}",
            ),
        ]
    )


def build_weekly_consumption_cause_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 소비 원인 해석용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 이미 식별된 주간 패턴과 문제 소비를 바탕으로 행동 원인을 해석하는 분석가다. "
                "원본 주간 JSON과 추출 지표 JSON의 수치 근거를 우선 사용하라. 습관성, 보상성, "
                "스트레스성, 편의성 기반, 소액 누적형 소비를 구분해 설명하라.",
            ),
            (
                "human",
                "아래 주간 JSON 데이터와 선행 분석 결과를 바탕으로 소비 원인을 해석하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 주간 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}",
            ),
        ]
    )


def build_weekly_consumption_action_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 다음 주 행동 개선 포인트 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 다음 주 소비 코칭 액션 플랜을 만드는 코치다. 즉시 줄일 수 있는 소비, "
                "대체 가능한 소비, 예산 통제가 필요한 영역, 다음 주 행동 미션, 그룹 경쟁 지표를 "
                "구체적으로 제안하라. 행동 항목은 짧고 실행 가능해야 하며 target_json_path에는 "
                "연결되는 JSON 경로를 적어라.",
            ),
            (
                "human",
                "아래 주간 JSON 데이터와 선행 분석을 바탕으로 행동 개선 포인트를 도출하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 주간 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}\n\n"
                "소비 원인 결과:\n{cause_text}",
            ),
        ]
    )


__all__ = [
    "build_weekly_consumption_pattern_prompt",
    "build_weekly_consumption_problem_prompt",
    "build_weekly_consumption_cause_prompt",
    "build_weekly_consumption_action_prompt",
]
