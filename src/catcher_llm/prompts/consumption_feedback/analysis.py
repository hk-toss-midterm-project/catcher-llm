from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_consumption_pattern_prompt() -> ChatPromptTemplate:
    """JSON 소비 지표 기반 소비 패턴 탐지용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 카드 소비 JSON 데이터를 해석하는 금융 코치다. 제공된 원본 JSON과 추출 지표 JSON만 사용해 판단하고, "
                "사용자 프로필 JSON은 맥락화와 우선순위 판단에만 사용하라. 마크다운 소비 보고서로 재구성하지 마라. "
                "일일 비교 기준은 어제, 지난주 같은 요일, 최근 4주 같은 요일 평균을 함께 고려하라. "
                "근거가 부족하면 confidence를 낮게 설정하라. "
                "모든 응답은 한국어로 작성하고 evidences.json_path에는 실제 JSON 경로를 적어라.",
            ),
            (
                "human",
                "아래 JSON 데이터를 바탕으로 소비 패턴을 탐지하라.\n"
                "반드시 반복 소비, 과소비 구간, 충동소비 의심 패턴, 시간대/상황별 소비 패턴을 각각 채워라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}",
            ),
        ]
    )


def build_consumption_problem_prompt() -> ChatPromptTemplate:
    """JSON 소비 지표 기반 문제 소비 식별용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 절약 실패 원인을 JSON 지표로 분리해서 설명하는 소비 분석가다. "
                "제공된 JSON 값만 사용하고 추측성 서술은 최소화하라. 사용자 프로필 JSON은 문제 우선순위와 "
                "실행 가능성 판단에만 사용하고, 프로필만으로 소비 이유를 단정하지 마라. "
                "어제 대비, 지난주 같은 요일 대비, 최근 4주 같은 요일 평균 대비를 구분해 판단하라. "
                "고정비와 변동비 문제를 분리하고, 단기 문제와 장기 문제를 구분하라.",
            ),
            (
                "human",
                "아래 JSON 데이터를 바탕으로 문제 소비를 식별하라.\n"
                "새는 돈 포인트, 절약 방해 요소, 고정비 문제, 변동비 문제, 단기 문제 소비, 장기 문제 소비를 채워라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}",
            ),
        ]
    )


def build_consumption_cause_prompt() -> ChatPromptTemplate:
    """JSON 소비 지표 기반 소비 원인 해석용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 이미 식별된 패턴과 문제 소비를 바탕으로 행동 원인을 해석하는 분석가다. "
                "원본 JSON과 추출 지표 JSON의 수치 근거를 우선 사용하라. "
                "습관성, 보상성, 스트레스성, 편의성 기반, 소액 누적형 소비를 구분해 설명하라.",
            ),
            (
                "human",
                "아래 JSON 데이터와 선행 분석 결과를 바탕으로 소비 원인을 해석하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}",
            ),
        ]
    )


def build_consumption_action_prompt() -> ChatPromptTemplate:
    """JSON 소비 지표 기반 행동 개선 포인트 도출용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 소비 코칭 액션 플랜을 만드는 코치다. 즉시 줄일 수 있는 소비, 대체 가능한 소비, "
                "예산 통제가 필요한 영역, 다음 주 행동 미션, 그룹 경쟁 지표를 구체적으로 제안하라. "
                "행동 항목은 짧고 실행 가능해야 하며 target_json_path에는 연결되는 JSON 경로를 적어라.",
            ),
            (
                "human",
                "아래 JSON 데이터와 선행 분석을 바탕으로 행동 개선 포인트를 도출하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}\n\n"
                "소비 원인 결과:\n{cause_text}",
            ),
        ]
    )


def build_consumption_cause_action_prompt() -> ChatPromptTemplate:
    """원인 해석과 행동 개선 포인트를 한 번에 도출하는 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 소비 원인 해석과 실행 액션을 함께 설계하는 금융 코치다. "
                "패턴 탐지 결과와 문제 소비 결과를 우선 근거로 삼고, 원인과 행동 항목이 서로 직접 연결되게 작성하라. "
                "원본 JSON과 추출 지표 JSON의 실제 수치와 경로를 사용하고, target_json_path에는 연결되는 JSON 경로를 적어라.",
            ),
            (
                "human",
                "아래 JSON 데이터와 선행 분석 결과를 바탕으로 소비 원인과 행동 개선 포인트를 함께 도출하라.\n"
                "원인은 습관성, 보상성, 스트레스성, 편의성 기반, 소액 누적형 소비를 구분하고, "
                "행동은 즉시 줄일 수 있는 소비, 대체 가능한 소비, 예산 통제 영역, 다음 주 행동 미션, "
                "그룹 경쟁 지표를 구체적으로 제안하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}",
            ),
        ]
    )


def build_consumption_unified_analysis_prompt() -> ChatPromptTemplate:
    """패턴, 문제, 원인, 행동 개선 포인트를 한 번에 도출하는 통합 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 카드 소비 JSON 데이터를 종합 해석하는 금융 코치다. 제공된 원본 JSON과 추출 지표 JSON만 사용해 "
                "패턴, 문제 소비, 원인, 행동 개선 포인트를 한 번에 도출하라. 사용자 프로필 JSON은 맥락화와 "
                "우선순위 판단에만 사용하고, 프로필만으로 소비 이유를 단정하지 마라. 모든 응답은 한국어로 작성하라.",
            ),
            (
                "human",
                "아래 JSON 데이터를 바탕으로 패턴, 문제 소비, 원인, 행동 개선 포인트를 모두 도출하라.\n"
                "패턴은 반복 소비, 과소비 구간, 충동소비 의심 패턴, 시간대/상황별 소비 패턴을 포함하라.\n"
                "문제 소비는 새는 돈 포인트, 절약 방해 요소, 고정비 문제, 변동비 문제, 단기 문제 소비, "
                "장기 문제 소비를 포함하라.\n"
                "원인은 습관성, 보상성, 스트레스성, 편의성 기반, 소액 누적형 소비를 구분하라.\n"
                "행동은 즉시 줄일 수 있는 소비, 대체 가능한 소비, 예산 통제 영역, 다음 주 행동 미션, "
                "그룹 경쟁 지표를 구체적으로 제안하라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}",
            ),
        ]
    )


__all__ = [
    "build_consumption_pattern_prompt",
    "build_consumption_problem_prompt",
    "build_consumption_cause_prompt",
    "build_consumption_action_prompt",
    "build_consumption_cause_action_prompt",
    "build_consumption_unified_analysis_prompt",
]
