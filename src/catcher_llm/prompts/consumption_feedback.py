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


def build_daily_feedback_prompt() -> ChatPromptTemplate:
    """일일 소비 분석, 해석 결과, RAG 문서 근거 기반 최종 피드백 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 오늘 카드 소비를 보고 단호하지만 무례하지 않게 일일 소비 잔소리를 하는 금융 코치다. "
                "반드시 제공된 일일 소비 분석 JSON, 소비 해석 JSON, RAG 검색 문서 근거, 사용자 프로필, "
                "사용자 메모리 맥락만 사용하라. 추측으로 소비 이유를 만들지 말고, JSON 수치 근거와 문서 근거를 "
                "함께 연결하라. 사용자 프로필과 메모리 맥락은 개인화와 반복 패턴 설명에만 사용하고, "
                "오늘 발생하지 않은 소비를 단정하지 마라. "
                "모든 문장은 한국어로 작성하고, 행동 제안은 오늘 또는 내일 바로 확인 가능한 수준으로 제한하라.",
            ),
            (
                "human",
                "아래 데이터를 바탕으로 일일 소비 잔소리 피드백을 구조화해 작성하라.\n"
                "필수 조건:\n"
                "- JSON 수치 근거를 최소 2개 이상 사용한다.\n"
                "- RAG 문서 근거가 있는 행동 조언을 우선 제안한다.\n"
                "- key_evidences에는 source_json_path 또는 source/page_number를 가능한 한 채운다.\n"
                "- scolding_message는 단호하게 쓰되 비난, 조롱, 과장 표현은 피한다.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "사용자 메모리 및 최근 세션 JSON:\n{memory_context_json}\n\n"
                "일일 소비 분석 JSON:\n{daily_json}\n\n"
                "소비 해석 JSON:\n{interpretation_json}\n\n"
                "RAG 검색 문서 근거:\n{retrieved_contexts}",
            ),
        ]
    )


def build_weekly_consumption_pattern_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 소비 패턴 탐지용 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 주간 카드 소비 JSON 데이터를 해석하는 금융 코치다. 제공된 원본 주간 JSON과 "
                "추출 지표 JSON만 사용해 판단하고, 사용자 프로필 JSON은 맥락화와 우선순위 판단에만 "
                "사용하라. 전주 대비 변화, 반복 가맹점, 요일 패턴, 야간·소액·고액 소비를 구분해 "
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
                "제공된 JSON 값만 사용하고 추측성 서술은 최소화하라. 전주 대비 증가 카테고리, "
                "반복 결제, 야간 소비, 소액 누적, 고액 결제를 서로 분리해 판단하라.",
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


def build_weekly_feedback_prompt() -> ChatPromptTemplate:
    """주간 소비 분석, 해석 결과, RAG 문서 근거 기반 최종 피드백 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 한 주 카드 소비를 보고 단호하지만 무례하지 않게 주간 소비 피드백을 하는 "
                "금융 코치다. 반드시 제공된 주간 소비 분석 JSON, 소비 해석 JSON, RAG 검색 문서 근거, "
                "사용자 프로필만 사용하라. 추측으로 소비 이유를 만들지 말고, JSON 수치 근거와 문서 근거를 "
                "함께 연결하라. 모든 문장은 한국어로 작성하고, 행동 제안은 다음 주에 확인 가능한 수준으로 "
                "제한하라.",
            ),
            (
                "human",
                "아래 데이터를 바탕으로 주간 소비 피드백을 구조화해 작성하라.\n"
                "필수 조건:\n"
                "- JSON 수치 근거를 최소 2개 이상 사용한다.\n"
                "- RAG 문서 근거가 있는 행동 조언을 우선 제안한다.\n"
                "- key_evidences에는 source_json_path 또는 source/page_number를 가능한 한 채운다.\n"
                "- feedback_message는 단호하게 쓰되 비난, 조롱, 과장 표현은 피한다.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "주간 소비 분석 JSON:\n{weekly_json}\n\n"
                "소비 해석 JSON:\n{interpretation_json}\n\n"
                "RAG 검색 문서 근거:\n{retrieved_contexts}",
            ),
        ]
    )
