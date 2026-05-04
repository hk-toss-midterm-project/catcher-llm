from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from catcher_llm.prompts.consumption_feedback.action_rules import (
    WEEKLY_INTERPRETATION_CLASSIFICATION_RULES,
)

_WEEKLY_CAUSE_INTERVENTION_TARGET_RULES = (
    "원인 해석 단계에서는 cause_result.intervention_targets를 함께 작성하라. "
    "intervention_targets는 최종 행동 지시나 미션이 아니라 최종 피드백 검토 후보이다. "
    "action_result를 별도 생성하지 않는다. "
    "각 타겟은 linked_cause로 원인 항목과 직접 연결하고, target_json_path는 실제 판단 필드까지 좁혀라. "
    "여가 지출 감소를 충동 소비, 스트레스 원인, 소비 증대 필요로 단정하지 마라. "
    "전기요금·관리비 같은 납부/고정비는 불필요 지출로 단정하지 말고 점검 후보로만 낮춰 써라. "
    "직업·페르소나만으로 스트레스성, 보상성 원인을 만들지 마라. "
    "습관성·보상성·스트레스성·편의성·소액 누적으로 단정하기 어려운 경우에도 원인 결과를 비워두지 말고, "
    "단일 고액 결제 이벤트는 one_off_high_spending_causes, 고정비 납부 타이밍은 fixed_cost_timing_causes, "
    "요일·기간 집중은 period_concentration_causes에 근거 중심 후보로 작성하라. "
    "고정비 항목은 one_off_high_spending_causes에 중복으로 넣지 말고 fixed_cost_timing_causes에만 넣어라. "
    "고액 결제 근거는 items[n]처럼 배열 항목만 쓰지 말고 items[n].amount까지 좁혀라. "
    "요일·기간 집중 원인은 최대 소비 요일의 실제 금액, 요일 편중도, 주말·주중 비중, "
    "같은 날 발생한 핵심 고액 결제를 함께 연결해 설명하라."
)

_WEEKLY_ACTION_CANDIDATE_OUTPUT_RULES = (
    "출력 규칙: title/detail/expected_effect는 최종 피드백 검토 후보형 표현으로 작성하라. "
    "금지 표현: '줄여보세요', '하세요', '재조정하세요', '계획하세요', '절약 가능'처럼 "
    "사용자에게 바로 지시하거나 효과를 확정하는 문장을 쓰지 마라. "
    "근거 없는 가맹점명·서비스명·상품명은 만들지 말고, 프로필만으로 생활 제안을 만들지 마라. "
    "action_type은 generic immediate/substitution/mission이 아니라 구체적인 snake_case 후보 유형으로 "
    "쓰고, 가능하면 _candidate로 끝내라. "
    "target_json_path는 배열 인덱스만 단독으로 쓰지 말고 total_amount, diff_amount, amount, "
    "ratio_percent 같은 실제 판단 필드까지 좁혀라. "
    "예산 통제 후보는 연소득·월소득·월 목표 소비 한도와 예산 사용률 지표를 함께 근거로 삼아라. "
    "expected_effect는 확정 절약액이 아니라 가능성 검토, 원인 확인, 리스크 완화 가능성처럼 낮춰 써라. "
    "group_competition_metrics는 소비 총액 같은 결과 지표를 우선 제안하지 말고 주문 횟수, "
    "예산 초과 카테고리 수, 사전 점검률 같은 행동 기반 지표로 작성하라."
)


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
                "해석하라. "
                f"{WEEKLY_INTERPRETATION_CLASSIFICATION_RULES} "
                "모든 응답은 한국어로 작성하고 evidences.json_path에는 실제 JSON 경로를 적어라.",
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
                "지난달 같은 주차 대비 변화와 반복 결제, 야간 소비, 소액 누적, 고액 결제를 서로 분리해 판단하라. "
                f"{WEEKLY_INTERPRETATION_CLASSIFICATION_RULES}",
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
                "스트레스성, 편의성 기반, 소액 누적형 소비를 구분하되, 주간 단일 고액 결제, "
                "고정비 납부 타이밍, 요일·기간 집중도 별도 원인 후보로 설명하라. "
                "one_off_high_spending_causes에는 납부·고정비가 아닌 재량 고액 결제만 넣고, "
                "도시가스·통신비·유선방송요금 같은 납부 항목은 fixed_cost_timing_causes로 분리하라. "
                "period_concentration_causes는 단순히 주말 비중만 말하지 말고 피크 요일 금액, "
                "요일 편중도, 주말 소비 비중, 같은 날 발생한 핵심 고액 결제를 연결해 작성하라. "
                f"{WEEKLY_INTERPRETATION_CLASSIFICATION_RULES} "
                f"{_WEEKLY_CAUSE_INTERVENTION_TARGET_RULES}",
            ),
            (
                "human",
                "아래 주간 JSON 데이터와 선행 분석 결과를 바탕으로 소비 원인을 해석하라.\n\n"
                "원인별로 최종 피드백 검토 후보가 필요하면 intervention_targets에만 담고, 최종 행동 지시로 쓰지 마라.\n\n"
                "사용자 프로필 JSON:\n{user_profile_json}\n\n"
                "원본 주간 JSON:\n{raw_json}\n\n"
                "추출 지표 JSON:\n{indicator_json}\n\n"
                "패턴 탐지 결과:\n{pattern_text}\n\n"
                "문제 소비 결과:\n{problem_text}",
            ),
        ]
    )


def build_weekly_consumption_action_prompt() -> ChatPromptTemplate:
    """주간 소비 JSON 지표 기반 개선 후보와 개입 타겟 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 다음 주 소비 코칭 액션 플랜을 확정하는 코치가 아니라 최종 피드백 결정을 돕는 "
                "개선 후보·개입 타겟 추출기다. 즉시 점검할 소비 후보, 대체 가능성이 있는 소비 후보, "
                "예산 통제가 필요한 영역 후보, 다음 주 행동 후보, 그룹 경쟁 지표 후보를 구체적으로 정리하라. "
                "행동 항목은 사용자가 바로 받을 최종 미션이 아니라 검토 후보이며, 최종 미션으로 확정하지 마라. "
                "target_json_path에는 연결되는 JSON 경로를 적어라. "
                f"{_WEEKLY_ACTION_CANDIDATE_OUTPUT_RULES}",
            ),
            (
                "human",
                "아래 주간 JSON 데이터와 선행 분석을 바탕으로 개선 후보와 개입 타겟을 도출하라.\n"
                "각 항목은 최종 피드백 판단에 쓰일 중간 산출물로 작성하라.\n"
                "후보 항목은 사용자가 수행할 최종 문장으로 쓰지 말고, 검토할 행동 단서와 근거 경로를 남겨라.\n\n"
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
