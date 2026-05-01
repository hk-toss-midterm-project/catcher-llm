from __future__ import annotations

ACTION_CANDIDATE_OUTPUT_RULES = (
    "출력 규칙: title/detail/expected_effect는 RAG 검색·검토용 후보형 표현으로 작성하라. "
    "금지 표현: '줄여보세요', '하세요', '재조정하세요', '계획하세요', '절약 가능'처럼 "
    "사용자에게 바로 지시하거나 효과를 확정하는 문장을 쓰지 마라. "
    "근거 없는 가맹점명·서비스명·상품명은 만들지 말고, 프로필만으로 생활 제안을 만들지 마라. "
    "action_type은 generic immediate/substitution/mission이 아니라 구체적인 snake_case 후보 유형으로 "
    "쓰고, 가능하면 _candidate로 끝내라. "
    "target_json_path는 배열 인덱스만 단독으로 쓰지 말고 total_amount, diff_amount, amount, "
    "ratio_percent 같은 실제 판단 필드까지 좁혀라. "
    "expected_effect는 확정 절약액이 아니라 가능성 검토, 원인 확인, 리스크 완화 가능성처럼 낮춰 써라. "
    "group_competition_metrics는 소비 총액 같은 결과 지표를 우선 제안하지 말고 주문 횟수, "
    "예산 초과 카테고리 수, 사전 점검률 같은 행동 기반 지표로 작성하라."
)

CAUSE_INTERVENTION_TARGET_RULES = (
    "원인 해석 단계에서는 cause_result.intervention_targets를 함께 작성하라. "
    "intervention_targets는 최종 행동 지시나 미션이 아니라 RAG 검색 타겟 후보이다. "
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

COMMON_INTERPRETATION_CLASSIFICATION_RULES = (
    "분류 규칙: 상위 가맹점 목록은 반복 근거가 아니다. "
    "visit_count가 2 미만이고 consecutive_merchants 근거가 없으면 repeated_consumption과 habitual_causes에 넣지 마라. "
    "1회 여가 결제만으로 impulse_patterns, reward_causes, stress_causes를 만들지 마라. "
    "감소한 카테고리나 diff_amount 또는 diff_point가 0 이하인 항목은 절약 관점의 문제 소비로 분류하지 마라. "
    "지속 증가나 장기 문제는 기간별 비교 기준 또는 추이 데이터가 있을 때만 작성하라. "
    "납부·관리비·전기요금 같은 고정비는 stress_causes나 reward_causes가 아니라 fixed_cost 점검 타겟으로만 다뤄라. "
    "방문이라고 쓰지 말고 카드 데이터 기준의 결제라고 써라. "
)

DAILY_INTERPRETATION_CLASSIFICATION_RULES = (
    f"{COMMON_INTERPRETATION_CLASSIFICATION_RULES}"
    "가맹점 고액 결제 근거는 anomaly_detection.high_spending_items[n].amount를 사용하라. "
    "카테고리 비중 변화 근거는 stable_metrics.category_ratio_changes[n].diff_point 또는 today_ratio_percent를 사용하라. "
    "일일 장기 문제는 daily_comparisons.recent_4week_same_weekday_average 같은 일일 비교 근거가 있을 때만 작성하라. "
)

WEEKLY_INTERPRETATION_CLASSIFICATION_RULES = (
    f"{COMMON_INTERPRETATION_CLASSIFICATION_RULES}"
    "가맹점 고액 결제 근거는 weekly_summary가 아니라 waste_detection.high_spending.items[n].amount를 사용하라. "
    "카테고리 증감 문제 근거는 전체 증감액이 아니라 category_summary[n].diff_amount 또는 total_amount를 사용하라. "
    "카테고리의 장기 문제는 전체 주간 비교만으로 단정하지 말고 카테고리 추이 근거가 있을 때만 작성하라. "
    "핵심 고액 결제 타겟이 있으면 같은 결제를 설명하는 주말·야간·전체 비교 집계 타겟을 중복 생성하지 마라. "
    "weekday_concentration_ratio_percent는 최대 소비 요일의 집중도이며 주중 소비 비중이 아니다. "
)

MONTHLY_INTERPRETATION_CLASSIFICATION_RULES = (
    f"{COMMON_INTERPRETATION_CLASSIFICATION_RULES}"
    "가맹점 고액 결제 근거는 high_spending.items[n].amount를 사용하라. "
    "카테고리 증감 문제 근거는 전체 월간 증감액이 아니라 category_deep[n].diff_amount 또는 total_amount를 사용하라. "
    "월간 장기 문제는 monthly_comparisons.recent_3month_average 또는 weekly_trend.weekly_breakdown 같은 월간 추이 근거가 있을 때만 작성하라. "
)

INTERPRETATION_CLASSIFICATION_RULES = WEEKLY_INTERPRETATION_CLASSIFICATION_RULES

__all__ = [
    "ACTION_CANDIDATE_OUTPUT_RULES",
    "CAUSE_INTERVENTION_TARGET_RULES",
    "COMMON_INTERPRETATION_CLASSIFICATION_RULES",
    "DAILY_INTERPRETATION_CLASSIFICATION_RULES",
    "INTERPRETATION_CLASSIFICATION_RULES",
    "MONTHLY_INTERPRETATION_CLASSIFICATION_RULES",
    "WEEKLY_INTERPRETATION_CLASSIFICATION_RULES",
]
