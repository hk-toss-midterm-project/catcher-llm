from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from catcher_llm.schemas.consumption_feedback import (
    AnomalyDetection,
    CategoryDirection,
    CategoryRatioChange,
    CategoryShiftIndicator,
    CauseAnalysisResult,
    HighSpendingItem,
    MainCategoryShift,
    MetricValue,
    PatternAnalysisResult,
    PreviousDayComparison,
    ProblemAnalysisResult,
    SpendingIndicatorPayload,
    SpendingMetric,
    UserProfileContext,
    UserSpendingData,
)


def load_user_spending_data(json_path: Path) -> UserSpendingData:
    """사용자 소비 JSON 파일을 읽어 검증된 입력 모델로 변환한다."""
    with json_path.open(encoding="utf-8") as file:
        raw_payload: object = json.load(file)
    return UserSpendingData.model_validate(raw_payload)


def parse_user_spending_data(payload: object) -> UserSpendingData:
    """이미 메모리에 있는 소비 분석 JSON 객체를 검증된 입력 모델로 변환한다."""
    return UserSpendingData.model_validate(payload)


def get_category_direction(diff_point: float) -> CategoryDirection:
    """카테고리 비중 차이를 증가, 감소, 변화 없음으로 분류한다."""
    if diff_point > 0:
        return "increase"
    if diff_point < 0:
        return "decrease"
    return "flat"


def make_spending_metric(
    name: str,
    value: MetricValue,
    unit: str,
    source_json_path: str,
    description: str,
) -> SpendingMetric:
    """소비 지표 이름, 값, 단위, JSON 경로를 하나의 모델로 묶는다."""
    return SpendingMetric(
        name=name,
        value=value,
        unit=unit,
        source_json_path=source_json_path,
        description=description,
    )


def build_category_shift_indicators(
    changes: list[CategoryRatioChange],
) -> list[CategoryShiftIndicator]:
    """카테고리 비중 변화 원본 배열을 JSON 경로가 포함된 지표 목록으로 변환한다."""
    return [
        CategoryShiftIndicator(
            category=change.category,
            usual_ratio_percent=change.usual_ratio_percent,
            today_ratio_percent=change.today_ratio_percent,
            diff_point=change.diff_point,
            direction=get_category_direction(change.diff_point),
            source_json_path=f"stable_metrics.category_ratio_changes[{index}]",
        )
        for index, change in enumerate(changes)
    ]


def get_largest_category_increase(
    changes: list[CategoryShiftIndicator],
) -> CategoryShiftIndicator | None:
    """비중이 가장 크게 증가한 카테고리 지표를 찾는다."""
    increased_changes = [change for change in changes if change.direction == "increase"]
    if not increased_changes:
        return None
    return max(increased_changes, key=lambda change: change.diff_point)


def get_largest_category_decrease(
    changes: list[CategoryShiftIndicator],
) -> CategoryShiftIndicator | None:
    """비중이 가장 크게 감소한 카테고리 지표를 찾는다."""
    decreased_changes = [change for change in changes if change.direction == "decrease"]
    if not decreased_changes:
        return None
    return min(decreased_changes, key=lambda change: change.diff_point)


def get_largest_high_spending_item(items: list[HighSpendingItem]) -> HighSpendingItem | None:
    """특이 지출 항목 중 금액이 가장 큰 항목을 찾는다."""
    if not items:
        return None
    return max(items, key=lambda item: item.amount)


def _build_high_spending_metric(anomaly: AnomalyDetection) -> SpendingMetric | None:
    """최대 고액 지출 항목이 있으면 핵심 지표 모델로 변환한다."""
    high_spending_item = get_largest_high_spending_item(anomaly.high_spending_items)
    if high_spending_item is None:
        return None
    return make_spending_metric(
        "최대 고액 지출 금액",
        high_spending_item.amount,
        "KRW",
        "anomaly_detection.high_spending_items",
        f"고액 지출 항목 중 가장 큰 결제 금액: {high_spending_item.description}",
    )


def _build_previous_day_metrics(previous_day: PreviousDayComparison) -> list[SpendingMetric]:
    """전날 대비 지출과 결제 건수 차이를 핵심 지표 목록으로 변환한다."""
    return [
        make_spending_metric(
            "전날 대비 지출 증감액",
            previous_day.amount_diff,
            "KRW",
            "previous_day_comparison.amount_diff",
            "전날 총 지출과 오늘 총 지출의 차이",
        ),
        make_spending_metric(
            "전날 대비 지출 증감률",
            previous_day.amount_diff_rate_percent,
            "percent",
            "previous_day_comparison.amount_diff_rate_percent",
            "전날 총 지출 대비 오늘 지출 증감률",
        ),
        make_spending_metric(
            "전날 대비 결제 건수 증감",
            previous_day.count_diff,
            "count",
            "previous_day_comparison.count_diff",
            "전날 결제 건수와 오늘 결제 건수의 차이",
        ),
    ]


def _build_document_daily_metrics(user_data: UserSpendingData) -> list[SpendingMetric]:
    """문서 기준 일일 추가 지표를 해석 체인용 핵심 지표 목록으로 변환한다."""
    daily_metrics = user_data.daily_metrics
    special_metrics = daily_metrics.special_metrics
    return [
        make_spending_metric(
            "야간 소비 비중",
            daily_metrics.late_night_ratio_percent,
            "percent",
            "daily_metrics.late_night_ratio_percent",
            "분석 기준일 총 소비 중 심야 시간대 소비가 차지하는 비중",
        ),
        make_spending_metric(
            "일일 예산 소진율",
            daily_metrics.daily_budget_usage_rate_percent,
            "percent",
            "daily_metrics.daily_budget_usage_rate_percent",
            "설정된 일일 예산 대비 분석 기준일 소비 금액 비율",
        ),
        make_spending_metric(
            "무소비일 여부",
            daily_metrics.no_spending_day,
            "boolean",
            "daily_metrics.no_spending_day",
            "분석 기준일에 소비가 전혀 없었는지 여부",
        ),
        make_spending_metric(
            "일일 이상 소비 점수",
            daily_metrics.daily_anomaly_score,
            "ratio",
            "daily_metrics.daily_anomaly_score",
            "과거 원본 일평균 대비 분석 기준일 소비 배율",
        ),
        make_spending_metric(
            "충동소비 점수",
            special_metrics.impulse_spending_score,
            "score",
            "daily_metrics.special_metrics.impulse_spending_score",
            "야간·비필수·거래 빈도 증가를 함께 고려한 일일 충동소비 점수",
        ),
        make_spending_metric(
            "하루 소비 위험도",
            special_metrics.daily_spending_risk,
            "ratio",
            "daily_metrics.special_metrics.daily_spending_risk",
            "과거 원본 일평균 대비 하루 소비 위험 배율",
        ),
    ]


def build_core_metrics(user_data: UserSpendingData) -> list[SpendingMetric]:
    """분석에 자주 쓰는 핵심 소비 지표를 원본 JSON에서 직접 추출한다."""
    stable_metrics = user_data.stable_metrics
    anomaly = user_data.anomaly_detection
    previous_day = user_data.previous_day_comparison
    payment_behavior = user_data.payment_behavior_analysis
    frictionless_spending = payment_behavior.frictionless_spending
    transaction_density = payment_behavior.transaction_density

    metrics = [
        make_spending_metric(
            "과거 안정 일평균",
            stable_metrics.past_daily_stable_average,
            "KRW",
            "stable_metrics.past_daily_stable_average",
            "클리핑 데이터 기준 과거 일평균 소비 금액",
        ),
        make_spending_metric(
            "오늘 총 지출액",
            stable_metrics.today_total,
            "KRW",
            "stable_metrics.today_total",
            "분석 기준일의 총 소비 금액",
        ),
        make_spending_metric(
            "평소 대비 지출 증가율",
            stable_metrics.increase_rate_percent,
            "percent",
            "stable_metrics.increase_rate_percent",
            "과거 안정 일평균 대비 오늘 총 지출 증가율",
        ),
        make_spending_metric(
            "원본 일평균 대비 소비 배율",
            anomaly.spike_ratio,
            "ratio",
            "anomaly_detection.spike_ratio",
            "원본 데이터 과거 일평균 대비 오늘 지출 배율",
        ),
        make_spending_metric(
            "소비 급증 여부",
            anomaly.is_spike,
            "boolean",
            "anomaly_detection.is_spike",
            "오늘 지출이 이상 소비 기준을 넘었는지 여부",
        ),
        make_spending_metric(
            "고액 지출 기준값",
            anomaly.high_spending_threshold,
            "KRW",
            "anomaly_detection.high_spending_threshold",
            "건별 고액 지출 판단에 사용한 상한 기준값",
        ),
        make_spending_metric(
            "고액 지출 건수",
            len(anomaly.high_spending_items),
            "count",
            "anomaly_detection.high_spending_items",
            "고액 지출 기준값을 초과한 결제 항목 수",
        ),
        *_build_previous_day_metrics(previous_day),
        make_spending_metric(
            "오늘 소비 피크 시간대",
            user_data.time_slot_analysis.peak_slot or "",
            "time_slot",
            "time_slot_analysis.peak_slot",
            "분석 기준일 지출액이 가장 큰 시간대",
        ),
        make_spending_metric(
            "마찰력 없는 지출 비중",
            frictionless_spending.ratio_percent,
            "percent",
            "payment_behavior_analysis.frictionless_spending.ratio_percent",
            "온라인/간편결제/앱결제/배달 결제액이 오늘 총 지출에서 차지하는 비중",
        ),
        make_spending_metric(
            "마찰력 없는 지출 건수",
            frictionless_spending.transaction_count,
            "count",
            "payment_behavior_analysis.frictionless_spending.transaction_count",
            "온라인/간편결제/앱결제/배달 키워드가 포함된 당일 결제 건수",
        ),
        make_spending_metric(
            "마찰력 없는 지출액",
            frictionless_spending.total_amount,
            "KRW",
            "payment_behavior_analysis.frictionless_spending.total_amount",
            "온라인/간편결제/앱결제/배달 키워드가 포함된 당일 결제 금액 합계",
        ),
        make_spending_metric(
            "오늘 결제 횟수",
            transaction_density.transaction_count,
            "count",
            "payment_behavior_analysis.transaction_density.transaction_count",
            "분석 기준일 전체 결제 건수",
        ),
        make_spending_metric(
            "1회 결제당 평균 금액",
            transaction_density.average_amount_per_transaction,
            "KRW",
            "payment_behavior_analysis.transaction_density.average_amount_per_transaction",
            "분석 기준일 총 지출을 결제 횟수로 나눈 건당 평균 금액",
        ),
        *_build_document_daily_metrics(user_data),
    ]

    high_spending_metric = _build_high_spending_metric(anomaly)
    if high_spending_metric is not None:
        metrics.append(high_spending_metric)

    return metrics


def extract_spending_indicators(user_data: UserSpendingData) -> SpendingIndicatorPayload:
    """일일 소비 분석 JSON에서 분석 가능한 소비 지표를 마크다운 변환 없이 직접 추출한다."""
    category_changes = build_category_shift_indicators(
        user_data.stable_metrics.category_ratio_changes
    )
    previous_day = user_data.previous_day_comparison
    return SpendingIndicatorPayload(
        member_id=user_data.member_id,
        analysis_date=user_data.analysis_date,
        metrics=build_core_metrics(user_data),
        category_ratio_changes=category_changes,
        largest_category_increase=get_largest_category_increase(category_changes),
        largest_category_decrease=get_largest_category_decrease(category_changes),
        high_spending_items=user_data.anomaly_detection.high_spending_items,
        main_category_shift=MainCategoryShift(
            previous_category=previous_day.yesterday_main_category,
            current_category=previous_day.today_main_category,
            source_json_path="previous_day_comparison",
        ),
        time_slot_diffs=user_data.time_slot_analysis.time_slots,
    )


def _serialize_user_profile_context(user_profile: object | None) -> str:
    """사용자 프로필 객체를 해석 체인 입력용 JSON 문자열로 변환한다."""
    if user_profile is None:
        return "{}"
    if isinstance(user_profile, UserProfileContext):
        return user_profile.model_dump_json(indent=2)
    if isinstance(user_profile, dict):
        return json.dumps(cast(dict[str, object], user_profile), ensure_ascii=False, indent=2)
    return json.dumps({"value": str(user_profile)}, ensure_ascii=False, indent=2)


def make_spending_analysis_input(
    user_data: UserSpendingData,
    *,
    user_profile: object | None = None,
) -> dict[str, str]:
    """소비 분석 체인에 넣을 원본 JSON, 추출 지표 JSON, 사용자 프로필 입력을 생성한다."""
    spending_indicators = extract_spending_indicators(user_data)
    return {
        "raw_json": user_data.model_dump_json(indent=2),
        "indicator_json": spending_indicators.model_dump_json(indent=2),
        "user_profile_json": _serialize_user_profile_context(user_profile),
    }


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
