from __future__ import annotations

from catcher_llm.schemas.consumption_feedback import JsonObject

_HIGH_RATIO_PERCENT = 50.0
_HIGH_RATIO_DIFF_POINT = 30.0
_SMALL_DENOMINATOR_RATIO = 0.8


def build_ratio_context_warning(
    *,
    category: str,
    period_label: str,
    reference_label: str,
    current_ratio_percent: float,
    reference_ratio_percent: float | None = None,
    current_amount: float | int | None = None,
    reference_amount: float | int | None = None,
    current_total: float | int | None = None,
    reference_total: float | int | None = None,
    current_count: int | None = None,
    low_count_threshold: int = 2,
) -> JsonObject | None:
    """특정 항목 비중이 작은 분모 때문에 과장될 수 있는지 판단해 경고 JSON을 만든다."""
    ratio_diff = (
        current_ratio_percent - reference_ratio_percent
        if reference_ratio_percent is not None
        else 0.0
    )
    is_ratio_notable = (
        current_ratio_percent >= _HIGH_RATIO_PERCENT or ratio_diff >= _HIGH_RATIO_DIFF_POINT
    )
    if not is_ratio_notable:
        return None

    reasons: list[str] = []
    if (
        current_total is not None
        and reference_total is not None
        and float(reference_total) > 0
        and float(current_total) <= float(reference_total) * _SMALL_DENOMINATOR_RATIO
    ):
        reasons.append(
            f"{period_label} 총소비가 {reference_label}보다 낮아 특정 지출 비중이 크게 보일 수 있음"
        )

    if current_count is not None and current_count <= low_count_threshold:
        reasons.append(
            f"{period_label} 해당 카테고리 거래 건수가 {current_count}건이라 비중만으로 반복 습관을 단정하기 어려움"
        )

    if (
        current_amount is not None
        and reference_amount is not None
        and float(reference_amount) > 0
        and float(current_amount)
        <= max(float(reference_amount) * 1.15, float(reference_amount) + 10_000)
    ):
        reasons.append(
            f"{period_label} 절대금액이 {reference_label} 대비 크게 늘지 않아 비중 증가만으로 과소비를 단정하기 어려움"
        )

    if not reasons:
        return None

    warning: JsonObject = {
        "category": category,
        "period_label": period_label,
        "reference_label": reference_label,
        "current_ratio_percent": round(float(current_ratio_percent), 4),
        "reasons": reasons,
        "interpretation_rule": (
            "비중 수치만으로 급증, 습관 악화, 예산 초과를 단정하지 말고 "
            "절대금액, 거래 건수, 반복성, 예산 대비 영향을 함께 확인한다."
        ),
    }
    if reference_ratio_percent is not None:
        warning["reference_ratio_percent"] = round(float(reference_ratio_percent), 4)
        warning["ratio_diff_point"] = round(float(ratio_diff), 4)
    if current_amount is not None:
        warning["current_amount"] = int(round(float(current_amount)))
    if reference_amount is not None:
        warning["reference_amount"] = int(round(float(reference_amount)))
    if current_total is not None:
        warning["current_total"] = int(round(float(current_total)))
    if reference_total is not None:
        warning["reference_total"] = int(round(float(reference_total)))
    if current_count is not None:
        warning["current_count"] = current_count
    return warning
