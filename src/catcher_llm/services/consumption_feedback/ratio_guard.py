from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

from pydantic import BaseModel

from catcher_llm.schemas.consumption_feedback import JsonObject, JsonValue

_OVERSTATED_RATIO_PATTERN = re.compile(
    r"(총\s*[0-9,]+원|급증|예산.{0,20}초과|저축 목표.{0,20}(영향|부정|타격)|전체 지출의)"
)


def collect_ratio_context_warnings(items: Iterable[object]) -> list[JsonObject]:
    """분석 모델이나 dict 목록에서 ratio_context_warning 경고만 추출한다."""
    warnings: list[JsonObject] = []
    for item in items:
        if isinstance(item, BaseModel):
            raw_item = item.model_dump()
        elif isinstance(item, dict):
            raw_item = item
        else:
            continue

        warning = raw_item.get("ratio_context_warning")
        if isinstance(warning, dict):
            warnings.append(cast(JsonObject, warning))
    return warnings


def coerce_ratio_context_message(
    *,
    message: str,
    warnings: Iterable[JsonObject],
    period_label: str,
    mission: str,
) -> str:
    """비중 과장 경고가 있는데 생성 문장이 단정적이면 안전한 맥락 문장으로 교체한다."""
    warning_list = list(warnings)
    if not warning_list or _OVERSTATED_RATIO_PATTERN.search(message) is None:
        return message

    warning = warning_list[0]
    category = _string_value(warning.get("category")) or "해당 항목"
    amount = _number_value(warning.get("current_amount"))
    count = _integer_value(warning.get("current_count"))

    amount_text = f"{int(round(amount)):,}원" if amount is not None else "확인된 금액"
    count_text = f"{count}건" if count is not None else "많지 않은 거래"
    mission_text = mission.strip()

    parts = [
        f"{category} 지출은 {amount_text}으로 확인됐습니다.",
        (
            f"{period_label} {count_text}의 결제라면 오늘 한 번의 지출만으로 "
            f"{category}비 습관이 나빠졌다고 보기는 어렵습니다."
        ),
        "다만 같은 금액대의 이동 지출이 반복되는지만 확인해보세요.",
    ]
    if mission_text:
        parts.append(mission_text)
    return " ".join(parts)


def _number_value(value: JsonValue | object) -> float | None:
    """JSON 값에서 실수로 해석 가능한 숫자를 꺼낸다."""
    if isinstance(value, int | float):
        return float(value)
    return None


def _integer_value(value: JsonValue | object) -> int | None:
    """JSON 값에서 정수로 해석 가능한 숫자를 꺼낸다."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return None


def _string_value(value: JsonValue | object) -> str | None:
    """JSON 값에서 빈 문자열이 아닌 텍스트를 꺼낸다."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
