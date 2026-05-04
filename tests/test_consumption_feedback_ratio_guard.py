from __future__ import annotations

from catcher_llm.services.consumption_feedback.ratio_guard import (
    coerce_ratio_context_message,
)


def test_coerce_ratio_context_message_rewrites_overstated_ratio_feedback() -> None:
    """작은 분모 경고가 있을 때 비중 급증 단정문을 결론형 피드백으로 교체하는지 검증한다."""
    warning = {
        "category": "교통",
        "current_ratio_percent": 87.8,
        "current_amount": 62_600,
        "current_count": 1,
        "interpretation_rule": "비중 수치만으로 급증을 단정하지 않는다.",
    }
    original = (
        "오늘 하루 동안 총 71,300원을 지출하셨군요. 특히 교통비가 62,600원으로 "
        "전체 지출의 87.8%를 차지하며, 평소 4.29%에서 급증했습니다. "
        "이렇게 높은 비중의 지출은 예산을 초과하게 만들 수 있습니다."
    )

    rewritten = coerce_ratio_context_message(
        message=original,
        warnings=[warning],
        period_label="오늘",
        mission="내일은 할인 카드나 정기권을 확인해보세요.",
    )

    assert "총 71,300원" not in rewritten
    assert "급증했습니다" not in rewritten
    assert "예산을 초과" not in rewritten
    assert "분모" not in rewritten
    assert "비중이 실제보다 크게 보일 수" not in rewritten
    assert "62,600원" in rewritten
    assert "오늘 한 번의 지출만으로 교통비 습관이 나빠졌다고 보기는 어렵습니다" in rewritten
    assert "반복되는지만 확인해보세요" in rewritten
    assert "할인 카드나 정기권" in rewritten


def test_coerce_ratio_context_message_keeps_safe_feedback() -> None:
    """비중을 단정하지 않은 안전한 피드백은 그대로 유지하는지 검증한다."""
    original = "교통 지출은 오늘 한 번 발생했으니 반복 여부를 내일 확인해보세요."

    rewritten = coerce_ratio_context_message(
        message=original,
        warnings=[{"category": "교통", "current_ratio_percent": 87.8}],
        period_label="오늘",
        mission="",
    )

    assert rewritten == original
