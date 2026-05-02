from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal


@dataclass(frozen=True)
class FeedbackTimingRecord:
    """피드백 생성 단계별 실행 시간과 상태를 담는 공통 기록이다."""

    step_key: str
    step_name: str
    elapsed_seconds: float
    status: Literal["success", "error"] = "success"
    detail: str | None = None
    error: str | None = None


FeedbackTimingCallback = Callable[[FeedbackTimingRecord], None]


def emit_feedback_timing(
    callback: FeedbackTimingCallback | None,
    record: FeedbackTimingRecord,
) -> None:
    """타이밍 수집 콜백을 호출하되 콜백 오류가 피드백 생성을 막지 않게 한다."""
    if callback is None:
        return
    try:
        callback(record)
    except Exception:
        return


def run_timed_feedback_step[T](
    *,
    step_key: str,
    step_name: str,
    operation: Callable[[], T],
    timing_callback: FeedbackTimingCallback | None = None,
    detail: str | None = None,
) -> T:
    """피드백 생성의 한 단계를 실행하고 성공/실패 소요 시간을 기록한다."""
    start = perf_counter()
    try:
        result = operation()
    except Exception as exc:
        emit_feedback_timing(
            timing_callback,
            FeedbackTimingRecord(
                step_key=step_key,
                step_name=step_name,
                elapsed_seconds=perf_counter() - start,
                status="error",
                detail=detail,
                error=str(exc),
            ),
        )
        raise

    emit_feedback_timing(
        timing_callback,
        FeedbackTimingRecord(
            step_key=step_key,
            step_name=step_name,
            elapsed_seconds=perf_counter() - start,
            status="success",
            detail=detail,
        ),
    )
    return result


__all__ = [
    "FeedbackTimingCallback",
    "FeedbackTimingRecord",
    "emit_feedback_timing",
    "run_timed_feedback_step",
]
