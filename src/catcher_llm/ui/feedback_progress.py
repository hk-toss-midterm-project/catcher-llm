from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from json import dumps
from time import perf_counter, time
from typing import Protocol
from uuid import uuid4

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


@dataclass(frozen=True)
class FeedbackProgressStep:
    """피드백 생성 진행 바에 표시할 단계 키와 화면 이름을 표현한다."""

    step_key: str
    step_name: str


class FeedbackTimingRecordLike(Protocol):
    """진행 바 갱신에 필요한 타이밍 기록 필드만 정의한다."""

    step_key: str
    step_name: str
    elapsed_seconds: float
    status: str
    error: str | None


DAILY_FEEDBACK_PROGRESS_STEPS: tuple[FeedbackProgressStep, ...] = (
    FeedbackProgressStep("daily_analysis", "일일 소비 분석 JSON 생성"),
    FeedbackProgressStep("parse_daily_analysis", "일일 분석 모델 검증"),
    FeedbackProgressStep("user_profile", "사용자 프로필 조회"),
    FeedbackProgressStep("interpretation_chain", "소비 해석 체인 실행"),
    FeedbackProgressStep("retrieval_queries", "RAG 검색 질의 생성"),
    FeedbackProgressStep("rag_retrieval", "RAG 문서 검색"),
    FeedbackProgressStep("memory_context", "피드백 메모리 조회"),
    FeedbackProgressStep("feedback_chain", "최종 피드백 체인 실행"),
    FeedbackProgressStep("save_session", "피드백 세션 저장"),
    FeedbackProgressStep("refresh_memory", "장기 메모리 요약 갱신"),
)
WEEKLY_FEEDBACK_PROGRESS_STEPS: tuple[FeedbackProgressStep, ...] = (
    FeedbackProgressStep("weekly_analysis", "주간 소비 분석 JSON 생성"),
    FeedbackProgressStep("parse_weekly_analysis", "주간 분석 모델 검증"),
    FeedbackProgressStep("user_profile", "사용자 프로필 조회"),
    FeedbackProgressStep("interpretation_chain", "소비 해석 체인 실행"),
    FeedbackProgressStep("retrieval_queries", "RAG 검색 질의 생성"),
    FeedbackProgressStep("rag_retrieval", "RAG 문서 검색"),
    FeedbackProgressStep("memory_context", "피드백 메모리 조회"),
    FeedbackProgressStep("feedback_chain", "최종 피드백 체인 실행"),
    FeedbackProgressStep("save_session", "피드백 세션 저장"),
    FeedbackProgressStep("refresh_memory", "장기 메모리 요약 갱신"),
)
MONTHLY_FEEDBACK_PROGRESS_STEPS: tuple[FeedbackProgressStep, ...] = (
    FeedbackProgressStep("monthly_analysis", "월간 소비 분석 JSON 생성"),
    FeedbackProgressStep("parse_monthly_analysis", "월간 분석 모델 검증"),
    FeedbackProgressStep("user_profile", "사용자 프로필 조회"),
    FeedbackProgressStep("interpretation_chain", "소비 해석 체인 실행"),
    FeedbackProgressStep("retrieval_queries", "RAG 검색 질의 생성"),
    FeedbackProgressStep("rag_retrieval", "RAG 문서 검색"),
    FeedbackProgressStep("memory_context", "피드백 메모리 조회"),
    FeedbackProgressStep("feedback_chain", "최종 피드백 체인 실행"),
    FeedbackProgressStep("save_session", "피드백 세션 저장"),
    FeedbackProgressStep("refresh_memory", "장기 메모리 요약 갱신"),
)


def _step_name_by_key(steps: Sequence[FeedbackProgressStep]) -> dict[str, str]:
    """진행 단계 키로 화면 표시용 단계 이름을 찾을 수 있는 맵을 만든다."""
    return {step.step_key: step.step_name for step in steps}


def _find_next_step_name(
    steps: Sequence[FeedbackProgressStep],
    completed_step_keys: set[str],
) -> str | None:
    """아직 완료되지 않은 다음 진행 단계 이름을 반환한다."""
    for step in steps:
        if step.step_key not in completed_step_keys:
            return step.step_name
    return None


def _progress_percent(completed_count: int, total_count: int) -> int:
    """완료 단계 수를 Streamlit progress 정수 퍼센트로 변환한다."""
    if total_count <= 0:
        return 100
    return min(100, max(0, round(completed_count / total_count * 100)))


def _format_elapsed_seconds(seconds: float) -> str:
    """초 단위 소요 시간을 진행 UI에 표시할 문자열로 변환한다."""
    return f"{seconds:.3f}초"


def _build_progress_text(
    *,
    completed_count: int,
    total_count: int,
    current_step_name: str,
) -> str:
    """진행 바에 표시할 완료 단계 수와 현재 단계 문구를 만든다."""
    return f"{completed_count}/{total_count}단계 완료 · 현재 단계: {current_step_name}"


def _build_total_elapsed_html(
    *,
    element_id: str,
    started_at_epoch_ms: int,
    final_elapsed_seconds: float | None = None,
) -> str:
    """총 진행 시간을 브라우저에서 실시간 갱신하거나 최종 값으로 고정하는 HTML을 만든다."""
    displayed_elapsed_seconds = final_elapsed_seconds or 0.0
    safe_element_id = escape(element_id, quote=True)
    safe_display_text = escape(_format_elapsed_seconds(displayed_elapsed_seconds))

    script = ""
    if final_elapsed_seconds is None:
        element_id_json = dumps(element_id)
        script = f"""
<script>
(() => {{
  const target = document.getElementById({element_id_json});
  const startedAt = {started_at_epoch_ms};
  const formatSeconds = (seconds) => `${{seconds.toFixed(3)}}초`;
  const render = () => {{
    if (!target || !document.body.contains(target)) {{
      return false;
    }}
    target.textContent = formatSeconds((Date.now() - startedAt) / 1000);
    return true;
  }};
  render();
  const timer = setInterval(() => {{
    if (!render()) {{
      clearInterval(timer);
    }}
  }}, 100);
}})();
</script>"""

    return (
        '<div style="font-size: 0.875rem; color: rgba(49, 51, 63, 0.78); '
        'line-height: 1.5; margin: 0.15rem 0 0.35rem 0;">'
        f'총 진행 시간: <span id="{safe_element_id}">{safe_display_text}</span>'
        f"</div>{script}"
    )


def _render_live_elapsed_timer(
    placeholder: DeltaGenerator,
    *,
    element_id: str,
    started_at_epoch_ms: int,
    final_elapsed_seconds: float | None = None,
) -> None:
    """총 진행 시간 타이머 HTML을 Streamlit placeholder에 렌더링한다."""
    placeholder.html(
        _build_total_elapsed_html(
            element_id=element_id,
            started_at_epoch_ms=started_at_epoch_ms,
            final_elapsed_seconds=final_elapsed_seconds,
        ),
        unsafe_allow_javascript=True,
    )


def _elapsed_since_started(started_at: float) -> float:
    """진행 시작 시점부터 현재까지의 총 소요 시간을 계산한다."""
    return perf_counter() - started_at


def create_feedback_progress_callback(
    steps: Sequence[FeedbackProgressStep],
) -> Callable[[FeedbackTimingRecordLike], None]:
    """Streamlit 진행 바와 현재 단계 텍스트를 갱신하는 타이밍 콜백을 생성한다."""
    if not steps:
        progress_bar = st.progress(100, text="피드백 생성 단계가 없습니다.")
        status_placeholder = st.empty()

        def _noop_callback(record: FeedbackTimingRecordLike) -> None:
            """단계 목록이 없을 때 콜백 입력을 무시하고 완료 상태를 유지한다."""
            progress_bar.progress(100, text="피드백 생성 단계가 없습니다.")
            status_placeholder.caption(f"마지막 기록: {record.step_name}")

        return _noop_callback

    total_count = len(steps)
    step_names = _step_name_by_key(steps)
    completed_step_keys: set[str] = set()
    progress_started_at = perf_counter()
    timer_started_at_epoch_ms = round(time() * 1000)
    timer_element_id = f"feedback-total-elapsed-{uuid4().hex}"
    progress_bar = st.progress(
        0,
        text=_build_progress_text(
            completed_count=0,
            total_count=total_count,
            current_step_name=steps[0].step_name,
        ),
    )
    status_placeholder = st.empty()
    status_placeholder.caption(f"현재 단계: {steps[0].step_name}")
    elapsed_placeholder = st.empty()
    _render_live_elapsed_timer(
        elapsed_placeholder,
        element_id=timer_element_id,
        started_at_epoch_ms=timer_started_at_epoch_ms,
    )

    def _update_progress(record: FeedbackTimingRecordLike) -> None:
        """서비스 타이밍 기록을 받아 진행률과 현재 단계 표시를 갱신한다."""
        completed_step_keys.add(record.step_key)
        completed_count = len(completed_step_keys)
        current_step_name = step_names.get(record.step_key, record.step_name)
        progress_value = _progress_percent(completed_count, total_count)

        if record.status == "error":
            final_elapsed_seconds = _elapsed_since_started(progress_started_at)
            progress_bar.progress(
                progress_value,
                text=_build_progress_text(
                    completed_count=completed_count,
                    total_count=total_count,
                    current_step_name=f"{current_step_name} 실패",
                ),
            )
            _render_live_elapsed_timer(
                elapsed_placeholder,
                element_id=timer_element_id,
                started_at_epoch_ms=timer_started_at_epoch_ms,
                final_elapsed_seconds=final_elapsed_seconds,
            )
            status_placeholder.error(
                f"현재 단계: {current_step_name} 실패"
                + (f" · {record.error}" if record.error else "")
            )
            return

        next_step_name = _find_next_step_name(steps, completed_step_keys)
        if next_step_name is None:
            final_elapsed_seconds = _elapsed_since_started(progress_started_at)
            progress_bar.progress(
                100,
                text=_build_progress_text(
                    completed_count=total_count,
                    total_count=total_count,
                    current_step_name="모든 피드백 생성 단계 완료",
                ),
            )
            _render_live_elapsed_timer(
                elapsed_placeholder,
                element_id=timer_element_id,
                started_at_epoch_ms=timer_started_at_epoch_ms,
                final_elapsed_seconds=final_elapsed_seconds,
            )
            status_placeholder.success(
                f"현재 단계: 모든 피드백 생성 단계 완료 · 총 {_format_elapsed_seconds(final_elapsed_seconds)}"
            )
            return

        progress_bar.progress(
            progress_value,
            text=_build_progress_text(
                completed_count=completed_count,
                total_count=total_count,
                current_step_name=next_step_name,
            ),
        )
        status_placeholder.caption(f"현재 단계: {next_step_name}")

    return _update_progress


__all__ = [
    "DAILY_FEEDBACK_PROGRESS_STEPS",
    "MONTHLY_FEEDBACK_PROGRESS_STEPS",
    "WEEKLY_FEEDBACK_PROGRESS_STEPS",
    "FeedbackProgressStep",
    "FeedbackTimingRecordLike",
    "create_feedback_progress_callback",
]
