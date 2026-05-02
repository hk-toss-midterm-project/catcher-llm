from __future__ import annotations

from catcher_llm.ui.feedback_progress import (
    _build_progress_text,
    _build_total_elapsed_html,
    _format_elapsed_seconds,
)


def test_feedback_progress_formats_elapsed_seconds() -> None:
    """피드백 진행 UI가 초 단위를 세 자리 소수로 표시하는지 검증한다."""
    assert _format_elapsed_seconds(1.23456) == "1.235초"
    assert _format_elapsed_seconds(0) == "0.000초"


def test_feedback_progress_text_excludes_elapsed_seconds() -> None:
    """피드백 진행 바 문구가 완료 단계와 다음 단계만 표시하는지 검증한다."""
    assert (
        _build_progress_text(
            completed_count=2,
            total_count=10,
            current_step_name="RAG 문서 검색",
        )
        == "2/10단계 완료 · 현재 단계: RAG 문서 검색"
    )


def test_feedback_progress_live_timer_html_updates_total_seconds() -> None:
    """피드백 진행 UI가 브라우저 타이머로 총 진행 초만 실시간 갱신하는지 검증한다."""
    html = _build_total_elapsed_html(
        element_id="feedback-elapsed-test",
        started_at_epoch_ms=1234,
    )

    assert "총 진행 시간:" in html
    assert "Date.now()" in html
    assert "setInterval" in html
    assert "1234" in html
    assert "단계별" not in html


def test_feedback_progress_final_elapsed_html_is_static_total_only() -> None:
    """피드백 완료 UI가 단계별 목록 없이 최종 총 진행 초만 표시하는지 검증한다."""
    html = _build_total_elapsed_html(
        element_id="feedback-elapsed-test",
        started_at_epoch_ms=1234,
        final_elapsed_seconds=3.4567,
    )

    assert "총 진행 시간:" in html
    assert "3.457초" in html
    assert "setInterval" not in html
    assert "단계별" not in html
