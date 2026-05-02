from __future__ import annotations

import re
from collections.abc import MutableMapping
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
import streamlit_date_picker
from streamlit_date_picker import PickerType, date_picker

_DATE_PICKER_COMPONENT_HEIGHT = "400px"
DEFAULT_CALENDAR_DATE = date(2026, 4, 1)
DEFAULT_CALENDAR_MONTH = "2026-04"
_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})$")
_WEEK_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<week>\d{1,2})(?!-\d)")
_DATE_PICKER_TRIGGER_CSS = """
<style>
div[data-testid="stPopover"] > div[aria-haspopup="true"] > button[data-testid="stPopoverButton"] {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 2.5rem !important;
    padding: 0.25rem 0.75rem !important;
    border: 1px solid color-mix(in srgb, var(--text-color) 20%, transparent) !important;
    border-radius: 0.5rem !important;
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    font-weight: 400 !important;
    box-shadow: none !important;
}

div[data-testid="stPopover"] > div[aria-haspopup="true"] > button[data-testid="stPopoverButton"]:hover {
    border-color: color-mix(in srgb, var(--text-color) 35%, transparent) !important;
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}

div[data-testid="stPopover"] > div[aria-haspopup="true"] > button[data-testid="stPopoverButton"]:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 1px var(--primary-color) !important;
}

div[data-testid="stPopover"] > div[aria-haspopup="true"] > button[data-testid="stPopoverButton"] p {
    width: 100% !important;
    margin: 0 !important;
    text-align: left !important;
    color: var(--text-color) !important;
    font-weight: 400 !important;
}

label[data-testid="stWidgetLabel"].catcher-date-picker-label {
    display: block;
    margin-bottom: 4px;
    color: var(--text-color) !important;
}

label[data-testid="stWidgetLabel"].catcher-date-picker-label p {
    margin: 0;
    color: var(--text-color) !important;
}

div[data-testid="stVerticalBlock"]:has(
    > div[data-testid="stElementContainer"] label[data-testid="stWidgetLabel"].catcher-date-picker-label
):has(> div[data-testid="stLayoutWrapper"] div[data-testid="stPopover"]) {
    row-gap: 4px;
    gap: 4px;
}

div[data-testid="stVerticalBlock"]:has(
    > div[data-testid="stElementContainer"] label[data-testid="stWidgetLabel"].catcher-date-picker-label
) > div[data-testid="stElementContainer"]:has(label[data-testid="stWidgetLabel"].catcher-date-picker-label) {
    margin-bottom: 0;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) {
    align-items: stretch;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-self: stretch;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) div[data-testid="stMetric"] {
    min-height: 38px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
</style>
"""
_DATE_PICKER_POPOVER_CSS = """
<style>
div[data-baseweb="popover"] {
    min-height: 400px !important;
    overflow: visible !important;
}

div[data-baseweb="popover"] > div {
    min-height: 400px !important;
    overflow: visible !important;
}

div[data-baseweb="popover"] div[style*="height: 50px"],
div[data-testid="stPopoverBody"] div[style*="height: 50px"] {
    min-height: 400px !important;
    height: 400px !important;
    overflow: visible !important;
}

div[data-testid="stPopoverBody"] div[data-testid="stCustomComponentV1"],
div[data-testid="stPopoverBody"] div[data-testid="stCustomComponentV1"] iframe {
    min-height: 400px !important;
    height: 400px !important;
}
</style>
"""


def render_date_picker_styles() -> None:
    """날짜 선택기 트리거 박스 전역 스타일을 현재 페이지에 주입한다."""
    st.markdown(_DATE_PICKER_TRIGGER_CSS, unsafe_allow_html=True)


def _patch_date_picker_bundle_source(source: str) -> tuple[str, bool]:
    """streamlit-date-picker 번들 소스의 높이 하드코딩을 400px로 치환한다."""
    patched_source = source
    for height in ("50px", "200px", "350px", "360px", "520px"):
        patched_source = patched_source.replace(
            f'height:"{height}"', f'height:"{_DATE_PICKER_COMPONENT_HEIGHT}"'
        )
        patched_source = patched_source.replace(
            f"height:'{height}'", f"height:'{_DATE_PICKER_COMPONENT_HEIGHT}'"
        )
    return patched_source, patched_source != source


def _date_picker_bundle_paths() -> list[Path]:
    """streamlit-date-picker 패키지의 빌드된 JS 번들 경로 목록을 반환한다."""
    package_dir = Path(streamlit_date_picker.__file__).resolve().parent
    return sorted((package_dir / "frontend" / "build" / "static" / "js").glob("main.*.js"))


@lru_cache(maxsize=1)
def _ensure_date_picker_bundle_height() -> None:
    """streamlit-date-picker 내부 컨테이너 높이를 한 번만 400px로 보정한다."""
    for bundle_path in _date_picker_bundle_paths():
        try:
            source = bundle_path.read_text(encoding="utf-8")
        except OSError:
            continue

        patched_source, changed = _patch_date_picker_bundle_source(source)
        if not changed:
            continue

        try:
            bundle_path.write_text(patched_source, encoding="utf-8")
        except OSError:
            continue


def _parse_iso_date(value: object) -> date | None:
    """ISO 날짜·날짜시간 문자열을 date 값으로 안전하게 변환한다."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    month_match = _MONTH_PATTERN.match(normalized)
    if month_match is not None:
        return date(int(month_match.group("year")), int(month_match.group("month")), 1)

    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return date.fromisoformat(normalized[:10])
        except ValueError:
            return None


def _parse_picker_week(value: object) -> date | None:
    """streamlit-date-picker week 반환 문자열에서 UI 주간의 일요일 날짜를 추출한다."""
    if not isinstance(value, str):
        return None

    week_match = _WEEK_PATTERN.match(value.strip())
    if week_match is None:
        return None

    try:
        return _start_of_sunday_week(int(week_match.group("year")), int(week_match.group("week")))
    except ValueError:
        return None


def _start_of_sunday_week(year: int, week_number: int) -> date:
    """일요일 시작 달력에서 특정 연도·주차의 시작일을 계산한다."""
    if week_number < 1:
        msg = "week_number must be positive"
        raise ValueError(msg)

    first_day = date(year, 1, 1)
    first_week_start = first_day - timedelta(days=(first_day.weekday() + 1) % 7)
    return first_week_start + timedelta(weeks=week_number - 1)


def _coerce_date(value: object, default: date) -> date:
    """streamlit-date-picker 반환값을 date로 변환하고 비어 있으면 기본 날짜를 반환한다."""
    return _parse_iso_date(value) or default


def _coerce_month(value: object, default: date) -> str:
    """streamlit-date-picker 반환값을 YYYY-MM 문자열로 변환하고 비어 있으면 기본 월을 반환한다."""
    picked_date = _coerce_date(value, default)
    return f"{picked_date.year:04d}-{picked_date.month:02d}"


def _coerce_week_range(value: object, default_start: date) -> tuple[date, date]:
    """streamlit-date-picker 반환값을 일요일 시작 주간 범위로 변환한다."""
    picked_date = _parse_picker_week(value) or _coerce_date(value, default_start)
    week_start = picked_date - timedelta(days=(picked_date.weekday() + 1) % 7)
    return week_start, week_start + timedelta(days=6)


def _coerce_picker_date(picker_type: PickerType, value: object, default: date) -> date:
    """picker 타입별 반환 문자열을 세션 저장용 date 값으로 변환한다."""
    return _parse_picker_date(picker_type, value) or default


def _parse_picker_date(picker_type: PickerType, value: object) -> date | None:
    """picker 타입별 반환 문자열을 date 값으로 변환하고 실패하면 None을 반환한다."""
    if picker_type == PickerType.week:
        return _parse_picker_week(value) or _parse_iso_date(value)
    return _parse_iso_date(value)


def _as_datetime(value: date) -> datetime:
    """date 값을 streamlit-date-picker 기본값에 맞는 datetime으로 변환한다."""
    return datetime.combine(value, time.min)


def _session_key(key: str) -> str:
    """날짜 선택값을 저장할 Streamlit session_state 키를 만든다."""
    return f"catcher_date_picker_{key}"


def _picker_key(key: str) -> str:
    """popover 내부 streamlit-date-picker 컴포넌트 키를 만든다."""
    return f"{_session_key(key)}_component"


def _pending_session_key(key: str) -> str:
    """popover에서 적용 전 선택값을 저장할 Streamlit session_state 키를 만든다."""
    return f"{_session_key(key)}_pending"


def _popover_session_key(key: str) -> str:
    """적용 후 popover를 닫기 위해 key를 갱신하는 토큰 세션 키를 만든다."""
    return f"{_session_key(key)}_popover_token"


def _get_state_date(key: str, default: date) -> date:
    """세션에 저장된 선택 날짜를 읽고 없거나 잘못되면 기본 날짜로 초기화한다."""
    state_key = _session_key(key)
    stored_date = _parse_iso_date(st.session_state.get(state_key))
    if stored_date is None:
        st.session_state[state_key] = default.isoformat()
        return default
    return stored_date


def _set_pending_state_date(
    state: MutableMapping[str, object], key: str, selected_date: date
) -> None:
    """적용 전 picker 선택 날짜를 주어진 세션 상태에 저장한다."""
    state[_pending_session_key(key)] = selected_date.isoformat()


def _apply_pending_state_date(state: MutableMapping[str, object], key: str, default: date) -> date:
    """pending 선택 날짜를 확정 선택값으로 저장하고 적용된 날짜를 반환한다."""
    selected_date = _parse_iso_date(state.get(_pending_session_key(key))) or default
    state[_session_key(key)] = selected_date.isoformat()
    state[_pending_session_key(key)] = selected_date.isoformat()
    return selected_date


def _update_pending_state_from_picker(
    state: MutableMapping[str, object],
    key: str,
    picker_type: PickerType,
    picker_value: object,
    default: date,
) -> date:
    """picker가 유효한 값을 반환할 때만 pending 선택값을 갱신하고 현재 pending 날짜를 반환한다."""
    picked_date = _parse_picker_date(picker_type, picker_value)
    if picked_date is not None:
        _set_pending_state_date(state, key, picked_date)
        return picked_date

    return _parse_iso_date(state.get(_pending_session_key(key))) or default


def _update_popover_close_token(state: MutableMapping[str, object], key: str) -> int:
    """popover key 갱신용 토큰을 증가시키고 새 토큰 값을 반환한다."""
    current_value = state.get(_popover_session_key(key))
    current_token = current_value if isinstance(current_value, int) else 0
    next_token = current_token + 1
    state[_popover_session_key(key)] = next_token
    return next_token


def _render_picker_label(label: str) -> None:
    """날짜 선택기 타이틀을 Streamlit widget label과 같은 HTML 구조로 렌더링한다."""
    safe_label = escape(label)
    st.html(
        '<label data-testid="stWidgetLabel" for="" '
        'class="catcher-date-picker-label st-emotion-cache-1weic72 e1kt9bl70">'
        '<span aria-hidden="true">'
        '<div data-testid="stMarkdownContainer" class="st-emotion-cache-rsr9ey e18fvw670">'
        f"<p>{safe_label}</p>"
        "</div>"
        "</span>"
        "</label>"
    )


def _render_picker_popover(
    *,
    label: str,
    value_label: str,
    picker_type: PickerType,
    selected_date: date,
    key: str,
) -> None:
    """현재 선택값을 요약하고 popover 안에서 streamlit-date-picker를 렌더링한다."""
    _render_picker_label(label)
    popover_token = st.session_state.get(_popover_session_key(key))
    if not isinstance(popover_token, int):
        popover_token = 0

    with st.popover(
        value_label,
        width="stretch",
        key=f"{_session_key(key)}_popover_{popover_token}",
    ):
        st.html(_DATE_PICKER_POPOVER_CSS)
        _ensure_date_picker_bundle_height()
        picked_value = date_picker(
            picker_type=picker_type,
            value=_as_datetime(selected_date),
            key=_picker_key(key),
        )
        picked_date = _update_pending_state_from_picker(
            st.session_state,
            key,
            picker_type,
            picked_value,
            selected_date,
        )
        st.caption(f"선택값: {picked_date}")

        if st.button("적용", key=f"{_session_key(key)}_apply", width="stretch"):
            _apply_pending_state_date(st.session_state, key, picked_date)
            _update_popover_close_token(st.session_state, key)
            st.rerun()


def select_daily_date(label: str, *, default: date, key: str) -> date:
    """일일 분석용 단일 날짜를 popover date picker에서 선택한다."""
    selected_date = _get_state_date(key, default)
    _render_picker_popover(
        label=label,
        value_label=selected_date.isoformat(),
        picker_type=PickerType.date,
        selected_date=selected_date,
        key=key,
    )
    return selected_date


def select_week_range(label: str, *, default_start: date, key: str) -> tuple[date, date]:
    """주간 분석용 주차를 popover week picker에서 선택하고 일~토 범위를 반환한다."""
    selected_date = _get_state_date(key, default_start)
    week_start, week_end = _coerce_week_range(selected_date, default_start)
    _render_picker_popover(
        label=label,
        value_label=f"{week_start} ~ {week_end}",
        picker_type=PickerType.week,
        selected_date=selected_date,
        key=key,
    )
    return week_start, week_end


def select_month(label: str, *, default_month: str, key: str) -> str:
    """월간 분석용 월을 popover month picker에서 선택하고 YYYY-MM 문자열을 반환한다."""
    year, month = (int(part) for part in default_month.split("-", maxsplit=1))
    default_date = date(year, month, 1)
    selected_date = _get_state_date(key, default_date)
    selected_month = _coerce_month(selected_date, default_date)
    month_start = date(selected_date.year, selected_date.month, 1)
    _render_picker_popover(
        label=label,
        value_label=selected_month,
        picker_type=PickerType.month,
        selected_date=month_start,
        key=key,
    )
    return selected_month
