import html
import json
from urllib.parse import quote, unquote

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.user_data_service import authenticate_user, ensure_user_database
from catcher_llm.utils import format_income_to_10k_won

settings = get_settings()

LOGIN_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30
LOGIN_USER_ID_COOKIE = "catcher_login_user_id"
LOGIN_USER_NAME_COOKIE = "catcher_login_user_name"

st.set_page_config(
    page_title="Catcher",
    page_icon="💯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def init_user_database() -> None:
    """앱 시작 시 사용자 SQLite 데이터베이스를 한 번 초기화한다."""
    ensure_user_database(settings=settings)


def _render_cookie_script(cookie_assignments: list[str]) -> None:
    """브라우저 쿠키 변경 스크립트를 숨김 컴포넌트로 실행한다."""
    script_lines = [
        f"document.cookie = {json.dumps(assignment)};" for assignment in cookie_assignments
    ]
    st.iframe(f"<script>{''.join(script_lines)}</script>", height=1)


def persist_login_cookie(user_id: int, user_name: str) -> None:
    """로그인 성공 후 새로고침 복원을 위한 사용자 식별 쿠키를 저장한다."""
    cookie_options = f"path=/; max-age={LOGIN_COOKIE_MAX_AGE_SECONDS}; SameSite=Lax"
    _render_cookie_script(
        [
            f"{LOGIN_USER_ID_COOKIE}={user_id}; {cookie_options}",
            f"{LOGIN_USER_NAME_COOKIE}={quote(user_name, safe='')}; {cookie_options}",
        ]
    )


def clear_login_cookie() -> None:
    """로그아웃 또는 쿠키 인증 실패 시 로그인 쿠키를 즉시 만료한다."""
    expired_cookie_options = "path=/; max-age=0; SameSite=Lax"
    _render_cookie_script(
        [
            f"{LOGIN_USER_ID_COOKIE}=; {expired_cookie_options}",
            f"{LOGIN_USER_NAME_COOKIE}=; {expired_cookie_options}",
        ]
    )


def _read_context_cookie(cookie_name: str) -> str | None:
    """Streamlit 요청 컨텍스트에서 지정한 쿠키 값을 문자열로 읽는다."""
    try:
        cookie_value = st.context.cookies.get(cookie_name)
    except (AttributeError, RuntimeError):
        return None

    if cookie_value is None:
        return None

    normalized_value = str(cookie_value).strip()
    if normalized_value == "":
        return None

    return normalized_value


def restore_login_from_cookie() -> None:
    """새 Streamlit 세션이 시작될 때 쿠키 기반 로그인 상태를 복원한다."""
    if st.session_state.logged_in:
        return

    raw_user_id = _read_context_cookie(LOGIN_USER_ID_COOKIE)
    raw_user_name = _read_context_cookie(LOGIN_USER_NAME_COOKIE)
    if raw_user_id is None or raw_user_name is None:
        return

    try:
        user_id = int(raw_user_id)
    except ValueError:
        clear_login_cookie()
        return

    user_name = unquote(raw_user_name)
    profile = authenticate_user(user_id, user_name, settings=settings)
    if profile is None:
        clear_login_cookie()
        return

    st.session_state.logged_in = True
    st.session_state.user_id = user_id
    st.session_state.user_profile = profile


def apply_pending_logout() -> bool:
    """로그아웃 버튼 요청을 쿠키 자동 복원보다 먼저 반영한다."""
    if st.session_state.get("logout_requested") is not True:
        return False

    clear_login_cookie()
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_profile = None
    st.session_state.logout_requested = False
    return True


def logout() -> None:
    """로그아웃 버튼 클릭을 다음 앱 실행 시작부에서 처리하도록 표시한다."""
    st.session_state.logout_requested = True


def render_profile_text_panel(
    *,
    title: str,
    text: object,
    empty_message: str,
    collapsible: bool = False,
    expanded: bool = False,
) -> None:
    """긴 프로필 텍스트를 별도 패널 또는 접이식 패널로 읽기 쉽게 렌더링한다."""
    if text is None or str(text).strip() == "":
        if collapsible:
            with st.expander(title, expanded=expanded):
                st.info(empty_message)
        else:
            st.info(empty_message)
        return

    escaped_text = html.escape(str(text).strip()).replace("\n", "<br>")
    if collapsible:
        with st.expander(title, expanded=expanded):
            st.markdown(
                f"""
                <div style="background-color:#F5F9FF; border:1px solid #D6E4F0; padding:22px 24px; border-radius:18px;">
                    <div style="margin:0; font-size:18px; color:#1D4ED8; font-weight:700; line-height:1.6; white-space:pre-wrap; overflow-wrap:anywhere;">{escaped_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    st.markdown(
        f"""
        <div style="background-color:#F5F9FF; border:1px solid #D6E4F0; padding:22px 24px; border-radius:18px;">
            <div style="margin:0 0 10px 0; font-size:14px; color:#6B7280; font-weight:600;">{title}</div>
            <div style="margin:0; font-size:18px; color:#1D4ED8; font-weight:700; line-height:1.6; white-space:pre-wrap; overflow-wrap:anywhere;">{escaped_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def login_page() -> None:
    """로그인 전 메인 랜딩 메시지를 보여준다."""
    st.title("Catcher 소비 분석 서비스")
    st.info("왼쪽 사이드바에서 User ID와 이름을 입력하거나 회원 가입을 진행해 주세요.")


def profile_page() -> None:
    """로그인한 사용자의 기본 프로필과 목표 정보를 보여준다."""
    profile = st.session_state.user_profile
    assert profile is not None

    current_user_id = st.session_state.user_id
    if current_user_id is not None:
        refreshed_profile = authenticate_user(
            int(current_user_id),
            str(profile["name"]),
            settings=settings,
        )
        if refreshed_profile is not None:
            st.session_state.user_profile = refreshed_profile
            profile = refreshed_profile

    st.title("🙀 사용자 프로필")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("이름", profile["name"])
        st.metric("나이", f"{profile['age']}세")
        st.metric("성별", profile["gender"])

    with col2:
        st.metric("직업", profile["job"])
        st.metric("지역", profile["region"])
        st.metric("연소득", format_income_to_10k_won(profile.get("income")))
        st.metric("개인 점수", profile.get("personal_score") or 0)

    st.markdown("---")
    persona_text = profile.get("persona")
    render_profile_text_panel(
        title="🧩 나의 페르소나",
        text=persona_text,
        empty_message="아직 등록된 페르소나가 없습니다.",
        collapsible=True,
    )

    st.markdown("---")
    saving_goal_text = profile.get("saving_goal_text")
    render_profile_text_panel(
        title="🎯 나의 절약 목표",
        text=saving_goal_text,
        empty_message="아직 등록된 절약 목표가 없습니다.",
    )

    st.markdown("---")
    st.markdown(
        """
<div style="
    background-color:#E8F2FF;
    color:#1D4ED8;
    padding:16px;
    border-radius:12px;
    font-weight:500;
">
    로그인 완료! 왼쪽 메뉴에서 CSV 업로드, 리포트 조회, 그룹 경쟁을 선택해 보세요.
</div>
        """,
        unsafe_allow_html=True,
    )


init_user_database()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "logout_requested" not in st.session_state:
    st.session_state.logout_requested = False

logout_was_applied = apply_pending_logout()
if not logout_was_applied:
    restore_login_from_cookie()

login_pg = st.Page(login_page, title="로그인", url_path="login", default=True)
signup_pg = st.Page("pages/00_user_signup.py", title="회원 가입", icon="🧾", url_path="signup")
profile_pg = st.Page(profile_page, title="프로필", icon="🙀", url_path="profile", default=True)
upload_pg = st.Page("pages/01_csv_upload.py", title="CSV 업로드", icon="📥", url_path="upload")
report_pg = st.Page("pages/02_report.py", title="리포트 조회", icon="📫", url_path="report")
daily_report_pg = st.Page(
    "pages/04_daily_report.py",
    title="일일 리포트",
    icon="📝",
    url_path="daily-report",
)
weekly_report_pg = st.Page(
    "pages/05_weekly_report.py",
    title="주간 리포트",
    icon="🗓️",
    url_path="weekly-report",
)
monthly_report_pg = st.Page(
    "pages/06_monthly_report.py",
    title="월간 리포트",
    icon="📈",
    url_path="monthly-report",
)
group_pg = st.Page(
    "pages/03_group_competition.py",
    title="그룹 경쟁",
    icon="🏁",
    url_path="group-competition",
)

if not st.session_state.logged_in:
    pg = st.navigation([login_pg, signup_pg], position="hidden")
else:
    pg = st.navigation(
        [
            profile_pg,
            upload_pg,
            report_pg,
            daily_report_pg,
            weekly_report_pg,
            monthly_report_pg,
            group_pg,
        ],
        position="hidden",
    )

with st.sidebar:
    st.title("💯 Catcher")

    if not st.session_state.logged_in:
        st.subheader("로그인")

        input_user_id = st.text_input("User ID 입력")
        input_user_name = st.text_input("이름 입력")

        if st.button("로그인", width="stretch"):
            if input_user_id.strip() == "" or input_user_name.strip() == "":
                st.warning("User ID와 이름을 모두 입력해 주세요.")
            else:
                try:
                    user_id = int(input_user_id)
                except ValueError:
                    st.error("User ID는 숫자로 입력해 주세요.")
                    st.stop()

                profile = authenticate_user(
                    user_id,
                    input_user_name.strip(),
                    settings=settings,
                )

                if profile is None:
                    st.error("일치하는 사용자가 없습니다.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_profile = profile
                    persist_login_cookie(user_id, input_user_name.strip())
                    st.rerun()

        st.markdown("---")
        st.page_link(signup_pg, label="회원 가입", icon="🧾")

    else:
        if profile := st.session_state.user_profile:
            if st.session_state.user_id is not None:
                persist_login_cookie(int(st.session_state.user_id), str(profile["name"]))

            st.markdown(
                f"""
<div style="
    background-color:#E8F2FF;
    color:#1D4ED8;
    padding:12px;
    border-radius:10px;
    font-weight:600;
">
    {profile["name"]}님
</div>
""",
                unsafe_allow_html=True,
            )
            st.caption(f"User ID: {st.session_state.user_id}")

        st.markdown("---")
        st.page_link(profile_pg, label="프로필", icon="🙀")
        st.page_link(upload_pg, label="CSV 업로드", icon="📥")
        st.page_link(report_pg, label="리포트 조회", icon="📫")
        st.page_link(group_pg, label="그룹 경쟁", icon="🏁")

        st.markdown("---")

        st.button("로그아웃", width="stretch", on_click=logout)

pg.run()
