import html

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.user_data_service import authenticate_user, ensure_user_database

settings = get_settings()

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


def logout() -> None:
    """현재 로그인 세션 정보를 지우고 로그인 화면으로 되돌린다."""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_profile = None
    st.rerun()


def render_profile_text_panel(
    *,
    title: str,
    text: object,
    empty_message: str,
) -> None:
    """긴 프로필 텍스트를 별도 패널로 읽기 쉽게 렌더링한다."""
    if text is None or str(text).strip() == "":
        st.info(empty_message)
        return

    escaped_text = html.escape(str(text).strip()).replace("\n", "<br>")
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
        st.metric("연소득", profile["income"])
        st.metric("personal_score", profile.get("personal_score") or 0)

    st.markdown("---")
    persona_text = profile.get("persona")
    render_profile_text_panel(
        title="🧩 나의 페르소나",
        text=persona_text,
        empty_message="아직 등록된 페르소나가 없습니다.",
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

login_pg = st.Page(login_page, title="로그인", url_path="login", default=True)
signup_pg = st.Page("pages/00_user_signup.py", title="회원 가입", icon="🧾", url_path="signup")
profile_pg = st.Page(profile_page, title="프로필", icon="🙀", url_path="profile", default=True)
upload_pg = st.Page("pages/01_csv_upload.py", title="CSV 업로드", icon="📥", url_path="upload")
report_pg = st.Page("pages/02_report.py", title="리포트 조회", icon="📫", url_path="report")
group_pg = st.Page(
    "pages/03_group_competition.py",
    title="그룹 경쟁",
    icon="🏁",
    url_path="group-competition",
)

if not st.session_state.logged_in:
    pg = st.navigation([login_pg, signup_pg], position="hidden")
else:
    pg = st.navigation([profile_pg, upload_pg, report_pg, group_pg], position="hidden")

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
                    st.rerun()

        st.markdown("---")
        st.page_link(signup_pg, label="회원 가입", icon="🧾")

    else:
        if profile := st.session_state.user_profile:
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

        if st.button("로그아웃", width="stretch"):
            logout()

pg.run()
