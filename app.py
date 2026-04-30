import streamlit as st
import streamlit.components.v1 as components

from catcher_llm.config.settings import get_settings
from catcher_llm.services.user_data_service import authenticate_user, ensure_user_database

settings = get_settings()

st.set_page_config(
    page_title="Catcher", page_icon="💸", layout="wide", initial_sidebar_state="expanded"
)

# 기본 Streamlit 페이지 메뉴 숨기기
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
    ensure_user_database(settings=settings)


init_user_database()

# -----------------------------
# 세션 초기화
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None


# -----------------------------
# 로그아웃 함수
# -----------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_profile = None
    st.rerun()


# -----------------------------
# 페이지 정의
# -----------------------------
def login_page():
    st.title("Catcher 소비 분석 서비스")
    st.info("왼쪽 사이드바에서 User ID와 이름을 입력하고 로그인해주세요.")


def profile_page():
    profile = st.session_state.user_profile
    assert profile is not None

    st.title("👤 사용자 프로필")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("이름", profile["name"])
        st.metric("나이", f"{profile['age']}세")
        st.metric("성별", profile["gender"])

    with col2:
        st.metric("직업", profile["job"])
        st.metric("지역", profile["region"])
        st.metric("연봉", profile["income"])

    with col3:
        st.metric("최상위 카드 등급", profile["card_grade"])
        st.metric("페르소나", profile["persona"])

    # -----------------------------
    # 절약 목표 (토스 스타일)
    # -----------------------------
    st.markdown("---")

    saving_goal_text = profile.get("saving_goal_text")

    if saving_goal_text:
        goal_html = f"""
        <div style="background-color:#F5F9FF; border:1px solid #D6E4F0; padding:22px 24px; border-radius:18px;">
            <div style="margin:0 0 10px 0; font-size:14px; color:#6B7280; font-weight:600;">🎯 나의 절약 목표</div>
            <div style="margin:0; font-size:18px; color:#1D4ED8; font-weight:700; line-height:1.6;">{saving_goal_text}</div>
        </div>
        """

        components.html(goal_html, height=130)

    else:
        st.info("아직 등록된 절약 목표가 없습니다.")

    # -----------------------------
    # 안내 박스
    # -----------------------------
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
    로그인 완료! 왼쪽 메뉴에서 CSV 업로드 또는 리포트 조회를 선택하세요.
</div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 네비게이션 라우팅
# -----------------------------
profile_pg = st.Page(profile_page, title="프로필", icon="👤", url_path="profile", default=True)
upload_pg = st.Page("pages/01_csv_upload.py", title="CSV 업로드", icon="📂", url_path="upload")
report_pg = st.Page("pages/02_report.py", title="리포트 조회", icon="📑", url_path="report")

if not st.session_state.logged_in:
    pg = st.navigation([st.Page(login_page, title="로그인", url_path="login")], position="hidden")
else:
    pg = st.navigation([profile_pg, upload_pg, report_pg], position="hidden")

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.title("💸 Catcher")

    if not st.session_state.logged_in:
        st.subheader("로그인")

        input_user_id = st.text_input("User ID 입력")
        input_user_name = st.text_input("이름 입력")

        if st.button("로그인", width="stretch"):
            if input_user_id.strip() == "" or input_user_name.strip() == "":
                st.warning("User ID와 이름을 모두 입력해주세요.")
            else:
                try:
                    user_id = int(input_user_id)
                except ValueError:
                    st.error("User ID는 숫자로 입력해주세요.")
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

        st.page_link(profile_pg, label="프로필", icon="👤")
        st.page_link(upload_pg, label="CSV 업로드", icon="📂")
        st.page_link(report_pg, label="리포트 조회", icon="📑")

        st.markdown("---")

        if st.button("로그아웃", width="stretch"):
            logout()

# 선택된 페이지 렌더링
pg.run()
