from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.session_history_service import build_user_session_id
from catcher_llm.services.user_data_service import authenticate_user, ensure_user_database

settings = get_settings()
LANGCHAIN_SESSION_ID_KEY = "langchain_session_id"
CHAT_STATE_LOADED_KEY = "chat_state_loaded_from_sqlite"
CHAT_MESSAGES_KEY = "chat_messages"

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
    """앱 시작 시 사용자 SQLite 데이터베이스를 준비한다."""
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

if LANGCHAIN_SESSION_ID_KEY not in st.session_state:
    st.session_state[LANGCHAIN_SESSION_ID_KEY] = None

if CHAT_STATE_LOADED_KEY not in st.session_state:
    st.session_state[CHAT_STATE_LOADED_KEY] = False

if (
    st.session_state.logged_in
    and isinstance(st.session_state.user_id, int)
    and st.session_state[LANGCHAIN_SESSION_ID_KEY] is None
):
    st.session_state[LANGCHAIN_SESSION_ID_KEY] = build_user_session_id(st.session_state.user_id)


# -----------------------------
# 로그아웃 함수
# -----------------------------
def logout() -> None:
    """로그인 상태와 화면 세션에 남은 사용자별 채팅 상태를 초기화한다."""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_profile = None
    st.session_state[LANGCHAIN_SESSION_ID_KEY] = None
    st.session_state[CHAT_STATE_LOADED_KEY] = False
    st.session_state.pop(CHAT_MESSAGES_KEY, None)
    st.rerun()


# -----------------------------
# 페이지 정의
# -----------------------------
def login_page() -> None:
    """로그인 전 안내 화면을 렌더링한다."""
    st.title("Catcher 소비 분석 서비스")
    st.info("왼쪽 사이드바에서 User ID와 이름을 입력하고 로그인해주세요.")


def profile_page() -> None:
    """로그인한 사용자의 기본 프로필과 절약 목표를 보여준다."""
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
    # 절약 목표 추가 영역
    # -----------------------------
    st.markdown("---")
    st.subheader("🎯 나의 절약 목표")

    saving_goal_text = profile.get("saving_goal_text")

    if saving_goal_text:
        st.markdown(
            f"""
            <div style="
                background-color:#FFF7ED;
                color:#C2410C;
                padding:18px;
                border-radius:12px;
                font-size:17px;
                font-weight:600;
                line-height:1.6;
            ">
                {saving_goal_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("아직 등록된 절약 목표가 없습니다.")

    st.markdown("---")
    st.markdown(
        """
        <div style="
            background-color:#E8F2FF;
            color:#1D4ED8;
            padding:16px;
            border-radius:10px;
            font-weight:500;
            ">
            로그인 완료! 왼쪽 메뉴에서 CSV 업로드, 리포트 조회, AI 상담을 선택하세요.
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
chat_pg = st.Page("pages/03_chat.py", title="AI 상담", icon="💬", url_path="chat")

if not st.session_state.logged_in:
    pg = st.navigation([st.Page(login_page, title="로그인", url_path="login")], position="hidden")
else:
    pg = st.navigation([profile_pg, upload_pg, report_pg, chat_pg], position="hidden")

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.title("💸 Catcher")

    if not st.session_state.logged_in:
        st.subheader("로그인")

        input_user_id = st.text_input("User ID 입력")
        input_user_name = st.text_input("이름 입력")

        if st.button("로그인", use_container_width=True):
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
                    st.session_state[LANGCHAIN_SESSION_ID_KEY] = build_user_session_id(user_id)
                    st.session_state[CHAT_STATE_LOADED_KEY] = False
                    st.session_state.pop(CHAT_MESSAGES_KEY, None)

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
        st.page_link(chat_pg, label="AI 상담", icon="💬")

        st.markdown("---")

        if st.button("로그아웃", use_container_width=True):
            logout()

# 선택된 페이지 렌더링
pg.run()
