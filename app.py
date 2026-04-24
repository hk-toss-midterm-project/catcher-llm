import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Catcher",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기본 Streamlit 페이지 메뉴 숨기기
st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 경로
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MEMBER_PATH = BASE_DIR / "data" / "raw" / "csv" / "members_v1.csv"

@st.cache_data
def load_members():
    df = pd.read_csv(MEMBER_PATH)
    df["id"] = df["id"].astype(int)
    df["name"] = df["name"].astype(str)
    return df

members_df = load_members()

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

                matched_user = members_df[
                    (members_df["id"] == user_id) &
                    (members_df["name"] == input_user_name.strip())
                ]

                if matched_user.empty:
                    st.error("일치하는 사용자가 없습니다.")
                else:
                    user = matched_user.iloc[0]

                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_profile = {
                        "name": user["name"],
                        "age": user["age"],
                        "job": user["직업"],
                        "gender": user["성별"],
                        "income": user["연봉"],
                        "region": user["지역"],
                        "card_grade": user["최상위 카드등급"],
                        "persona": user["페르소나"]
                    }

                    st.rerun()

    else:
        profile = st.session_state.user_profile

        st.success(f"{profile['name']}님")
        st.caption(f"User ID: {st.session_state.user_id}")

        st.markdown("---")

        st.page_link("app.py", label="프로필", icon="👤")
        st.page_link("pages/01_csv_upload.py", label="CSV 업로드", icon="📂")
        st.page_link("pages/02_report.py", label="리포트 조회", icon="📑")

        st.markdown("---")

        if st.button("로그아웃", use_container_width=True):
            logout()

# -----------------------------
# 메인 화면: 로그인 체크
# -----------------------------
if (
    "logged_in" not in st.session_state
    or st.session_state.logged_in is False
    or "user_profile" not in st.session_state
    or st.session_state.user_profile is None
):
    st.title("Catcher 소비 분석 서비스")
    st.info("왼쪽 사이드바에서 User ID와 이름을 입력하고 로그인해주세요.")
    st.stop()

profile = st.session_state.user_profile

# -----------------------------
# 프로필 화면
# -----------------------------
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

st.markdown("---")
st.success("로그인 완료! 왼쪽 메뉴에서 CSV 업로드 또는 리포트 조회를 선택하세요.")