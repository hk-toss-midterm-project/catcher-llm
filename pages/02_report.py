import streamlit as st

st.set_page_config(
    page_title="리포트 조회",
    page_icon="📑",
    layout="wide"
)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해주세요.")
    st.page_link("app.py", label="로그인 화면으로 이동")
    st.stop()


with st.sidebar:
    st.title("💸 Catcher")
    st.success(f"{st.session_state.user_profile['name']}님")
    st.caption(f"User ID: {st.session_state.user_id}")

    st.markdown("---")
    st.page_link("app.py", label="프로필", icon="👤")
    st.page_link("pages/01_csv_upload.py", label="CSV 업로드", icon="📂")
    st.page_link("pages/02_report.py", label="리포트 조회", icon="📑")


st.title("📑 리포트 조회")

report_type = st.radio(
    "조회할 리포트를 선택하세요.",
    ["일간 레포트", "주간 레포트", "월간 레포트"],
    horizontal=True
)

if "uploaded_df" not in st.session_state:
    st.info("먼저 CSV 업로드 페이지에서 데이터를 업로드해주세요.")
    st.stop()

df = st.session_state.uploaded_df
profile = st.session_state.user_profile

st.subheader(f"{profile['name']}님의 {report_type}")

col1, col2 = st.columns(2)
col1.metric("데이터 건수", f"{len(df)}건")
col2.metric("컬럼 수", f"{len(df.columns)}개")

st.markdown("---")

if report_type == "일간 레포트":
    st.subheader("☀️ 일간 소비 요약")
    st.info("오늘의 소비 패턴, 야간 소비, 반복 소비를 보여주는 영역입니다.")

elif report_type == "주간 레포트":
    st.subheader("📆 주간 소비 요약")
    st.info("이번 주 카테고리별 소비, 전주 대비 변화, 반복 소비를 보여주는 영역입니다.")

elif report_type == "월간 레포트":
    st.subheader("🗓️ 월간 소비 요약")
    st.info("이번 달 총소비, 상위 소비처, 절약 가능 금액을 보여주는 영역입니다.")

st.subheader("업로드 데이터 미리보기")
st.dataframe(df.head())
