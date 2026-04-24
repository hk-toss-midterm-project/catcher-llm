import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CSV 업로드",
    page_icon="📂",
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


st.title("📂 CSV 업로드")

uploaded_file = st.file_uploader(
    "소비 데이터 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_df = df

    st.success("CSV 업로드 완료")

    col1, col2 = st.columns(2)
    col1.metric("데이터 건수", f"{len(df)}건")
    col2.metric("컬럼 수", f"{len(df.columns)}개")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())
else:
    st.info("CSV 파일을 업로드하면 리포트 조회에서 사용할 수 있습니다.")