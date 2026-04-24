import pandas as pd
import streamlit as st

st.title("📂 CSV 업로드")

uploaded_file = st.file_uploader("소비 데이터 CSV 파일을 업로드하세요.", type=["csv"])

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
