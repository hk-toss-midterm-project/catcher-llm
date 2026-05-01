import pandas as pd
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.transaction_upload_service import (
    resolve_transaction_upload_column_mapping,
    upload_transactions_dataframe,
)

settings = get_settings()

st.set_page_config(page_title="CSV 업로드", page_icon="📂", layout="wide")

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

st.title("📂 CSV 업로드")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해주세요.")
    st.stop()

uploaded_file = st.file_uploader("소비 데이터 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_df = df

    st.success("CSV 업로드 완료")

    col1, col2 = st.columns(2)
    col1.metric("데이터 건수", f"{len(df)}건")
    col2.metric("컬럼 수", f"{len(df.columns)}개")

    st.subheader("원본 CSV 컬럼")
    st.write(", ".join(str(column_name) for column_name in df.columns))

    column_mapping = resolve_transaction_upload_column_mapping(
        [str(column_name) for column_name in df.columns]
    )
    st.subheader("감지된 컬럼 매핑")
    if column_mapping:
        mapping_df = pd.DataFrame(
            [
                {"DB 필드": field_name, "CSV 컬럼": csv_column}
                for field_name, csv_column in column_mapping.items()
            ]
        )
        st.dataframe(mapping_df, width="stretch", hide_index=True)
    else:
        st.warning("transactions 테이블에 매핑할 수 있는 컬럼을 찾지 못했습니다.")

    st.caption(
        "description, category, payment_channel 값이 없으면 가맹점명을 기준으로 LangChain이 추론한 뒤 SQLite transactions 테이블에 저장합니다."
    )

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    if st.button("거래 내역 DB 저장", width="stretch"):
        user_id = st.session_state.user_id
        if user_id is None:
            st.error("로그인한 사용자 ID를 찾을 수 없습니다.")
            st.stop()

        try:
            with st.spinner("가맹점명을 분석하고 거래 내역을 SQLite에 저장하는 중..."):
                result = upload_transactions_dataframe(
                    df,
                    user_id=int(user_id),
                    settings=settings,
                )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"{result.inserted_count}건의 거래 내역을 저장했습니다.")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("추론한 가맹점 수", f"{result.inferred_count}개")
            metric_col2.metric("건너뛴 행", f"{result.skipped_count}건")
else:
    st.info("CSV 파일을 업로드하면 거래 내역을 SQLite transactions 테이블에 저장할 수 있습니다.")
