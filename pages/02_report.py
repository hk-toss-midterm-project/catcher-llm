from datetime import timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="리포트 조회", page_icon="📑", layout="wide")

# -----------------------------
# 기본 페이지 메뉴 숨기기
# -----------------------------
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

# -----------------------------
# 로그인 확인
# -----------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해주세요.")
    st.stop()

profile = st.session_state.user_profile

# -----------------------------
# 메인 UI
# -----------------------------
st.title("📑 리포트 조회")

# 날짜 선택 (달력)
selected_date = st.date_input("📅 날짜 선택")

# 리포트 타입 선택
st.markdown("리포트 유형 선택")

if "report_type" not in st.session_state:
    st.session_state.report_type = "일간 레포트"

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("일간 레포트", use_container_width=True):
        st.session_state.report_type = "일간 레포트"

with col2:
    if st.button("주간 레포트", use_container_width=True):
        st.session_state.report_type = "주간 레포트"

with col3:
    if st.button("월간 레포트", use_container_width=True):
        st.session_state.report_type = "월간 레포트"

report_type = st.session_state.report_type

st.markdown(
    f"""
    <div style="
        background-color:#E8F2FF;
        color:#1D4ED8;
        padding:12px 16px;
        border-radius:12px;
        font-weight:700;
        margin-top:10px;
    ">
        선택된 리포트: {report_type}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# -----------------------------
# 데이터 확인
# -----------------------------
if "uploaded_df" not in st.session_state:
    st.info("먼저 CSV 업로드 페이지에서 데이터를 업로드해주세요.")
    st.stop()

df = st.session_state.uploaded_df.copy()

# 👉 너 데이터에 맞게 반드시 수정해야 함
DATE_COL = "date"  # 예: "거래일자"
AMOUNT_COL = "amount"  # 예: "결제금액"

if DATE_COL not in df.columns:
    st.error(f"CSV에 '{DATE_COL}' 컬럼이 없습니다. 컬럼명을 확인하세요.")
    st.stop()

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
selected_date = pd.to_datetime(selected_date)

# -----------------------------
# 📅 일간 레포트
# -----------------------------
if report_type == "일간 레포트":
    report_df = df[df[DATE_COL].dt.date == selected_date.date()]

    st.subheader(f"☀️ 일간 레포트 ({selected_date.date()})")

    if report_df.empty:
        st.warning("해당 날짜의 데이터가 없습니다.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("총 소비", f"{report_df[AMOUNT_COL].sum():,.0f}원")
        col2.metric("거래 건수", f"{len(report_df)}건")
        col3.metric("평균 결제", f"{report_df[AMOUNT_COL].mean():,.0f}원")

        st.markdown("### 소비 내역")
        st.dataframe(report_df)

# -----------------------------
# 📆 주간 레포트
# -----------------------------
elif report_type == "주간 레포트":
    start_date = selected_date - timedelta(days=selected_date.weekday())
    end_date = start_date + timedelta(days=6)

    report_df = df[(df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)]

    st.subheader(f"📆 주간 레포트 ({start_date.date()} ~ {end_date.date()})")

    if report_df.empty:
        st.warning("해당 주간 데이터가 없습니다.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("주간 총 소비", f"{report_df[AMOUNT_COL].sum():,.0f}원")
        col2.metric("거래 건수", f"{len(report_df)}건")
        col3.metric("일 평균", f"{report_df[AMOUNT_COL].sum() / 7:,.0f}원")

        st.markdown("### 소비 내역")
        st.dataframe(report_df)

# -----------------------------
# 🗓️ 월간 레포트
# -----------------------------
elif report_type == "월간 레포트":
    year = selected_date.year
    month = selected_date.month

    report_df = df[(df[DATE_COL].dt.year == year) & (df[DATE_COL].dt.month == month)]

    st.subheader(f"🗓️ 월간 레포트 ({year}년 {month}월)")

    if report_df.empty:
        st.warning("해당 월 데이터가 없습니다.")
    else:
        days_in_month = selected_date.days_in_month

        col1, col2, col3 = st.columns(3)

        col1.metric("월간 총 소비", f"{report_df[AMOUNT_COL].sum():,.0f}원")
        col2.metric("거래 건수", f"{len(report_df)}건")
        col3.metric("일 평균", f"{report_df[AMOUNT_COL].sum() / days_in_month:,.0f}원")

        st.markdown("### 소비 내역")
        st.dataframe(report_df)
