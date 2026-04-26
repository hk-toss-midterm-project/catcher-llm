from datetime import timedelta

import pandas as pd
import streamlit as st
import plotly.express as px


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
DATE_COL = "사용 시간"
AMOUNT_COL = "사용 금액"
CATEGORY_COL = "업종 카테고리"
PLACE_COL = "결제 내역"

if DATE_COL not in df.columns:
    st.error(f"CSV에 '{DATE_COL}' 컬럼이 없습니다. 컬럼명을 확인하세요.")
    st.stop()

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
selected_date = pd.to_datetime(selected_date)

# -----------------------------
# 📅 일간 레포트
# -----------------------------
if report_type == "일간 레포트":
    yesterday = selected_date - timedelta(days=1)

    today_df = df[df[DATE_COL].dt.date == selected_date.date()]
    yesterday_df = df[df[DATE_COL].dt.date == yesterday.date()]

    st.subheader(f"☀️ 일간 레포트 ({selected_date.date()})")

    if today_df.empty:
        st.warning("해당 날짜의 데이터가 없습니다.")
    else:
        today_total = today_df[AMOUNT_COL].sum()
        today_count = len(today_df)
        today_avg = today_df[AMOUNT_COL].mean()

        yesterday_total = yesterday_df[AMOUNT_COL].sum() if not yesterday_df.empty else 0
        diff_amount = today_total - yesterday_total

        if yesterday_total > 0:
            diff_rate = (diff_amount / yesterday_total) * 100
        else:
            diff_rate = 0

        # -----------------------------
        # 핵심 지표
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "오늘 총 소비",
            f"{today_total:,.0f}원",
            f"{diff_amount:,.0f}원"
        )

        col2.metric(
            "거래 건수",
            f"{today_count}건"
        )

        col3.metric(
            "평균 결제 금액",
            f"{today_avg:,.0f}원"
        )

        st.markdown("### 🧾 전날 대비 소비 요약")

        if yesterday_df.empty:
            compare_text = "전날 데이터가 없어 비교는 어렵지만, 오늘 소비 내역을 기준으로 분석했어요."
        else:
            if diff_amount > 0:
                compare_text = f"오늘은 전날보다 {diff_amount:,.0f}원 더 사용했어요. 약 {diff_rate:.1f}% 증가했어요."
            elif diff_amount < 0:
                compare_text = f"오늘은 전날보다 {abs(diff_amount):,.0f}원 덜 사용했어요. 약 {abs(diff_rate):.1f}% 감소했어요."
            else:
                compare_text = "오늘 소비 금액은 전날과 동일해요."

        st.info(compare_text)

        # -----------------------------
        # 카테고리 분석
        # -----------------------------
        st.markdown("### 📌 오늘 가장 많이 쓴 카테고리")

        category_sum = (
            today_df.groupby(CATEGORY_COL)[AMOUNT_COL]
            .sum()
            .sort_values(ascending=False)
        )

        fig = px.pie(
            values=category_sum.values,
            names=category_sum.index,
            title="카테고리별 소비 비율"
        )


        top_category = category_sum.index[0]
        top_category_amount = category_sum.iloc[0]

        st.markdown(
            f"""
            오늘 가장 소비가 많았던 카테고리는  
            **{top_category}**이고, 총 **{top_category_amount:,.0f}원**을 사용했어요.
            """
        )

        st.plotly_chart(fig, use_container_width=True)
        fig.update_traces(textposition='inside', textinfo='percent+label')

        # -----------------------------
        # 소비 피드백
        # -----------------------------
        st.markdown("### 💬 오늘의 소비 피드백")

        food_keywords = ["배달의민족", "쿠팡이츠", "스타벅스", "이디야", "투썸플레이스", "맥도날드", "CU", "GS25"]
        food_like_df = today_df[today_df[PLACE_COL].astype(str).isin(food_keywords)]

        late_night_df = today_df[today_df[DATE_COL].dt.hour >= 21]

        feedbacks = []

        if diff_amount > 0 and yesterday_total > 0:
            feedbacks.append(f"전날보다 소비가 늘었어요. 특히 증가한 소비가 꼭 필요한 지출이었는지 확인해보면 좋아요.")

        if top_category == "식비":
            feedbacks.append("오늘은 식비 비중이 높아요. 카페, 편의점, 배달 소비가 반복되면 하루 총액이 쉽게 커질 수 있어요.")

        if len(food_like_df) >= 3:
            feedbacks.append("식비성 결제가 여러 번 발생했어요. 내일은 카페나 간식 소비를 한 번만 줄여도 절약 효과가 있어요.")

        if len(late_night_df) > 0:
            feedbacks.append("21시 이후 소비가 있었어요. 야간 소비는 충동 소비로 이어지기 쉬우니 한 번 점검해보면 좋아요.")

        if not feedbacks:
            feedbacks.append("오늘 소비는 비교적 안정적인 편이에요. 이 흐름을 유지해보세요.")

        for feedback in feedbacks:
            st.write(f"- {feedback}")

        for feedback in feedbacks:
            st.write(f"- {feedback}")

        # -----------------------------
        # 👍👎 피드백 평가
        # -----------------------------
        st.markdown("### 피드백이 도움이 되었나요?")

        if "daily_feedback_reaction" not in st.session_state:
            st.session_state.daily_feedback_reaction = None

        col_like, col_dislike = st.columns(2)

        with col_like:
            if st.button("👍 좋아요", use_container_width=True):
                st.session_state.daily_feedback_reaction = "좋아요"

        with col_dislike:
            if st.button("👎 싫어요", use_container_width=True):
                st.session_state.daily_feedback_reaction = "싫어요"

        if st.session_state.daily_feedback_reaction == "좋아요":
            st.success("좋아요를 선택했어요. 더 비슷한 방식으로 피드백할게요!")

        elif st.session_state.daily_feedback_reaction == "싫어요":
            st.warning("싫어요를 선택했어요. 다음 피드백은 더 구체적으로 개선해볼게요.")

        # -----------------------------
        # 오늘 소비 내역
        # -----------------------------
        st.markdown("### 📋 오늘 소비 내역")
        st.dataframe(today_df, use_container_width=True)

        # -----------------------------
        # 전날 소비 내역
        # -----------------------------
        if not yesterday_df.empty:
            with st.expander("전날 소비 내역 보기"):
                st.dataframe(yesterday_df, use_container_width=True)
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
