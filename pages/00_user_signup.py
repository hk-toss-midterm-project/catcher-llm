import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.user_data_service import UserRegistrationInput, register_user

settings = get_settings()

st.set_page_config(page_title="회원 가입", page_icon="📝", layout="wide")

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

st.title("📝 회원 가입")
st.caption("User ID는 가입 완료 시 자동 발급됩니다.")

gender_label_map = {
    "Female": "여성",
    "Male": "남성",
    "Other": "기타",
}

with st.form("user_signup_form", clear_on_submit=False):
    st.markdown("#### 기본 정보")
    name_column, age_column, gender_column = st.columns([1.4, 0.8, 0.8])

    with name_column:
        name = st.text_input("이름", placeholder="홍길동")

    with age_column:
        age = st.number_input("나이", min_value=0, max_value=130, value=30, step=1)

    with gender_column:
        gender = st.selectbox(
            "성별",
            options=list(gender_label_map.keys()),
            index=1,
            format_func=gender_label_map.__getitem__,
        )

    st.markdown("#### 소득과 목표")
    job_column, region_column = st.columns(2)

    with job_column:
        occupation = st.text_input("직업", placeholder="데이터 분석가")

    with region_column:
        region = st.text_input("거주 지역", placeholder="서울 서울-마포구")

    income_column, spending_goal_column = st.columns(2)

    with income_column:
        annual_income = st.number_input(
            "연소득",
            min_value=0,
            value=0,
            step=1_000_000,
            format="%d",
        )

    with spending_goal_column:
        target_max_spending_amount = st.number_input(
            "월 목표 최대 소비 금액",
            min_value=0,
            value=0,
            step=100_000,
            format="%d",
        )

    st.markdown("#### 상세 프로필")
    persona = st.text_area(
        "페르소나",
        placeholder="소비 성향, 생활 패턴, 직업/관심사 맥락을 입력하세요.",
        height=150,
    )
    saving_goal_text = st.text_area(
        "절약 목표",
        placeholder="예: 전세 보증금 마련을 위해 월별 지출 상한을 지키기",
        height=110,
    )

    submitted = st.form_submit_button("가입하기", width="stretch")

if submitted:
    try:
        result = register_user(
            UserRegistrationInput(
                name=name,
                age=int(age),
                occupation=occupation,
                gender=gender,
                annual_income=int(annual_income),
                region=region,
                persona=persona,
                saving_goal_text=saving_goal_text,
                target_max_spending_amount=int(target_max_spending_amount),
            ),
            settings=settings,
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
    else:
        st.success(f"회원가입이 완료되었습니다. User ID: {result.user_id}")
        st.info("왼쪽 로그인 메뉴에서 User ID와 이름으로 로그인해주세요.")
