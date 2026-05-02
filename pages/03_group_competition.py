from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.group_competition_service import (
    CompetitionCreateInput,
    GroupCreateInput,
    TransactionShareInput,
    create_competition,
    create_group,
    get_group_feed,
    get_group_leaderboard,
    list_group_competitions,
    list_user_groups,
    share_transaction_to_group,
)
from catcher_llm.services.user_data_service import get_user_transactions

settings = get_settings()

st.set_page_config(page_title="그룹 경쟁", page_icon="🏁", layout="wide")

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


def _build_group_label(group: dict[str, int | str]) -> str:
    """그룹 선택 박스에 표시할 그룹 요약 라벨을 만든다."""
    return f"{group['name']} · 멤버 {group['member_count']}명 · {group['role']}"


def _build_competition_label(competition: dict[str, int | str]) -> str:
    """대회 선택 박스에 표시할 대회 요약 라벨을 만든다."""
    return f"{competition['title']} ({competition['start_date']} ~ {competition['end_date']})"


st.title("🏁 그룹 경쟁")
st.caption("그룹을 만들고, 소비를 공유하고, 포인트 리더보드를 확인하세요.")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해 주세요.")
    st.stop()

user_id = st.session_state.user_id
if user_id is None:
    st.error("로그인한 사용자 정보를 찾을 수 없습니다.")
    st.stop()

groups = list_user_groups(int(user_id), settings=settings)

st.markdown("## 내 그룹")
if groups:
    group_table = pd.DataFrame(groups)
    st.dataframe(group_table, width="stretch", hide_index=True)
else:
    st.info("아직 참여한 그룹이 없습니다. 아래에서 첫 그룹을 만들어 보세요.")

st.markdown("## 그룹 만들기")
with st.form("create_group_form", clear_on_submit=False):
    group_name = st.text_input("그룹 이름", placeholder="예: 절약 원정대")
    group_description = st.text_area(
        "그룹 소개",
        placeholder="예: 식비와 생활비를 함께 점검하는 그룹",
        height=100,
    )
    member_ids_text = st.text_input(
        "초대할 사용자 ID",
        placeholder="예: 2, 3, 4",
        help="쉼표로 구분해 입력하면 됩니다. 비워두면 본인만 있는 그룹이 만들어집니다.",
    )
    create_group_submitted = st.form_submit_button("그룹 만들기", width="stretch")

if create_group_submitted:
    try:
        invited_member_ids = [
            int(raw_id.strip()) for raw_id in member_ids_text.split(",") if raw_id.strip()
        ]
        result = create_group(
            GroupCreateInput(
                owner_user_id=int(user_id),
                name=group_name,
                description=group_description,
                member_user_ids=invited_member_ids,
            ),
            settings=settings,
        )
    except ValueError as error:
        st.error(str(error))
    else:
        st.success(
            f"그룹이 생성되었습니다. 그룹 ID: {result.group_id}, 멤버 수: {result.member_count}명"
        )
        st.rerun()

if not groups:
    st.stop()

selected_group = st.selectbox(
    "조회할 그룹 선택",
    options=groups,
    format_func=_build_group_label,
)
selected_group_id = int(selected_group["group_id"])

competitions = list_group_competitions(selected_group_id, settings=settings)

competition_col, feed_col = st.columns([1, 1.2])

with competition_col:
    st.markdown("## 대회 만들기")
    with st.form("create_competition_form", clear_on_submit=False):
        competition_title = st.text_input("대회 이름", placeholder="예: 5월 절약 대결")
        competition_start_date = st.date_input("시작일", value=date.today())
        competition_end_date = st.date_input("종료일", value=date.today())
        create_competition_submitted = st.form_submit_button("대회 만들기", width="stretch")

    if create_competition_submitted:
        try:
            create_competition(
                CompetitionCreateInput(
                    group_id=selected_group_id,
                    title=competition_title,
                    start_date=competition_start_date,
                    end_date=competition_end_date,
                ),
                settings=settings,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("대회가 생성되었습니다.")
            st.rerun()

    st.markdown("## 리더보드")
    if competitions:
        selected_competition = st.selectbox(
            "리더보드 대회 선택",
            options=competitions,
            format_func=_build_competition_label,
        )
        leaderboard = get_group_leaderboard(
            int(selected_competition["competition_id"]),
            settings=settings,
        )
        if leaderboard:
            st.dataframe(pd.DataFrame(leaderboard), width="stretch", hide_index=True)
        else:
            st.info("아직 적립된 포인트가 없습니다.")
    else:
        selected_competition = None
        st.info("먼저 대회를 만들어 주세요.")

with feed_col:
    st.markdown("## 소비 공유")
    user_transactions = get_user_transactions(int(user_id), settings=settings)
    if user_transactions:
        transaction_options = {
            (
                f"{transaction['id']} · {transaction['used_at']} · "
                f"{transaction['description'] or '설명 없음'} · {transaction['amount'] or 0}원"
            ): int(transaction["id"])
            for transaction in user_transactions
        }

        with st.form("share_transaction_form", clear_on_submit=False):
            selected_transaction_label = st.selectbox(
                "공유할 소비 선택",
                options=list(transaction_options.keys()),
            )
            share_comment = st.text_area(
                "공유 코멘트",
                placeholder="왜 이 소비를 공유하는지 짧게 적어 보세요.",
                height=90,
            )
            share_submitted = st.form_submit_button("소비 공유", width="stretch")

        if share_submitted:
            try:
                share_result = share_transaction_to_group(
                    TransactionShareInput(
                        group_id=selected_group_id,
                        shared_by_user_id=int(user_id),
                        transaction_id=transaction_options[selected_transaction_label],
                        competition_id=(
                            int(selected_competition["competition_id"])
                            if competitions and selected_competition is not None
                            else None
                        ),
                        comment=share_comment,
                    ),
                    settings=settings,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"공유 완료! {share_result.awarded_points}포인트가 적립되었습니다.")
                st.rerun()
    else:
        st.info("공유할 소비 내역이 없습니다. 먼저 CSV를 업로드해 주세요.")

    st.markdown("## 그룹 피드")
    feed_items = get_group_feed(selected_group_id, settings=settings)
    if feed_items:
        st.dataframe(pd.DataFrame(feed_items), width="stretch", hide_index=True)
    else:
        st.info("아직 공유된 소비가 없습니다.")
