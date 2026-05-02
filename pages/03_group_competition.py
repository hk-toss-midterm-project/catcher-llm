from __future__ import annotations

from datetime import date

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


def _render_page_styles() -> None:
    """그룹 경쟁 페이지를 토스 스타일 카드 레이아웃으로 보이게 하는 CSS를 주입한다."""
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(49,130,246,0.10), transparent 24%),
        linear-gradient(180deg, #F4F7FB 0%, #EEF3F8 100%);
}

.block-container {
    padding-top: 2.3rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

.hero-card,
.toss-card {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(229,236,245,0.95);
    border-radius: 28px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(8px);
}

.hero-card {
    padding: 30px 32px 28px 32px;
    margin-bottom: 18px;
}

.hero-kicker {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: #EAF2FF;
    color: #3182F6;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 40px;
    line-height: 1.14;
    font-weight: 800;
    color: #191F28;
    letter-spacing: -0.04em;
    margin: 0 0 10px 0;
}

.hero-desc {
    font-size: 16px;
    color: #667085;
    line-height: 1.7;
    margin: 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin: 18px 0 26px 0;
}

.metric-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
    border: 1px solid #E7EEF8;
    border-radius: 22px;
    padding: 22px 22px 18px 22px;
}

.metric-label {
    font-size: 14px;
    color: #8B95A1;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 28px;
    color: #191F28;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}

.metric-caption {
    font-size: 13px;
    color: #6B7684;
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #191F28;
    letter-spacing: -0.03em;
    margin: 30px 0 16px 0;
}

.section-caption {
    font-size: 15px;
    color: #8B95A1;
    margin: -4px 0 18px 0;
}

.toss-card {
    padding: 24px;
    margin-bottom: 16px;
}

.card-title {
    font-size: 22px;
    font-weight: 800;
    color: #191F28;
    margin: 0 0 8px 0;
}

.card-desc {
    font-size: 14px;
    color: #7C8798;
    line-height: 1.65;
    margin: 0 0 16px 0;
}

.group-chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.group-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 999px;
    background: #F2F6FB;
    color: #4E5968;
    font-size: 13px;
    font-weight: 700;
}

.leaderboard-shell {
    display: grid;
    gap: 12px;
}

.leaderboard-item {
    display: grid;
    grid-template-columns: 56px 1fr auto;
    align-items: center;
    gap: 14px;
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFF 100%);
    border: 1px solid #E8EEF6;
    border-radius: 22px;
    padding: 16px 18px;
}

.leader-rank {
    width: 56px;
    height: 56px;
    border-radius: 18px;
    background: #EAF2FF;
    color: #3182F6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 800;
}

.leader-name {
    font-size: 18px;
    font-weight: 800;
    color: #191F28;
    margin: 0 0 4px 0;
}

.leader-meta {
    font-size: 13px;
    color: #8B95A1;
    margin: 0;
}

.leader-score {
    text-align: right;
}

.leader-score-value {
    font-size: 24px;
    font-weight: 800;
    color: #191F28;
    margin: 0 0 4px 0;
}

.leader-score-label {
    font-size: 12px;
    color: #8B95A1;
    margin: 0;
}

.feed-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
    border: 1px solid #E8EEF6;
    border-radius: 24px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.feed-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.feed-user {
    font-size: 16px;
    font-weight: 800;
    color: #191F28;
}

.feed-time {
    font-size: 12px;
    color: #8B95A1;
}

.feed-amount {
    font-size: 24px;
    font-weight: 800;
    color: #3182F6;
    margin: 4px 0 8px 0;
}

.feed-desc {
    font-size: 14px;
    color: #4E5968;
    line-height: 1.65;
    margin: 0;
}

.feed-comment {
    margin-top: 12px;
    padding: 12px 14px;
    background: #F6F9FC;
    border-radius: 18px;
    color: #4E5968;
    font-size: 14px;
    line-height: 1.6;
}

div[data-testid="stForm"] {
    background: transparent;
    border: none;
    padding: 0;
}

div[data-testid="stSelectbox"] > label,
div[data-testid="stTextInput"] > label,
div[data-testid="stTextArea"] > label,
div[data-testid="stDateInput"] > label {
    font-weight: 700;
    color: #4E5968;
}

div.stButton > button,
div[data-testid="stForm"] button {
    border-radius: 18px;
    border: none;
    background: linear-gradient(180deg, #4EA1FF 0%, #3182F6 100%);
    color: white;
    font-weight: 800;
    height: 50px;
    box-shadow: 0 10px 24px rgba(49, 130, 246, 0.25);
}

div.stButton > button:hover,
div[data-testid="stForm"] button:hover {
    background: linear-gradient(180deg, #479BFA 0%, #2E78E4 100%);
}

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 32px;
    }
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


def _render_metric_card(*, label: str, value: str, caption: str) -> None:
    """상단 요약 지표 카드 하나를 렌더링한다."""
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-caption">{caption}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_feed_cards(feed_items: list[dict[str, int | str | None]]) -> None:
    """공유된 소비 피드를 카드 목록 형태로 렌더링한다."""
    if not feed_items:
        st.info("아직 공유된 소비가 없습니다.")
        return

    for item in feed_items:
        description = str(item["description"] or "설명 없는 소비")
        amount = int(item["amount"] or 0)
        comment = str(item["comment"] or "").strip()
        comment_block = f'<div class="feed-comment">{comment}</div>' if comment else ""
        st.markdown(
            f"""
<div class="feed-card">
    <div class="feed-top">
        <div class="feed-user">{item["shared_by_name"]}</div>
        <div class="feed-time">{item["shared_at"]}</div>
    </div>
    <div class="feed-amount">{amount:,}원</div>
    <p class="feed-desc">{description}</p>
    {comment_block}
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_leaderboard_cards(leaderboard: list[dict[str, int | str]]) -> None:
    """리더보드 결과를 토스 스타일 순위 카드로 렌더링한다."""
    if not leaderboard:
        st.info("아직 적립된 포인트가 없습니다.")
        return

    st.markdown('<div class="leaderboard-shell">', unsafe_allow_html=True)
    for item in leaderboard:
        st.markdown(
            f"""
<div class="leaderboard-item">
    <div class="leader-rank">{item["rank"]}</div>
    <div>
        <div class="leader-name">{item["user_name"]}</div>
        <p class="leader-meta">User ID {item["user_id"]}</p>
    </div>
    <div class="leader-score">
        <div class="leader-score-value">{int(item["points"]):,}</div>
        <p class="leader-score-label">personal_score</p>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _build_transaction_options(
    user_transactions: list[dict[str, int | str | None]],
) -> dict[str, int]:
    """공유 가능한 사용자 거래를 선택 박스용 라벨과 ID 매핑으로 변환한다."""
    return {
        (
            f"{transaction['id']} · {transaction['used_at']} · "
            f"{transaction['description'] or '설명 없음'} · {int(transaction['amount'] or 0):,}원"
        ): int(transaction["id"])
        for transaction in user_transactions
    }


_render_page_styles()

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해 주세요.")
    st.stop()

user_id = st.session_state.user_id
if user_id is None:
    st.error("로그인한 사용자 정보를 찾을 수 없습니다.")
    st.stop()

groups = list_user_groups(int(user_id), settings=settings)
user_transactions = get_user_transactions(int(user_id), settings=settings)

st.markdown(
    """
<div class="hero-card">
    <div class="hero-kicker">GROUP RACE</div>
    <h1 class="hero-title">그룹 경쟁</h1>
    <p class="hero-desc">
        내 소비를 그룹에 공유하고, personal_score 기준으로 누가 가장 꾸준히 점수를 쌓는지 확인해 보세요.
        토스처럼 가볍고 빠르게, 필요한 행동만 바로 할 수 있게 정리했어요.
    </p>
</div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    _render_metric_card(
        label="내 그룹",
        value=f"{len(groups)}개",
        caption="참여 중인 그룹 수",
    )
with metric_col2:
    _render_metric_card(
        label="내 거래",
        value=f"{len(user_transactions)}건",
        caption="공유 가능한 업로드 거래",
    )
with metric_col3:
    _render_metric_card(
        label="기본 공유 점수",
        value="10점",
        caption="소비 1건 공유 시 적립",
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## 내 그룹")
st.markdown(
    '<p class="section-caption">현재 참여 중인 그룹과 내 역할을 한눈에 확인하세요.</p>',
    unsafe_allow_html=True,
)

if groups:
    chip_markup = "".join(
        [
            (
                f'<div class="group-chip">{group["name"]} · '
                f"{group['member_count']}명 · {group['role']}</div>"
            )
            for group in groups
        ]
    )
    st.markdown(
        f"""
<div class="toss-card">
    <div class="card-title">내 그룹 요약</div>
    <p class="card-desc">참여 중인 그룹을 빠르게 훑고, 아래에서 바로 경쟁과 공유를 이어서 진행할 수 있어요.</p>
    <div class="group-chip-wrap">{chip_markup}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">아직 참여한 그룹이 없어요</div>
    <p class="card-desc">첫 그룹을 만들고 친구들과 소비를 공유해 보세요. 생성자는 자동으로 owner 역할을 받아요.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

create_col, manage_col = st.columns([0.9, 1.1], gap="large")

with create_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">그룹 만들기</div>
    <p class="card-desc">같이 경쟁할 멤버를 초대해서 새 그룹을 만들어요. 사용자 ID는 쉼표로 구분해 입력하면 돼요.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("create_group_form", clear_on_submit=False):
        group_name = st.text_input("그룹 이름", placeholder="예: 절약 원정대")
        group_description = st.text_area(
            "그룹 소개",
            placeholder="예: 식비와 생활비를 함께 점검하는 그룹",
            height=110,
        )
        member_ids_text = st.text_input(
            "초대할 사용자 ID",
            placeholder="예: 2, 3, 4",
            help="비워두면 본인만 있는 그룹이 만들어집니다.",
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
                f"그룹 생성 완료. 그룹 ID {result.group_id}, 멤버 {result.member_count}명으로 시작합니다."
            )
            st.rerun()

if not groups:
    st.stop()

with manage_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">활동할 그룹 선택</div>
    <p class="card-desc">선택한 그룹 기준으로 대회, 리더보드, 소비 공유 피드가 모두 바뀝니다.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    selected_group = st.selectbox(
        "조회할 그룹 선택",
        options=groups,
        format_func=_build_group_label,
    )

selected_group_id = int(selected_group["group_id"])
competitions = list_group_competitions(selected_group_id, settings=settings)
selected_competition = None

left_col, right_col = st.columns([0.95, 1.05], gap="large")

with left_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">대회 만들기</div>
    <p class="card-desc">이 그룹 안에서 기간을 정해 경쟁을 시작하세요. 현재 점수는 personal_score를 기준으로 보여줍니다.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown(
        '<p class="section-caption">그룹 멤버의 현재 personal_score 순위를 보여줍니다.</p>',
        unsafe_allow_html=True,
    )
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
        _render_leaderboard_cards(leaderboard)
    else:
        st.info("먼저 대회를 만들어 주세요.")

with right_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">소비 공유</div>
    <p class="card-desc">내 거래를 그룹에 올리고 10점을 쌓아 보세요. 같은 거래는 같은 그룹에 한 번만 공유할 수 있어요.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if user_transactions:
        transaction_options = _build_transaction_options(user_transactions)
        with st.form("share_transaction_form", clear_on_submit=False):
            selected_transaction_label = st.selectbox(
                "공유할 소비 선택",
                options=list(transaction_options.keys()),
            )
            share_comment = st.text_area(
                "공유 코멘트",
                placeholder="왜 이 소비를 공유하는지, 어떤 점을 같이 보고 싶은지 적어 보세요.",
                height=100,
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
                            if selected_competition is not None
                            else None
                        ),
                        comment=share_comment,
                    ),
                    settings=settings,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"공유 완료. {share_result.awarded_points}점이 적립되었습니다.")
                st.rerun()
    else:
        st.info("공유할 소비 내역이 없습니다. 먼저 CSV를 업로드해 주세요.")

    st.markdown("## 그룹 피드")
    st.markdown(
        '<p class="section-caption">그룹에 공유된 소비를 카드형 피드로 확인할 수 있어요.</p>',
        unsafe_allow_html=True,
    )
    feed_items = get_group_feed(selected_group_id, settings=settings)
    _render_feed_cards(feed_items)
