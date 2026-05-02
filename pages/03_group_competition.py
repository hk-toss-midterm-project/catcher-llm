from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.group_competition_service import (
    GroupCreateInput,
    create_group,
    get_group_leaderboard_for_group,
    get_group_member_feedback_status,
    list_user_groups,
)

settings = get_settings()

st.set_page_config(page_title="그룹 경쟁", page_icon="🏁", layout="wide")


def _render_page_styles() -> None:
    """그룹 경쟁 페이지를 토스 스타일 카드 레이아웃으로 꾸민다."""
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

.leaderboard-item,
.feedback-item {
    display: grid;
    align-items: center;
    gap: 14px;
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFF 100%);
    border: 1px solid #E8EEF6;
    border-radius: 22px;
    padding: 16px 18px;
}

.leaderboard-item {
    grid-template-columns: 56px 1fr auto;
}

.feedback-item {
    grid-template-columns: 1.2fr 0.8fr 1fr;
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

.leader-name,
.feedback-name {
    font-size: 18px;
    font-weight: 800;
    color: #191F28;
    margin: 0 0 4px 0;
}

.leader-meta,
.feedback-meta {
    font-size: 13px;
    color: #8B95A1;
    margin: 0;
}

.leader-score,
.feedback-score {
    text-align: right;
}

.leader-score-value,
.feedback-score-value {
    font-size: 24px;
    font-weight: 800;
    color: #191F28;
    margin: 0 0 4px 0;
}

.leader-score-label,
.feedback-score-label {
    font-size: 12px;
    color: #8B95A1;
    margin: 0;
}

.feedback-mission {
    padding: 12px 14px;
    background: #F6F9FC;
    border-radius: 18px;
    color: #4E5968;
    font-size: 14px;
    line-height: 1.55;
}

div[data-testid="stForm"] {
    background: transparent;
    border: none;
    padding: 0;
}

div[data-testid="stSelectbox"] > label,
div[data-testid="stTextInput"] > label,
div[data-testid="stTextArea"] > label {
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

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 32px;
    }

    .feedback-item {
        grid-template-columns: 1fr;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _build_group_label(group: dict[str, int | str]) -> str:
    """그룹 선택 박스에 표시할 그룹 요약 라벨을 만든다."""
    return f"{group['name']} · 멤버 {group['member_count']}명 · {group['role']}"


def _render_metric_card(*, label: str, value: str, caption: str) -> None:
    """상단 요약 지표 카드를 렌더링한다."""
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


def _render_leaderboard_cards(leaderboard: list[dict[str, int | str]]) -> None:
    """그룹 멤버 순위를 카드 목록으로 렌더링한다."""
    if not leaderboard:
        st.info("아직 그룹 멤버가 없습니다.")
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


def _render_feedback_status_cards(
    feedback_status: list[dict[str, int | str | None]],
) -> None:
    """멤버별 피드백 실천 현황을 카드 목록으로 렌더링한다."""
    if not feedback_status:
        st.info("아직 피드백 반응 기록이 없습니다.")
        return

    for item in feedback_status:
        latest_mission = str(item["latest_mission"] or "아직 기록된 미션이 없습니다.")
        st.markdown(
            f"""
<div class="feedback-item">
    <div>
        <div class="feedback-name">{item["user_name"]}</div>
        <p class="feedback-meta">
            확인한 피드백 {int(item["feedback_checked_count"])}회 ·
            긍정 반응 {int(item["positive_reaction_count"])}회
        </p>
    </div>
    <div class="feedback-score">
        <div class="feedback-score-value">{int(item["feedback_acceptance_rate"])}%</div>
        <p class="feedback-score-label">피드백 수용률</p>
    </div>
    <div class="feedback-mission">{latest_mission}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


_render_page_styles()

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해 주세요.")
    st.stop()

user_id = st.session_state.user_id
if user_id is None:
    st.error("로그인한 사용자 정보를 찾을 수 없습니다.")
    st.stop()

groups = list_user_groups(int(user_id), settings=settings)

st.markdown(
    """
<div class="hero-card">
    <div class="hero-kicker">GROUP RACE</div>
    <h1 class="hero-title">그룹 경쟁</h1>
    <p class="hero-desc">
        그룹을 만들면 바로 personal_score 기준 순위를 볼 수 있어요.
        이제 소비 공유보다, 서로가 피드백을 얼마나 잘 지키고 있는지 함께 확인하는 대시보드에 집중합니다.
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
        label="랭킹 기준",
        value="personal_score",
        caption="그룹 생성 직후 바로 순위 반영",
    )
with metric_col3:
    _render_metric_card(
        label="실천 비교",
        value="피드백 수용률",
        caption="상대방이 내 실천 정도를 볼 수 있음",
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## 내 그룹")
st.markdown(
    '<p class="section-caption">참여 중인 그룹을 빠르게 확인하고, 선택한 그룹 기준으로 바로 리더보드와 실천 현황을 봅니다.</p>',
    unsafe_allow_html=True,
)

if groups:
    chip_markup = "".join(
        [
            (
                f'<div class="group-chip">{group["name"]} · '
                f'{group["member_count"]}명 · {group["role"]}</div>'
            )
            for group in groups
        ]
    )
    st.markdown(
        f"""
<div class="toss-card">
    <div class="card-title">내 그룹 요약</div>
    <p class="card-desc">그룹 생성만 끝나면 별도 대회 생성 없이도 점수 순위가 바로 집계됩니다.</p>
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
    <p class="card-desc">새 그룹을 만들면 생성 직후 바로 멤버 랭킹이 활성화됩니다.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

create_col, dashboard_col = st.columns([0.9, 1.1], gap="large")

with create_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">그룹 만들기</div>
    <p class="card-desc">같이 경쟁할 멤버를 초대하세요. 그룹을 만든 순간 personal_score 기준 리더보드가 바로 열립니다.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("create_group_form", clear_on_submit=False):
        group_name = st.text_input("그룹 이름", placeholder="예: 절약 원정대")
        group_description = st.text_area(
            "그룹 소개",
            placeholder="예: 피드백 실천과 생활비 절약을 함께 체크하는 그룹",
            height=110,
        )
        member_ids_text = st.text_input(
            "초대할 사용자 ID",
            placeholder="예: 2, 3, 4",
            help="쉼표로 구분해 입력하세요. 비워두면 본인만 있는 그룹으로 시작합니다.",
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
                f"그룹 생성 완료. 그룹 ID {result.group_id}, 멤버 {result.member_count}명으로 바로 랭킹을 볼 수 있습니다."
            )
            st.rerun()

if not groups:
    st.stop()

with dashboard_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">활동할 그룹 선택</div>
    <p class="card-desc">선택한 그룹 기준으로 리더보드와 피드백 실천 현황이 바뀝니다.</p>
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
leaderboard = get_group_leaderboard_for_group(selected_group_id, settings=settings)
feedback_status = get_group_member_feedback_status(selected_group_id, settings=settings)

leaderboard_col, feedback_col = st.columns([0.92, 1.08], gap="large")

with leaderboard_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">리더보드</div>
    <p class="card-desc">대회 생성 없이도 현재 그룹 멤버의 personal_score 순위를 바로 확인할 수 있어요.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    _render_leaderboard_cards(leaderboard)

with feedback_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">피드백 실천 현황</div>
    <p class="card-desc">상대방이 내가 피드백을 얼마나 잘 받아들이고 지키는지 볼 수 있도록 최근 반응과 미션을 공개합니다.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    _render_feedback_status_cards(feedback_status)
