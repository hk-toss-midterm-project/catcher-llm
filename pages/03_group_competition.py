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
    st.markdown(
        """
<style>
:root {
    --bg: #f5f7fb;
    --card: #ffffff;
    --text: #191f28;
    --sub: #6b7684;
    --muted: #8b95a1;
    --line: #e5e8ef;
    --blue: #3182f6;
    --blue-dark: #1b64da;
    --blue-soft: #eaf2ff;
    --green: #10b981;
    --green-soft: #ecfdf5;
    --shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
    --radius-xl: 30px;
    --radius-lg: 24px;
    --radius-md: 18px;
}

[data-testid="stSidebarNav"] {
    display: none;
}

.stApp {
    background:
        radial-gradient(circle at 88% 8%, rgba(49,130,246,0.12), transparent 26%),
        linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero-card {
    padding: 34px 36px;
    border-radius: var(--radius-xl);
    background:
        radial-gradient(circle at 88% 18%, rgba(255,255,255,0.28), transparent 28%),
        linear-gradient(135deg, #0f172a 0%, #2563eb 54%, #38bdf8 100%);
    color: white;
    box-shadow: 0 28px 70px rgba(37,99,235,0.22);
    margin-bottom: 18px;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.22);
    font-size: 13px;
    font-weight: 850;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.15;
    font-weight: 950;
    letter-spacing: -1.1px;
    margin: 0 0 12px 0;
}

.hero-desc {
    max-width: 720px;
    font-size: 16px;
    color: rgba(255,255,255,0.88);
    line-height: 1.75;
    font-weight: 650;
    margin: 0;
}

.hero-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.hero-chip {
    padding: 10px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.2);
    font-size: 13px;
    font-weight: 850;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin: 18px 0 28px 0;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 22px 24px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.055);
    min-height: 126px;
}

.metric-label {
    font-size: 13px;
    color: var(--muted);
    font-weight: 850;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 29px;
    color: var(--text);
    font-weight: 950;
    letter-spacing: -0.7px;
    margin-bottom: 6px;
}

.metric-caption {
    font-size: 13px;
    color: var(--sub);
    font-weight: 650;
    line-height: 1.45;
}

.section-title {
    font-size: 23px;
    font-weight: 950;
    color: var(--text);
    margin: 28px 0 6px 0;
    letter-spacing: -0.5px;
}

.section-caption {
    font-size: 14px;
    color: var(--muted);
    margin: 0 0 14px 0;
    line-height: 1.6;
    font-weight: 650;
}

.toss-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    padding: 24px;
    margin-bottom: 16px;
}

.card-title {
    font-size: 21px;
    font-weight: 950;
    color: var(--text);
    margin: 0 0 8px 0;
    letter-spacing: -0.4px;
}

.card-desc {
    font-size: 14px;
    color: var(--sub);
    line-height: 1.65;
    margin: 0;
    font-weight: 650;
}

.group-chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}

.group-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 11px 14px;
    border-radius: 999px;
    background: #f2f6fb;
    border: 1px solid #e7eef8;
    color: #4e5968;
    font-size: 13px;
    font-weight: 850;
}

.leaderboard-panel {
    background: var(--card);
    border: 1px solid #dbeafe;
    border-radius: 28px;
    padding: 24px;
    box-shadow: 0 24px 62px rgba(37,99,235,0.12);
}

.leaderboard-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 16px;
}

.leaderboard-title {
    font-size: 25px;
    font-weight: 950;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.6px;
}

.leaderboard-badge {
    padding: 9px 12px;
    border-radius: 999px;
    background: var(--blue-soft);
    color: var(--blue-dark);
    font-size: 12px;
    font-weight: 900;
    white-space: nowrap;
}

.leaderboard-shell {
    display: grid;
    gap: 12px;
}

.leaderboard-item {
    display: grid;
    grid-template-columns: 58px 1fr auto;
    align-items: center;
    gap: 14px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #e7eef8;
    border-radius: 22px;
    padding: 16px 18px;
}

.leaderboard-item:first-child {
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
    border-color: #bfdbfe;
}

.leader-rank {
    width: 58px;
    height: 58px;
    border-radius: 20px;
    background: var(--blue);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 950;
    box-shadow: 0 12px 24px rgba(49,130,246,0.22);
}

.leader-name,
.feedback-name {
    font-size: 18px;
    font-weight: 950;
    color: var(--text);
    margin: 0 0 5px 0;
    letter-spacing: -0.3px;
}

.leader-meta,
.feedback-meta {
    font-size: 13px;
    color: var(--muted);
    margin: 0;
    font-weight: 650;
}

.leader-score,
.feedback-score {
    text-align: right;
}

.leader-score-value {
    font-size: 26px;
    font-weight: 950;
    color: var(--blue-dark);
    margin: 0 0 4px 0;
    letter-spacing: -0.6px;
}

.leader-score-label,
.feedback-score-label {
    font-size: 12px;
    color: var(--muted);
    margin: 0;
    font-weight: 750;
}

.feedback-panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 24px;
    box-shadow: var(--shadow);
}

.feedback-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 16px;
}

.feedback-title {
    font-size: 25px;
    font-weight: 950;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.6px;
}

.feedback-item {
    display: grid;
    grid-template-columns: 1.1fr 0.7fr 1fr;
    align-items: center;
    gap: 14px;
    background: #ffffff;
    border: 1px solid #e7eef8;
    border-radius: 22px;
    padding: 17px 18px;
    margin-bottom: 12px;
}

.feedback-score-value {
    font-size: 28px;
    font-weight: 950;
    color: var(--text);
    letter-spacing: -0.6px;
    margin-bottom: 4px;
}

.feedback-rate-bar {
    height: 8px;
    border-radius: 999px;
    background: #edf2f7;
    overflow: hidden;
    margin-top: 8px;
}

.feedback-rate-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--blue), #60a5fa);
}

.feedback-mission {
    padding: 13px 15px;
    background: #f6f9fc;
    border-radius: 18px;
    color: #4e5968;
    font-size: 14px;
    line-height: 1.55;
    font-weight: 650;
}

div[data-testid="stForm"] {
    background: transparent;
    border: none;
    padding: 0;
}

div[data-testid="stSelectbox"] > label,
div[data-testid="stTextInput"] > label,
div[data-testid="stTextArea"] > label {
    font-weight: 850;
    color: #4e5968;
    font-size: 13px;
}

div[data-baseweb="input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] {
    border-radius: 16px;
}

div.stButton > button,
div[data-testid="stForm"] button {
    border-radius: 17px;
    border: none;
    background: linear-gradient(180deg, #4ea1ff 0%, #3182f6 100%);
    color: white;
    font-weight: 900;
    height: 50px;
    box-shadow: 0 10px 24px rgba(49,130,246,0.22);
}

div.stButton > button:hover,
div[data-testid="stForm"] button:hover {
    border: none;
    color: white;
    filter: brightness(0.98);
}

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 32px;
    }

    .leaderboard-item,
    .feedback-item {
        grid-template-columns: 1fr;
        text-align: left;
    }

    .leader-score,
    .feedback-score {
        text-align: left;
    }

    .leaderboard-header,
    .feedback-header {
        flex-direction: column;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _build_group_label(group: dict[str, int | str]) -> str:
    return f"{group['name']} · 멤버 {group['member_count']}명 · {group['role']}"


def _render_metric_card(*, label: str, value: str, caption: str) -> None:
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
    if not leaderboard:
        st.info("아직 그룹 멤버가 없습니다.")
        return

    st.markdown('<div class="leaderboard-shell">', unsafe_allow_html=True)

    for item in leaderboard:
        rank = int(item["rank"])
        rank_label = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else str(rank)

        st.markdown(
            f"""
<div class="leaderboard-item">
    <div class="leader-rank">{rank_label}</div>
    <div>
        <div class="leader-name">{item["user_name"]}</div>
        <p class="leader-meta">User ID {item["user_id"]}</p>
    </div>
    <div class="leader-score">
        <div class="leader-score-value">{int(item["points"]):,}P</div>
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
    if not feedback_status:
        st.info("아직 피드백 반응 기록이 없습니다.")
        return

    for item in feedback_status:
        latest_mission = str(item["latest_mission"] or "아직 기록된 미션이 없습니다.")
        rate = int(item["feedback_acceptance_rate"])

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
        <div class="feedback-score-value">{rate}%</div>
        <p class="feedback-score-label">피드백 수용률</p>
        <div class="feedback-rate-bar">
            <div class="feedback-rate-fill" style="width:{max(0, min(rate, 100))}%;"></div>
        </div>
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

total_members = sum(int(group["member_count"]) for group in groups) if groups else 0

st.markdown(
    f"""
<div class="hero-card">
    <div class="hero-kicker">🏁 GROUP RACE</div>
    <h1 class="hero-title">같이 아끼고,<br>포인트로 경쟁해요</h1>
    <p class="hero-desc">
        그룹별 personal_score 랭킹과 피드백 수용률을 한 화면에서 확인합니다.
        점수는 명확하게, 실천 현황은 비교하기 쉽게 정리했습니다.
    </p>
    <div class="hero-chip-row">
        <div class="hero-chip">참여 그룹 {len(groups)}개</div>
        <div class="hero-chip">총 멤버 {total_members}명</div>
        <div class="hero-chip">랭킹 기준 personal_score</div>
    </div>
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
        caption="현재 참여 중인 그룹 수",
    )

with metric_col2:
    _render_metric_card(
        label="랭킹 기준",
        value="personal_score",
        caption="그룹원 포인트 순위가 바로 반영됩니다",
    )

with metric_col3:
    _render_metric_card(
        label="핵심 지표",
        value="피드백 수용률",
        caption="미션 확인과 긍정 반응을 비교합니다",
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">내 그룹</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-caption">참여 중인 그룹을 확인하고, 선택한 그룹 기준으로 리더보드와 실천 현황을 봅니다.</p>',
    unsafe_allow_html=True,
)

if groups:
    chip_markup = "".join(
        [
            (
                f'<div class="group-chip">👥 {group["name"]} · '
                f"{group['member_count']}명 · {group['role']}</div>"
            )
            for group in groups
        ]
    )
    st.markdown(
        f"""
<div class="toss-card">
    <div class="card-title">내 그룹 요약</div>
    <p class="card-desc">그룹을 선택하면 아래에서 랭킹과 피드백 실천 현황을 바로 확인할 수 있습니다.</p>
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

create_col, dashboard_col = st.columns([0.95, 1.05], gap="large")

with create_col:
    st.markdown(
        """
<div class="toss-card">
    <div class="card-title">그룹 만들기</div>
    <p class="card-desc">같이 경쟁할 멤버를 초대하세요. 그룹을 만든 순간 personal_score 기준 리더보드가 열립니다.</p>
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
<div class="leaderboard-panel">
    <div class="leaderboard-header">
        <div>
            <div class="leaderboard-title">리더보드</div>
            <p class="card-desc">현재 그룹의 personal_score 순위입니다.</p>
        </div>
        <div class="leaderboard-badge">TOP RANKING</div>
    </div>
        """,
        unsafe_allow_html=True,
    )
    _render_leaderboard_cards(leaderboard)
    st.markdown("</div>", unsafe_allow_html=True)

with feedback_col:
    st.markdown(
        """
<div class="feedback-panel">
    <div class="feedback-header">
        <div>
            <div class="feedback-title">피드백 실천 현황</div>
            <p class="card-desc">확인한 피드백과 긍정 반응을 비교해 수용률을 보여줍니다.</p>
        </div>
        <div class="leaderboard-badge">ACCEPTANCE RATE</div>
    </div>
        """,
        unsafe_allow_html=True,
    )
    _render_feedback_status_cards(feedback_status)
    st.markdown("</div>", unsafe_allow_html=True)
