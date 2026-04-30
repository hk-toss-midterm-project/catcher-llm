from __future__ import annotations

import html
import re
import sqlite3
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.daily_report_defaults import get_default_daily_report_selection
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_daily_date,
)

_USER_SCORE_COLUMN = "personal_score"

st.set_page_config(page_title="오늘의 소비 알림장", page_icon="🚨", layout="wide")


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _quote_sqlite_identifier(identifier: str) -> str:
    return identifier.replace('"', '""')


def _get_daily_report_sqlite_db_path() -> str:
    return str(get_settings().sqlite_db_path)


def _get_user_score_column(connection: sqlite3.Connection) -> str | None:
    rows = connection.execute('PRAGMA table_info("users")').fetchall()
    column_names = {str(row[1]) for row in rows}
    return _USER_SCORE_COLUMN if _USER_SCORE_COLUMN in column_names else None


def add_user_point(member_id: int, point: int = 50) -> None:
    with sqlite3.connect(_get_daily_report_sqlite_db_path()) as connection:
        score_column = _get_user_score_column(connection)
        if score_column is None:
            return

        escaped_score_column = _quote_sqlite_identifier(score_column)
        connection.execute(
            f"""
            UPDATE users
            SET "{escaped_score_column}" =
                CAST(COALESCE(NULLIF("{escaped_score_column}", ''), '0') AS INTEGER) + ?
            WHERE id = ?
            """,
            (point, member_id),
        )
        connection.commit()


def money(v: int | float) -> str:
    return f"{v:,.0f}원"


def pct(v: int | float) -> str:
    return f"{v:.1f}%"


def extract_monthly_goal(text: str | None) -> int:
    if not text:
        return 0

    match = re.search(r"(\d+)\s*만원", text)
    if match:
        return int(match.group(1)) * 10_000

    match = re.search(r"(\d+)\s*원", text)
    if match:
        return int(match.group(1))

    return 0


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1460px;
            padding-top: 2.1rem;
            padding-bottom: 2rem;
        }

        .title {
            font-size: 38px;
            font-weight: 950;
            color: #0f172a;
            margin-bottom: 8px;
            letter-spacing: -0.8px;
        }

        .subtitle {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 24px;
        }

        .hero-label {
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 12px;
            opacity: 0.92;
        }

        .hero-main {
            font-size: 30px;
            font-weight: 900;
            line-height: 1.45;
        }

        .section {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin: 24px 0 12px;
        }

        .mini-card {
            padding: 18px;
            border-radius: 20px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            min-height: 105px;
        }

        .mini-label {
            color: #64748b;
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 8px;
        }

        .mini-value {
            color: #0f172a;
            font-weight: 900;
            font-size: 23px;
        }

        .problem-box {
            padding: 26px;
            border-radius: 26px;
            background: #eff6ff;
            border: 1px solid #93c5fd;
            color: #1e3a8a;
            box-shadow: 0 8px 22px rgba(59, 130, 246, 0.08);
            min-height: 225px;
        }

        .problem-title {
            font-size: 25px;
            font-weight: 900;
            color: #0052CC;
            margin-bottom: 14px;
        }

        .problem-text {
            font-size: 16px;
            line-height: 1.75;
            font-weight: 700;
        }

        .action-card {
            padding: 26px;
            border-radius: 26px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            min-height: 225px;
        }

        .action-title {
            font-size: 25px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 18px;
        }

        .num {
            display: inline-flex;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #0052CC;
            color: white;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            margin-right: 12px;
        }

        .action-main {
            font-size: 17px;
            font-weight: 900;
            color: #0f172a;
        }

        .action-detail {
            margin-left: 52px;
            margin-top: 10px;
            color: #64748b;
            line-height: 1.65;
            font-weight: 650;
            font-size: 15px;
        }

        .effect {
            padding: 26px;
            border-radius: 26px;
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            color: #0c4a6e;
            font-size: 17px;
            font-weight: 800;
            min-height: 225px;
        }

        .effect-title {
            font-size: 25px;
            font-weight: 900;
            margin-bottom: 18px;
        }

        .saving-box {
            margin-top: 18px;
            padding: 16px;
            border-radius: 18px;
            background: white;
            border: 1px solid #bae6fd;
            color: #0052CC;
            font-size: 20px;
            font-weight: 900;
            text-align: center;
        }

        .feedback-box {
            padding: 22px;
            border-radius: 22px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            margin-top: 8px;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px;
            border: 1px solid #e5e7eb;
        }

        h3 {
            margin-top: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_main_category(daily_analysis):
    changes = daily_analysis.stable_metrics.category_ratio_changes
    if not changes:
        return "-", 0.0

    top = max(changes, key=lambda item: item.today_ratio_percent)
    return top.category, top.today_ratio_percent


def make_category_chart(daily_analysis):
    total = daily_analysis.stable_metrics.today_total
    changes = daily_analysis.stable_metrics.category_ratio_changes

    df = pd.DataFrame(
        [
            {
                "category": item.category,
                "amount": total * item.today_ratio_percent / 100,
            }
            for item in changes
            if item.today_ratio_percent > 0
        ]
    )

    if df.empty:
        df = pd.DataFrame({"category": ["데이터 없음"], "amount": [0]})

    fig = px.pie(
        df,
        names="category",
        values="amount",
        hole=0.68,
        color_discrete_sequence=[
            "#0052CC",
            "#0066FF",
            "#3B82F6",
            "#60A5FA",
            "#93C5FD",
            "#BFDBFE",
        ],
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        insidetextfont=dict(size=12, color="white"),
        marker=dict(line=dict(color="white", width=3)),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f}원<br>%{percent}<extra></extra>",
    )

    fig.update_layout(
        height=285,
        margin=dict(t=5, b=5, l=5, r=5),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>오늘</b><br>{money(total)}",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="#0f172a"),
                showarrow=False,
            )
        ],
    )

    return fig


def make_hour_chart(daily_analysis):
    rows = []

    for item in daily_analysis.time_slot_analysis.time_slots:
        data = item.model_dump()

        slot = (
            data.get("time_slot")
            or data.get("slot")
            or data.get("label")
            or data.get("time_range")
            or data.get("period")
            or "시간대"
        )

        amount = (
            data.get("today_amount")
            or data.get("current_amount")
            or data.get("amount")
            or data.get("today_total")
            or 0
        )

        rows.append({"time_slot": str(slot), "amount": amount})

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame({"time_slot": ["데이터 없음"], "amount": [0]})

    fig = px.bar(
        df,
        x="time_slot",
        y="amount",
        text="amount",
        color="amount",
        color_continuous_scale=["#dbeafe", "#0052CC"],
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}원",
        textposition="outside",
        marker_line_width=0,
        width=0.52,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=285,
        margin=dict(t=20, b=20, l=15, r=15),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        coloraxis_showscale=False,
        bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"),
        yaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
        xaxis=dict(tickfont=dict(size=10)),
    )

    return fig


def make_budget_gauge(today_amount: int | float, daily_budget: int | float):
    usage_rate = (today_amount / daily_budget * 100) if daily_budget else 0
    usage_rate = min(usage_rate, 120)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=usage_rate,
            number={"suffix": "%", "font": {"size": 34, "color": "#64748b"}},
            title={"text": "소비 한도 사용률", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 120]},
                "bar": {"color": "#0052CC"},
                "steps": [
                    {"range": [0, 70], "color": "#dcfce7"},
                    {"range": [70, 100], "color": "#fef3c7"},
                    {"range": [100, 120], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        )
    )

    fig.update_layout(
        height=285,
        margin=dict(t=35, b=5, l=15, r=15),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def get_action_text(feedback):
    actions = getattr(feedback, "action_items", []) or []

    if actions:
        first = actions[0]
        title = getattr(first, "title", None) or "내일 소비 규칙 정하기"
        detail = (
            getattr(first, "detail", None)
            or getattr(first, "description", None)
            or "오늘 가장 많이 쓴 소비를 내일 하루만 줄여보세요."
        )
        return clean_text(title), clean_text(detail)

    return "내일은 배달 음식 주문하지 않기", "식비 비중을 낮추기 위해 하루만 배달을 쉬어보세요."


def render_report_feedback():
    if "daily_report_feedback" not in st.session_state:
        st.session_state.daily_report_feedback = None

    if "daily_report_rewarded" not in st.session_state:
        st.session_state.daily_report_rewarded = False

    st.markdown('<div class="section">보고서 피드백</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feedback-box">
            <div style="font-weight:900; font-size:18px; color:#0f172a; margin-bottom:6px;">
                오늘의 보고서가 도움이 되었나요?
            </div>
            <div style="color:#64748b; font-size:14px; margin-bottom:16px;">
                피드백은 한 번만 포인트가 적립됩니다.
            </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1, 1, 4])

    like_selected = st.session_state.daily_report_feedback == "like"
    dislike_selected = st.session_state.daily_report_feedback == "dislike"

    def reward_once():
        if not st.session_state.daily_report_rewarded:
            add_user_point(int(st.session_state.member_id), 50)
            st.session_state.daily_report_rewarded = True

    with c1:
        if st.button(
            "👍 좋아요",
            type="primary" if like_selected else "secondary",
            use_container_width=True,
            key="daily_report_like",
        ):
            st.session_state.daily_report_feedback = "like"
            reward_once()
            st.rerun()

    with c2:
        if st.button(
            "👎 싫어요",
            type="primary" if dislike_selected else "secondary",
            use_container_width=True,
            key="daily_report_dislike",
        ):
            st.session_state.daily_report_feedback = "dislike"
            reward_once()
            st.rerun()

    if st.session_state.daily_report_feedback == "like":
        st.success("좋아요가 저장되었습니다.")
    elif st.session_state.daily_report_feedback == "dislike":
        st.warning("싫어요가 저장되었습니다.")



inject_css()
render_date_picker_styles()

st.markdown('<div class="title">🚨 오늘의 소비 알림장</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">일간 보고서는 분석보다 "내일 바로 바꿀 행동"에 집중합니다.</div>',
    unsafe_allow_html=True,
)

default_selection = get_default_daily_report_selection()
col1, col2, col3 = st.columns(3)

with col1:
    member_id = st.number_input(
        "Member ID",
        min_value=1,
        value=default_selection.member_id,
        step=1,
    )

with col2:
    analysis_date = select_daily_date(
        "분석 기준일",
        default=DEFAULT_CALENDAR_DATE,
        key="daily_report_analysis_date",
    )

with col3:
    previous_date = analysis_date - timedelta(days=1)

    st.markdown(
        f"""
        <div style="
            font-size:14px;
            color:#0f172a;
            margin-bottom:12px;
        ">
            비교 기준일
        </div>
        <div style="
            font-size:16px;
            color:#374151;
            font-weight:400;
            padding-top:2px;
        ">
            {previous_date.isoformat()}
        </div>
        """,
        unsafe_allow_html=True,
    )

run = st.button("오늘의 소비 알림장 생성", use_container_width=True)

st.session_state.member_id = int(member_id)

if run:
    from catcher_llm.services.consumption_feedback.daily_feedback import generate_daily_feedback

    settings = get_settings()

    with st.spinner("오늘의 소비 알림장을 생성하고 있어요..."):
        result = generate_daily_feedback(
            member_id=int(member_id),
            analysis_date=analysis_date,
            previous_date=previous_date,
            settings=settings,
            chunk_size=800,
            chunk_overlap=120,
            top_k=3,
            max_queries=4,
        )

    st.session_state.daily_report_result = result
    st.session_state.daily_report_feedback = None
    st.session_state.daily_report_rewarded = False

result = st.session_state.get("daily_report_result")

if result is None:
    st.info("Member ID와 날짜를 선택한 뒤, 오늘의 소비 알림장 생성을 눌러주세요.")
    st.stop()

if result.error:
    st.error(f"일간 피드백 생성 실패: {result.error}")
    st.stop()

if result.feedback is None:
    st.warning("피드백 결과가 비어 있습니다.")
    st.stop()

feedback = result.feedback
daily_analysis = result.daily_analysis

if daily_analysis is None:
    st.warning("일일 분석 데이터가 비어 있어 리포트를 만들 수 없습니다.")
    st.stop()

today_amount = daily_analysis.stable_metrics.today_total
past_average = daily_analysis.stable_metrics.past_daily_stable_average
change_rate = daily_analysis.previous_day_comparison.amount_diff_rate_percent or 0
previous_amount = today_amount - daily_analysis.previous_day_comparison.amount_diff
peak_time = daily_analysis.time_slot_analysis.peak_slot or "-"
main_category, main_ratio = get_main_category(daily_analysis)

user_profile = getattr(result, "user_profile", None)
saving_goal_text = getattr(user_profile, "saving_goal_text", None)
monthly_goal = extract_monthly_goal(saving_goal_text)
daily_saving_goal = round(monthly_goal / 30) if monthly_goal else 0
daily_budget = max(round(past_average - daily_saving_goal), 0)
budget_gap = daily_budget - today_amount if daily_budget else 0

llm_summary_title = clean_text(feedback.summary_title)
llm_feedback_message = clean_text(feedback.scolding_message)
llm_tomorrow_mission = clean_text(feedback.tomorrow_mission)
action_title, action_detail = get_action_text(feedback)

if previous_amount > today_amount:
    saving_amount = previous_amount - today_amount
    saving_box_title = "오늘 절약 금액"
else:
    saving_amount = daily_saving_goal if daily_saving_goal > 0 else round(today_amount * 0.1)
    saving_box_title = "내일 절약 목표"

if change_rate < 0:
    hero_gradient = "linear-gradient(135deg, #0052CC 0%, #0066FF 45%, #3B82F6 100%)"
    hero_shadow = "rgba(3, 102, 255, 0.24)"
    amount_color = "#c7f0d8"
else:
    hero_gradient = "linear-gradient(135deg, #DC2626 0%, #EF4444 45%, #F87171 100%)"
    hero_shadow = "rgba(239, 68, 68, 0.24)"
    amount_color = "#FED7AA"

st.markdown(
    f"""
    <div style="
        padding: 30px 34px;
        border-radius: 28px;
        background: {hero_gradient};
        color: white;
        margin: 22px 0 24px 0;
        box-shadow: 0 20px 42px {hero_shadow};
    ">
        <div class="hero-label">오늘의 소비 경고</div>
        <div class="hero-main">
            오늘은 <span style="color: {amount_color}; font-size: 44px;">{money(today_amount)}</span>을 썼고,<br>
            가장 많이 새는 곳은 <span style="color: #ffffff; font-size: 38px;">{main_category}</span>입니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section">오늘 핵심만 보기</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns([1, 1, 1.2, 1])

with m1:
    metric_card("오늘 소비", money(today_amount))
with m2:
    metric_card("전일 대비", f"{change_rate:.2f}%")
with m3:
    metric_card("피크 시간대", peak_time)
with m4:
    metric_card(f"{main_category} 비중", pct(main_ratio))

st.markdown('<div class="section">오늘 소비 대시보드</div>', unsafe_allow_html=True)

dashboard_left, dashboard_right = st.columns([2.1, 1])

with dashboard_left:
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("### 어디에 썼나")
        st.plotly_chart(make_category_chart(daily_analysis), use_container_width=True)

    with chart2:
        st.markdown("### 언제 썼나")
        st.plotly_chart(make_hour_chart(daily_analysis), use_container_width=True)

with dashboard_right:
    st.markdown("### 소비 한도")

    if daily_budget:
        st.plotly_chart(
            make_budget_gauge(today_amount, daily_budget),
            use_container_width=True,
        )

        remaining_label = "남은 금액" if budget_gap >= 0 else "초과 금액"

        st.markdown(
            f"""
            <div style="
                padding:14px 18px;
                border-radius:18px;
                background:#f8fafc;
                border:1px solid #e5e7eb;
                font-weight:800;
                color:#0f172a;
                text-align:center;
                font-size:14px;
                margin-top:-8px;
            ">
                권장 한도: {money(daily_budget)}<br>
                <span style="color:#64748b; font-weight:600;">
                    {remaining_label}: {money(abs(budget_gap))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("절약 목표 없음")

st.markdown(
    '<div class="section">오늘의 판단 · 내일 행동 · 기대 효과</div>', unsafe_allow_html=True
)

bottom1, bottom2, bottom3 = st.columns([1.25, 1.15, 1])

with bottom1:
    st.markdown(
        f"""
        <div class="problem-box">
            <div class="problem-title">{llm_summary_title}</div>
            <div class="problem-text">
                {llm_feedback_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with bottom2:
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">내일 미션</div>
            <div>
                <span class="num">01</span>
                <span class="action-main">{llm_tomorrow_mission}</span>
            </div>
            <div class="action-detail">
                {action_title}<br>
                {action_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with bottom3:
    st.markdown(
        f"""
        <div class="effect">
            <div class="effect-title">기대 효과</div>
            <div>
                내일 이 행동 하나만 지켜도 소비 패턴을 바꾸는 시작점이 됩니다.
            </div>
            <div class="saving-box">
                {saving_box_title}<br>
                약 {money(saving_amount)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("LLM 피드백 근거와 실행 항목"):
    st.subheader("피드백 근거")
    if feedback.key_evidences:
        st.dataframe(
            [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in feedback.key_evidences
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("표시할 피드백 근거가 없습니다.")

    st.subheader("오늘 할 일")
    if feedback.action_items:
        st.dataframe(
            [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in feedback.action_items
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("표시할 행동 항목이 없습니다.")

feedback_col1, feedback_col2 = st.columns([1.5, 1])

with feedback_col1:
    render_report_feedback()

with feedback_col2:
    st.markdown('<div class="section">상세 데이터</div>', unsafe_allow_html=True)
    with st.expander("상세 분석 & 데이터"):
        st.subheader("일일 분석 JSON")
        st.json(
            daily_analysis.model_dump() if hasattr(daily_analysis, "model_dump") else daily_analysis
        )

        st.subheader("최종 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))