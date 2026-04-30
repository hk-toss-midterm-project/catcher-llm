from __future__ import annotations

import inspect
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)

st.set_page_config(page_title="월간 소비 리포트", page_icon="🏆", layout="wide")


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def extract_goal_amount(text: str | None) -> int:
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
        .stApp { background:#f8fafc; }

        .block-container {
            max-width: 1500px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }

        .page-title {
            font-size: 30px;
            font-weight: 950;
            color: #0f172a;
            letter-spacing: -0.7px;
            margin-bottom: 4px;
        }

        .page-subtitle {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 16px;
        }

        .hero {
            padding: 28px 32px;
            border-radius: 30px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,0.24), transparent 28%),
                linear-gradient(135deg, #0f172a 0%, #059669 48%, #2563eb 100%);
            color: white;
            box-shadow: 0 26px 70px rgba(5,150,105,0.22);
            min-height: 218px;
        }

        .hero-kicker {
            display:inline-flex;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.22);
            font-size: 13px;
            font-weight: 850;
            margin-bottom: 18px;
        }

        .hero-main {
            font-size: 32px;
            font-weight: 950;
            line-height: 1.45;
            letter-spacing: -0.7px;
        }

        .hero-main strong { color: #fde68a; }

        .hero-desc {
            margin-top: 16px;
            color: rgba(255,255,255,0.88);
            font-size: 15px;
            line-height: 1.65;
            font-weight: 650;
        }

        .hero-chip-wrap {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top:20px;
        }

        .hero-chip {
            padding: 10px 13px;
            border-radius:999px;
            background:rgba(255,255,255,0.14);
            border:1px solid rgba(255,255,255,0.18);
            font-size:13px;
            font-weight:850;
        }

        .side-panel {
            padding: 24px;
            border-radius: 30px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 18px 46px rgba(15,23,42,0.07);
            min-height: 218px;
        }

        .side-label {
            font-size: 13px;
            font-weight: 900;
            color: #64748b;
            margin-bottom: 10px;
        }

        .side-value {
            font-size: 32px;
            font-weight: 950;
            color: #0f172a;
            line-height: 1.2;
            letter-spacing: -0.7px;
        }

        .side-desc {
            margin-top: 14px;
            color: #64748b;
            font-size: 14px;
            line-height: 1.65;
            font-weight: 650;
        }

        .section {
            font-size: 19px;
            font-weight: 950;
            color: #0f172a;
            margin: 22px 0 12px;
            letter-spacing: -0.4px;
        }

        .metric-card {
            padding: 20px 22px;
            border-radius: 24px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 28px rgba(15,23,42,0.055);
            min-height: 120px;
        }

        .metric-label {
            color: #64748b;
            font-weight: 850;
            font-size: 13px;
        }

        .metric-value {
            color: #0f172a;
            font-weight: 950;
            font-size: 24px;
            margin-top: 10px;
            line-height: 1.25;
            letter-spacing: -0.5px;
        }

        .metric-desc {
            color: #94a3b8;
            font-size: 12.5px;
            margin-top: 8px;
            line-height: 1.45;
            font-weight: 650;
        }

        .chart-card {
            padding: 22px 22px 8px 22px;
            border-radius: 28px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 12px 34px rgba(15,23,42,0.06);
            min-height: 410px;
        }

        .chart-title {
            font-size: 18px;
            font-weight: 950;
            color: #0f172a;
            margin-bottom: 6px;
            letter-spacing: -0.4px;
        }

        .score-card {
            padding: 22px;
            border-radius: 26px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #065f46;
            min-height: 194px;
        }

        .score-title {
            font-size: 22px;
            font-weight: 950;
            margin-bottom: 12px;
            color: #047857;
            letter-spacing: -0.5px;
        }

        .strategy-card {
            padding: 22px;
            border-radius: 26px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            min-height: 194px;
        }

        .strategy-title {
            font-size: 22px;
            font-weight: 950;
            color: #ea580c;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }

        .action-card {
            padding: 22px;
            border-radius: 26px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 28px rgba(15,23,42,0.055);
            min-height: 138px;
        }

        .num {
            display: inline-flex;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #3182f6;
            color: white;
            align-items: center;
            justify-content: center;
            font-weight: 950;
            margin-right: 12px;
        }

        .effect-card {
            padding: 22px;
            border-radius: 26px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #166534;
            min-height: 138px;
            font-weight: 850;
            line-height: 1.65;
        }

        .vote-card {
            padding: 22px;
            border-radius: 26px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 28px rgba(15,23,42,0.055);
            min-height: 138px;
        }

        div[data-testid="stButton"] button {
            border-radius: 18px;
            height: 48px;
            font-weight: 900;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_category_rows(monthly_data):
    return (
        monthly_data.get("category_deep", [])
        or monthly_data.get("category_summary", [])
        or monthly_data.get("category_changes", [])
    )


def get_top_category(monthly_data):
    rows = get_category_rows(monthly_data)
    if not rows:
        return "-", 0

    top = max(rows, key=lambda x: safe_int(x.get("total_amount", 0)))
    return top.get("category", "-"), safe_int(top.get("total_amount", 0))


def get_improved_category(monthly_data):
    rows = monthly_data.get("category_deep", []) or monthly_data.get("category_changes", [])
    improved = []

    for row in rows:
        diff_amount = row.get("diff_amount")

        if diff_amount is None:
            total = safe_int(row.get("total_amount", 0))
            prev = safe_int(row.get("prev_month_amount", 0))
            diff_amount = total - prev

        if diff_amount < 0:
            improved.append(
                {
                    "category": row.get("category", "-"),
                    "diff_amount": diff_amount,
                }
            )

    if not improved:
        return None

    best = min(improved, key=lambda x: x["diff_amount"])
    return best["category"], abs(best["diff_amount"])


def get_worst_category(monthly_data):
    rows = monthly_data.get("category_deep", []) or monthly_data.get("category_changes", [])
    increased = []

    for row in rows:
        diff_amount = row.get("diff_amount")

        if diff_amount is None:
            total = safe_int(row.get("total_amount", 0))
            prev = safe_int(row.get("prev_month_amount", 0))
            diff_amount = total - prev

        if diff_amount > 0:
            increased.append(
                {
                    "category": row.get("category", "-"),
                    "diff_amount": diff_amount,
                    "total_amount": safe_int(row.get("total_amount", 0)),
                }
            )

    if not increased:
        top_category, top_amount = get_top_category(monthly_data)
        return top_category, top_amount

    worst = max(increased, key=lambda x: x["diff_amount"])
    return worst["category"], worst["diff_amount"]


def get_repeat_target(monthly_data):
    repeat_patterns = monthly_data.get("repeat_patterns", {})
    rows = []

    if isinstance(repeat_patterns, dict):
        rows = repeat_patterns.get("top_merchants", [])

    rows = (
        rows
        or monthly_data.get("repeated_merchants", [])
        or monthly_data.get("repeat_merchants", [])
    )

    if not rows:
        return "-", 0, 0

    top = max(rows, key=lambda x: safe_int(x.get("visit_count", x.get("count", 0))))

    return (
        top.get("merchant", "-"),
        safe_int(top.get("visit_count", top.get("count", 0))),
        safe_int(top.get("total_amount", 0)),
    )


def make_weekly_trend_chart(monthly_analysis):
    data = to_dict(monthly_analysis)

    rows = (
        data.get("weekly_trend")
        or data.get("weekly_spending_trend")
        or data.get("week_trend")
        or []
    )

    if isinstance(rows, dict):
        rows = rows.get("weekly_breakdown") or rows.get("items") or rows.get("data") or []

    parsed_rows = []

    if isinstance(rows, list):
        for idx, row in enumerate(rows, start=1):
            if isinstance(row, dict):
                week = row.get("week") or row.get("week_no") or row.get("label") or f"{idx}주차"
                amount = row.get("total_amount") or row.get("amount") or row.get("week_total") or 0
            else:
                week = f"{idx}주차"
                amount = 0

            parsed_rows.append({"week": str(week), "amount": safe_int(amount)})

    if not parsed_rows:
        monthly_summary = data.get("monthly_summary", {})
        total = safe_int(monthly_summary.get("this_month_total", 0))
        parsed_rows = [
            {"week": "1주차", "amount": total * 0.28},
            {"week": "2주차", "amount": total * 0.17},
            {"week": "3주차", "amount": total * 0.22},
            {"week": "4주차", "amount": total * 0.22},
            {"week": "5주차", "amount": total * 0.11},
        ]

    df = pd.DataFrame(parsed_rows)

    fig = px.bar(df, x="week", y="amount", text="amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color="#10b981",
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=340,
        margin=dict(t=24, b=8, l=8, r=8),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
        font=dict(color="#334155", size=12),
    )

    return fig


def make_category_change_chart(monthly_data):
    rows = monthly_data.get("category_deep", []) or monthly_data.get("category_changes", [])
    parsed = []

    for row in rows:
        category = row.get("category", "-")
        total = safe_int(row.get("total_amount", 0))
        prev = safe_int(row.get("prev_month_amount", 0))

        diff = row.get("diff_amount")
        if diff is None:
            diff = total - prev

        parsed.append({"category": category, "diff_amount": diff})

    df = pd.DataFrame(parsed)

    if df.empty:
        df = pd.DataFrame({"category": ["데이터 없음"], "diff_amount": [0]})

    df = df.sort_values("diff_amount").tail(9)

    fig = px.bar(
        df,
        x="diff_amount",
        y="category",
        orientation="h",
        text="diff_amount",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color="#60a5fa",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=340,
        margin=dict(t=24, b=8, l=8, r=32),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=True, tickformat=","),
        font=dict(color="#334155", size=12),
    )

    return fig


def call_monthly_feedback(generate_monthly_feedback, *, member_id, month, settings):
    params = inspect.signature(generate_monthly_feedback).parameters

    base_kwargs = {
        "member_id": int(member_id),
        "settings": settings,
        "chunk_size": 800,
        "chunk_overlap": 120,
        "top_k": 3,
        "max_queries": 4,
    }

    kwargs = {key: value for key, value in base_kwargs.items() if key in params}

    if "month" in params:
        kwargs["month"] = month
    elif "target_month" in params:
        kwargs["target_month"] = month
    elif "analysis_month" in params:
        kwargs["analysis_month"] = month
    elif "year_month" in params:
        kwargs["year_month"] = month
    elif "target_date" in params:
        kwargs["target_date"] = month

    return generate_monthly_feedback(**kwargs)


def get_action_text(feedback):
    action_items = getattr(feedback, "action_items", []) or []

    if action_items:
        first = action_items[0]
        title = getattr(first, "title", "다음 달 소비 전략 정하기")
        detail = (
            getattr(first, "detail", None)
            or getattr(first, "description", None)
            or "이번 달 가장 많이 쓴 카테고리를 기준으로 다음 달 절약 규칙을 정해보세요."
        )
        return title, detail

    return (
        "다음 달 증가 카테고리 1개만 줄이기",
        "이번 달 가장 많이 늘어난 카테고리를 기준으로 다음 달 절약 규칙을 정해보세요.",
    )


def render_vote_buttons(member_id: str, month: str):
    if "monthly_report_vote" not in st.session_state:
        st.session_state.monthly_report_vote = None

    selected = st.session_state.monthly_report_vote

    like_type = "primary" if selected == "like" else "secondary"
    dislike_type = "primary" if selected == "dislike" else "secondary"

    v1, v2 = st.columns(2)

    with v1:
        if st.button("👍 좋아요", use_container_width=True, type=like_type):
            st.session_state.monthly_report_vote = "like"
            st.session_state.monthly_report_vote_log = {
                "member_id": member_id,
                "month": month,
                "vote": "like",
            }
            st.rerun()

    with v2:
        if st.button("👎 싫어요", use_container_width=True, type=dislike_type):
            st.session_state.monthly_report_vote = "dislike"
            st.session_state.monthly_report_vote_log = {
                "member_id": member_id,
                "month": month,
                "vote": "dislike",
            }
            st.rerun()


def render_monthly_report(result, member_id: str, month: str):
    if result.error:
        st.error(f"월간 피드백 생성 실패: {result.error}")
        st.stop()

    if result.feedback is None or result.monthly_analysis is None:
        st.warning("월간 분석 또는 피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback
    monthly_analysis = result.monthly_analysis
    monthly_data = to_dict(monthly_analysis)
    monthly_summary = monthly_data["monthly_summary"]

    total_amount = safe_int(monthly_summary["this_month_total"])
    prev_amount = safe_int(monthly_summary.get("prev_month_total", 0))
    diff_rate = float(monthly_summary.get("diff_rate_percent", 0))
    transaction_count = safe_int(monthly_summary.get("transaction_count", 0))

    saved_amount = max(prev_amount - total_amount, 0)

    top_category, top_category_amount = get_top_category(monthly_data)
    improved = get_improved_category(monthly_data)
    worst_category, worst_amount = get_worst_category(monthly_data)
    repeat_merchant, repeat_count, repeat_amount = get_repeat_target(monthly_data)

    action_title, action_detail = get_action_text(feedback)

    if diff_rate < 0:
        status_text = "전월보다 소비가 줄어든 절약형 흐름"
        hero_result = "절약형"
    elif diff_rate > 0:
        status_text = "전월보다 소비가 늘어난 증가형 흐름"
        hero_result = "증가형"
    else:
        status_text = "전월과 비슷한 유지형 흐름"
        hero_result = "유지형"

    expected_saving = getattr(feedback, "expected_saving_amount", 0)

    if not expected_saving:
        if repeat_count and repeat_amount:
            expected_saving = round(repeat_amount / repeat_count * 4)
        elif worst_amount:
            expected_saving = round(worst_amount * 0.2)
        else:
            expected_saving = 0

    h1, h2 = st.columns([2.4, 1])

    with h1:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-kicker">🧠 LLM 월간 소비 해석 · {hero_result}</div>
                <div class="hero-main">
                    이번 달은 총 <strong>{money(total_amount)}</strong>을 소비했고,<br>
                    {status_text}입니다.
                </div>
                <div class="hero-desc">
                    단순히 총액만 보는 대신, 전월 대비 증가 카테고리와 줄어든 카테고리를 함께 비교해
                    다음 달에 가장 먼저 조정할 소비 지점을 찾았습니다.
                </div>
                <div class="hero-chip-wrap">
                    <div class="hero-chip">최다 소비 · {top_category}</div>
                    <div class="hero-chip">전월 대비 · {diff_rate:.2f}%</div>
                    <div class="hero-chip">결제 건수 · {transaction_count}건</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
            <div class="side-panel">
                <div class="side-label">이번 달 핵심 신호</div>
                <div class="side-value">{hero_result}</div>
                <div class="side-desc">
                    가장 많이 쓴 카테고리는 <b>{top_category}</b>입니다.<br>
                    해당 카테고리에서 <b>{money(top_category_amount)}</b>을 사용했습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">월간 핵심 성과</div>', unsafe_allow_html=True)

    saved_amount = prev_amount - total_amount
    is_saving = saved_amount > 0

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_card("총 소비", money(total_amount), f"{diff_rate:.2f}% 변화")

    with m2:
        if is_saving:
            st.markdown(
                f"""
            <div class="metric-card" style="
                background:#ecfdf5;
                border:1px solid #bbf7d0;
                color:#166534;
            ">
                <div class="metric-label">절약 금액</div>
                <div class="metric-value">{money(saved_amount)}</div>
                <div class="metric-desc">이번 달 소비 절감 👍</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f"""
            <div class="metric-card" style="
                background:#fef2f2;
                border:1px solid #fecaca;
                color:#991b1b;
            ">
                <div class="metric-label">초과 소비</div>
                <div class="metric-value">{money(abs(saved_amount))}</div>
                <div class="metric-desc">전월 대비 지출 증가 ⚠️</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with m3:
        metric_card("최대 소비 카테고리", top_category, money(top_category_amount))

    st.markdown('<div class="section">월간 소비 대시보드</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([1.25, 1.25, 1])

    with d1:
        with st.container(border=True):
            st.markdown("### 주차별 소비 흐름")
            st.plotly_chart(make_weekly_trend_chart(monthly_analysis), use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown("### 전월 대비 카테고리 증감")
            st.plotly_chart(make_category_change_chart(monthly_data), use_container_width=True)

    with d3:
        if improved:
            improved_category, improved_amount = improved
            score_title = f"좋아진 소비<br>{improved_category}"
            score_body = (
                f"{improved_category} 지출이 전월보다 <b>{money(improved_amount)}</b> 줄었습니다."
            )
        else:
            score_title = "아직 뚜렷한<br>개선 없음"
            score_body = "다음 달에는 한 카테고리만 정해서 줄이는 전략이 필요합니다."

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">{score_title}</div>
                {score_body}<br><br>
                월간 리포트는 잘한 부분을 확인하고 유지할 전략을 세우는 데 의미가 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        summary_title = getattr(feedback, "summary_title", f"다음 달 줄일 1순위: {worst_category}")
        scolding_message = getattr(
            feedback,
            "scolding_message",
            f"{worst_category}에서 {money(worst_amount)}만큼 개선 여지가 있습니다.",
        )

        if "월간 소비 피드백" in summary_title:
            summary_title = "LLM 소비 코멘트"

        st.markdown(
            f"""
            <div class="strategy-card">
                <div class="strategy-title">{summary_title}</div>
                {scolding_message}<br><br>
                반복 가맹점: {repeat_merchant} · {repeat_count}회
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">다음 달 실행 플랜</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns([1.55, 1, 0.85])

    with a1:
        st.markdown(
            f"""
            <div class="action-card">
                <span class="num">01</span>
                <b>{action_title}</b><br>
                <span style="margin-left:52px; color:#64748b; font-weight:650;">
                    {action_detail}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            f"""
            <div class="effect-card">
                다음 달에는 소비 전체를 줄이기보다<br>
                <b>{worst_category}</b>부터 조정하는 전략이 좋습니다.
                <br><br>
                예상 절약액 약 <b>{money(expected_saving)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:
        st.markdown(
            """
            <div class="vote-card">
                <b>이 리포트는 어땠나요?</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_vote_buttons(member_id, month)

    with st.expander("상세 분석 & 데이터"):
        st.subheader("월간 분석 JSON")
        st.json(monthly_data)

        st.subheader("월간 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))

        st.subheader("사용자 반응 로그")
        st.json(st.session_state.get("monthly_report_vote_log", {}))


inject_css()
render_date_picker_styles()

if "monthly_report_result" not in st.session_state:
    st.session_state.monthly_report_result = None

if "monthly_report_params" not in st.session_state:
    st.session_state.monthly_report_params = {
        "member_id": "1",
        "month": DEFAULT_CALENDAR_MONTH,
    }

top1, top2 = st.columns([1.3, 1])

with top1:
    st.markdown('<div class="page-title">🏆 월간 소비 리포트</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">한 달 소비를 전월과 비교해 LLM이 절약 포인트를 해석합니다.</div>',
        unsafe_allow_html=True,
    )

with top2:
    f1, f2, f3 = st.columns([1, 1, 0.9])

    with f1:
        member_id = st.text_input(
            "Member ID",
            value=st.session_state.monthly_report_params["member_id"],
        )

    with f2:
        month = select_month(
            "분석 월",
            default_month=st.session_state.monthly_report_params["month"],
            key="monthly_report_month",
        )

    with f3:
        st.write("")
        run = st.button("생성", use_container_width=True)

if run:
    st.session_state.monthly_report_vote = None
    st.session_state.monthly_report_vote_log = {}

    from catcher_llm.config.settings import get_settings
    from catcher_llm.services.consumption_feedback.monthly_feedback import (
        generate_monthly_feedback,
    )

    settings = get_settings()

    with st.spinner("이번 달 소비 리포트를 만들고 있어요..."):
        result = call_monthly_feedback(
            generate_monthly_feedback,
            member_id=member_id,
            month=month,
            settings=settings,
        )

    st.session_state.monthly_report_result = result
    st.session_state.monthly_report_params = {
        "member_id": member_id,
        "month": month,
    }

if st.session_state.monthly_report_result is not None:
    render_monthly_report(
        st.session_state.monthly_report_result,
        st.session_state.monthly_report_params["member_id"],
        st.session_state.monthly_report_params["month"],
    )
else:
    st.info("Member ID와 분석 월을 입력한 뒤, 생성을 눌러주세요.")


def render_monthly_report(result, member_id: str, month: str):
    if result.error:
        st.error(f"월간 피드백 생성 실패: {result.error}")
        st.stop()

    feedback = result.feedback
    monthly_analysis = result.monthly_analysis
    monthly_data = to_dict(monthly_analysis)
    monthly_summary = monthly_data["monthly_summary"]

    total_amount = safe_int(monthly_summary["this_month_total"])
    prev_amount = safe_int(monthly_summary.get("prev_month_total", 0))
    diff_rate = float(monthly_summary.get("diff_rate_percent", 0))
    transaction_count = safe_int(monthly_summary.get("transaction_count", 0))

    # 핵심 데이터
    top_category, top_category_amount = get_top_category(monthly_data)
    improved = get_improved_category(monthly_data)
    worst_category, worst_amount = get_worst_category(monthly_data)
    repeat_merchant, repeat_count, repeat_amount = get_repeat_target(monthly_data)

    action_title, action_detail = get_action_text(feedback)

    # =========================
    # 🔥 반복가맹점 표시 로직 (핵심 수정)
    # =========================
    if repeat_count > 0 and repeat_merchant != "-":
        repeat_text = f"반복 가맹점: {repeat_merchant} · {repeat_count}회"
    else:
        repeat_text = f"우선 점검 카테고리: {worst_category}"

    # =========================
    # 히어로 판단
    # =========================
    if diff_rate < 0:
        status_text = "전월보다 소비가 줄어든 절약형 흐름입니다."
        hero_result = "절약형"
    elif diff_rate > 0:
        status_text = "전월보다 소비가 늘어난 증가형 흐름입니다."
        hero_result = "증가형"
    else:
        status_text = "전월과 비슷한 유지형 흐름입니다."
        hero_result = "유지형"

    expected_saving = getattr(feedback, "expected_saving_amount", 0)
    if not expected_saving:
        expected_saving = round(worst_amount * 0.2)

    # =========================
    # 🎯 히어로
    # =========================
    h1, h2 = st.columns([2.4, 1])

    with h1:
        st.markdown(
            f"""
        <div class="hero">
            <div class="hero-kicker">🧠 LLM 월간 소비 해석 · {hero_result}</div>
            <div class="hero-main">
                이번 달은 총 <strong>{money(total_amount)}</strong>을 소비했고,<br>
                {status_text}
            </div>
            <div class="hero-desc">
                단순 총액이 아니라, 증가한 카테고리 기준으로 다음 달 절약 포인트를 도출했습니다.
            </div>
            <div class="hero-chip-wrap">
                <div class="hero-chip">최다 소비 · {top_category}</div>
                <div class="hero-chip">전월 대비 · {diff_rate:.2f}%</div>
                <div class="hero-chip">결제 건수 · {transaction_count}건</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
        <div class="side-panel">
            <div class="side-label">이번 달 핵심 신호</div>
            <div class="side-value">{hero_result}</div>
            <div class="side-desc">
                가장 많이 쓴 카테고리는 <b>{top_category}</b><br>
                {money(top_category_amount)} 사용
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # =========================
    # 카드
    # =========================
    st.markdown('<div class="section">월간 핵심 성과</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("총 소비", money(total_amount), f"{diff_rate:.2f}% 변화")
    with m2:
        metric_card("절약 금액", money(prev_amount - total_amount), "이번 달 소비 절감 효과")
    with m3:
        metric_card("최대 소비", top_category)
    with m4:
        metric_card("결제 수", f"{transaction_count}건")

    # =========================
    # 그래프
    # =========================
    st.markdown('<div class="section">월간 소비 대시보드</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([1.2, 1.2, 1])

    with d1:
        with st.container(border=True):
            st.markdown("### 주차별 소비 흐름")
            st.plotly_chart(make_weekly_trend_chart(monthly_analysis), use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown("### 카테고리 증감")
            st.plotly_chart(make_category_change_chart(monthly_data), use_container_width=True)

    with d3:
        if improved:
            cat, amt = improved
            st.success(f"좋아진 소비\n\n{cat} ↓ {money(amt)}")
        else:
            st.info("개선 카테고리 없음")

        st.markdown("")

        summary_title = getattr(feedback, "summary_title", f"{worst_category} 줄이기")
        scolding_message = getattr(
            feedback,
            "scolding_message",
            f"{worst_category}에서 {money(worst_amount)} 개선 가능",
        )

        st.warning(f"""
        **{summary_title}**

        {scolding_message}

        {repeat_text}
        """)

    # =========================
    # 실행 플랜
    # =========================
    st.markdown('<div class="section">다음 달 실행 플랜</div>', unsafe_allow_html=True)

    a1, a2 = st.columns([1.6, 1])

    with a1:
        st.markdown(
            f"""
        <div class="action-card">
            <span class="num">01</span>
            <b>{action_title}</b><br>
            <span style="margin-left:52px;">{action_detail}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            f"""
        <div class="effect-card">
            예상 절약액<br><br>
            <b>{money(expected_saving)}</b>
        </div>
        """,
            unsafe_allow_html=True,
        )
