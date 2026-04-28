from __future__ import annotations

import inspect
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="월간 소비 성적표", page_icon="🏆", layout="wide")


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


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
        .block-container { max-width: 1120px; padding-top: 2rem; }

        .title {
            font-size: 34px;
            font-weight: 900;
            color: #0f172a;
        }

        .subtitle {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 24px;
        }

        .hero {
            padding: 36px;
            border-radius: 28px;
            background: linear-gradient(135deg, #064e3b 0%, #059669 45%, #2563eb 100%);
            color: white;
            margin: 28px 0 34px 0;
            box-shadow: 0 18px 42px rgba(5,150,105,0.24);
        }

        .hero-label {
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 16px;
            opacity: 0.95;
        }

        .hero-main {
            font-size: 31px;
            font-weight: 900;
            line-height: 1.55;
        }

        .amount {
            color: #fde047;
            font-size: 46px;
        }

        .rate {
            color: #a7f3d0;
            font-size: 40px;
        }

        .section {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin: 34px 0 16px;
        }

        .card {
            padding: 22px;
            border-radius: 22px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
            min-height: 138px;
        }

        .card-label {
            color: #64748b;
            font-weight: 700;
            font-size: 14px;
        }

        .card-value {
            color: #0f172a;
            font-weight: 900;
            font-size: 25px;
            margin-top: 12px;
        }

        .card-desc {
            color: #64748b;
            font-size: 13px;
            margin-top: 8px;
            line-height: 1.45;
        }

        .score-card {
            padding: 26px;
            border-radius: 24px;
            background: #ecfdf5;
            border: 1px solid #86efac;
            color: #065f46;
            min-height: 180px;
        }

        .score-title {
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 12px;
        }

        .strategy-card {
            padding: 26px;
            border-radius: 24px;
            background: #fff7ed;
            border: 1px solid #fdba74;
            color: #9a3412;
            min-height: 180px;
        }

        .strategy-title {
            font-size: 24px;
            font-weight: 900;
            color: #ea580c;
            margin-bottom: 12px;
        }

        .action {
            padding: 24px;
            border-radius: 22px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
        }

        .num {
            display: inline-flex;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #059669;
            color: white;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            margin-right: 12px;
        }

        .effect {
            padding: 24px;
            border-radius: 22px;
            background: #dcfce7;
            border: 1px solid #86efac;
            color: #166534;
            font-size: 18px;
            font-weight: 900;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-desc">{desc}</div>
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

    top = max(rows, key=lambda x: x.get("total_amount", 0))
    return top.get("category", "-"), top.get("total_amount", 0)


def get_improved_category(monthly_data):
    rows = monthly_data.get("category_deep", []) or monthly_data.get("category_changes", [])

    improved = []

    for row in rows:
        diff_amount = row.get("diff_amount")

        if diff_amount is None:
            total = row.get("total_amount", 0)
            prev = row.get("prev_month_amount", 0)
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
            total = row.get("total_amount", 0)
            prev = row.get("prev_month_amount", 0)
            diff_amount = total - prev

        if diff_amount > 0:
            increased.append(
                {
                    "category": row.get("category", "-"),
                    "diff_amount": diff_amount,
                    "total_amount": row.get("total_amount", 0),
                }
            )

    if not increased:
        top_category, top_amount = get_top_category(monthly_data)
        return top_category, top_amount

    worst = max(increased, key=lambda x: x["diff_amount"])
    return worst["category"], worst["diff_amount"]


def get_repeat_target(monthly_data):
    rows = (
        monthly_data.get("repeat_patterns", {}).get("top_merchants", [])
        or monthly_data.get("repeated_merchants", [])
        or monthly_data.get("repeat_merchants", [])
    )

    if not rows:
        return "-", 0, 0

    top = max(
        rows,
        key=lambda x: x.get("visit_count", x.get("count", 0)),
    )

    return (
        top.get("merchant", "-"),
        top.get("visit_count", top.get("count", 0)),
        top.get("total_amount", 0),
    )


def make_budget_usage_gauge(total_amount: int | float, monthly_budget: int | float):
    usage_rate = (total_amount / monthly_budget * 100) if monthly_budget else 0
    usage_rate = min(usage_rate, 120)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=usage_rate,
            number={"suffix": "%", "font": {"size": 40, "color": "#475569"}},
            title={"text": "월 소비 한도 사용률", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 120]},
                "bar": {"color": "#059669" if usage_rate <= 100 else "#ef4444"},
                "steps": [
                    {"range": [0, 70], "color": "#dcfce7"},
                    {"range": [70, 100], "color": "#fef3c7"},
                    {"range": [100, 120], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#2563eb", "width": 4},
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


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

            parsed_rows.append({"week": str(week), "amount": amount})

    if not parsed_rows:
        monthly_summary = data.get("monthly_summary", {})
        total = monthly_summary.get("this_month_total", 0)
        parsed_rows = [
            {"week": "1주차", "amount": total * 0.28},
            {"week": "2주차", "amount": total * 0.17},
            {"week": "3주차", "amount": total * 0.22},
            {"week": "4주차", "amount": total * 0.22},
            {"week": "5주차", "amount": total * 0.11},
        ]

    df = pd.DataFrame(parsed_rows)

    fig = px.bar(
        df,
        x="week",
        y="amount",
        text="amount",
        color="amount",
        color_continuous_scale=["#d1fae5", "#059669"],
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}원",
        textposition="outside",
        marker_line_width=0,
        width=0.55,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
        font=dict(color="#334155"),
    )

    return fig


def make_category_change_chart(monthly_data):
    rows = monthly_data.get("category_deep", []) or monthly_data.get("category_changes", [])

    parsed = []

    for row in rows:
        category = row.get("category", "-")
        total = row.get("total_amount", 0)
        prev = row.get("prev_month_amount", 0)

        diff = row.get("diff_amount")
        if diff is None:
            diff = total - prev

        parsed.append({"category": category, "diff_amount": diff})

    df = pd.DataFrame(parsed)

    if df.empty:
        df = pd.DataFrame({"category": ["데이터 없음"], "diff_amount": [0]})

    df = df.sort_values("diff_amount")

    fig = px.bar(
        df,
        x="diff_amount",
        y="category",
        orientation="h",
        text="diff_amount",
        color="diff_amount",
        color_continuous_scale=["#059669", "#f8fafc", "#ef4444"],
        color_continuous_midpoint=0,
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}원",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=20, r=40),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=True, tickformat=","),
        font=dict(color="#334155"),
    )

    return fig


def call_monthly_feedback(generate_monthly_feedback, *, member_id, month, settings):
    params = inspect.signature(generate_monthly_feedback).parameters

    kwargs = {
        "member_id": int(member_id),
        "settings": settings,
        "chunk_size": 800,
        "chunk_overlap": 120,
        "top_k": 3,
        "max_queries": 4,
    }

    if "month" in params:
        kwargs["month"] = month
    elif "target_month" in params:
        kwargs["target_month"] = month
    elif "analysis_month" in params:
        kwargs["analysis_month"] = month
    elif "year_month" in params:
        kwargs["year_month"] = month
    else:
        kwargs["month"] = month

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

    return "다음 달 고정비와 반복 소비 먼저 줄이기", "정기 결제와 반복 방문 가맹점부터 점검해보세요."


inject_css()

st.markdown('<div class="title">🏆 이번 달 소비 성적표</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">월간 보고서는 소비 총액보다 “절약 목표 달성률과 다음 달 전략”에 집중합니다.</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    member_id = st.text_input("Member ID", value="1")

with c2:
    month = st.text_input("분석 월", value="2024-03")

run = st.button("월간 소비 성적표 생성", use_container_width=True)

if run:
    from catcher_llm.config.settings import get_settings
    from catcher_llm.services.consumption_feedback.monthly_feedback import (
        generate_monthly_feedback,
    )

    settings = get_settings()

    with st.spinner("이번 달 소비 성적표를 만들고 있어요..."):
        result = call_monthly_feedback(
            generate_monthly_feedback,
            member_id=member_id,
            month=month,
            settings=settings,
        )

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

    total_amount = monthly_summary["this_month_total"]
    prev_amount = monthly_summary.get("prev_month_total", 0)
    diff_rate = monthly_summary.get("diff_rate_percent", 0)
    transaction_count = monthly_summary.get("transaction_count", 0)

    user_profile = getattr(result, "user_profile", None)
    saving_goal_text = getattr(user_profile, "saving_goal_text", None)
    goal_amount = extract_goal_amount(saving_goal_text)

    monthly_budget = max(prev_amount - goal_amount, 0) if goal_amount else prev_amount
    budget_gap = monthly_budget - total_amount

    saved_amount = max(prev_amount - total_amount, 0)

    goal_rate = (saved_amount / goal_amount * 100) if goal_amount else 0

    top_category, top_category_amount = get_top_category(monthly_data)
    improved = get_improved_category(monthly_data)
    worst_category, worst_amount = get_worst_category(monthly_data)
    repeat_merchant, repeat_count, repeat_amount = get_repeat_target(monthly_data)

    action_title, action_detail = get_action_text(feedback)

    if diff_rate < 0:
        status_text = "전월보다 소비를 줄였습니다"
    elif diff_rate > 0:
        status_text = "전월보다 소비가 늘었습니다"
    else:
        status_text = "전월과 소비가 비슷합니다"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">이번 달 소비 성적</div>
            <div class="hero-main">
                이번 달은 총 <span class="amount">{money(total_amount)}</span>을 소비했고,<br>
                <span class="rate">{status_text}</span>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">월간 핵심 성과</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("이번 달 총 소비", money(total_amount), f"전월 대비 {diff_rate:.2f}%")
    with m2:
        metric_card("실제 절약액", money(saved_amount), f"전월 소비 {money(prev_amount)} 기준")
    with m3:
        metric_card("최다 소비 카테고리", top_category, money(top_category_amount))
    with m4:
        metric_card("총 결제 건수", f"{transaction_count}건", "이번 달 전체 결제 횟수")

    st.markdown('<div class="section">소비 한도 사용률 & 월간 흐름</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    with g1:
        if goal_amount:
            st.plotly_chart(
                make_budget_usage_gauge(total_amount, monthly_budget),
                use_container_width=True,
            )

            st.markdown(
                f"""
                <div style="
                    padding:16px 20px;
                    border-radius:18px;
                    background:#f8fafc;
                    border:1px solid #e5e7eb;
                    font-weight:800;
                    color:#0f172a;
                ">
                    월 소비 한도: {money(monthly_budget)}
                    <span style="float:right;">
                        {'남은 한도' if budget_gap >= 0 else '초과 금액'}: {money(abs(budget_gap))}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("절약 목표가 없어 목표 달성률은 계산하지 않았습니다.")

    with g2:
        st.markdown("### 주차별 소비 흐름")
        st.plotly_chart(make_weekly_trend_chart(monthly_analysis), use_container_width=True)

    st.markdown('<div class="section">이번 달 카테고리 성적</div>', unsafe_allow_html=True)

    st.markdown("### 전월 대비 카테고리 증감")
    st.plotly_chart(make_category_change_chart(monthly_data), use_container_width=True)

    st.markdown('<div class="section">이번 달 판단</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        if improved:
            improved_category, improved_amount = improved
            score_title = f"가장 좋아진 소비: {improved_category}"
            score_body = f"{improved_category} 지출이 전월보다 {money(improved_amount)} 줄었습니다."
        else:
            score_title = "아직 뚜렷한 개선 카테고리가 없습니다"
            score_body = "다음 달에는 한 카테고리만 정해서 줄이는 전략이 필요합니다."

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">{score_title}</div>
                {score_body}<br><br>
                월간 보고서는 잘한 부분을 확인하고, 다음 달에 유지할 전략을 세우는 데 의미가 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        summary_title = getattr(feedback, "summary_title", f"다음 달 줄일 1순위: {worst_category}")
        scolding_message = getattr(
            feedback,
            "scolding_message",
            f"{worst_category}에서 {money(worst_amount)}만큼 개선 여지가 있습니다.",
        )

        st.markdown(
            f"""
            <div class="strategy-card">
                <div class="strategy-title">{summary_title}</div>
                {scolding_message}<br><br>
                반복 가맹점: {repeat_merchant} · {repeat_count}회 · {money(repeat_amount)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">다음 달 전략</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="action">
            <span class="num">01</span>
            <b>{action_title}</b><br>
            <span style="margin-left:54px; color:#64748b;">{action_detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">기대 효과</div>', unsafe_allow_html=True)

    expected_saving = getattr(feedback, "expected_saving_amount", 0)

    if not expected_saving:
        if repeat_count and repeat_amount:
            expected_saving = round(repeat_amount / repeat_count * 4)
        elif worst_amount:
            expected_saving = round(worst_amount * 0.2)
        else:
            expected_saving = 0

    st.markdown(
        f"""
        <div class="effect">
            다음 달에는 반복 소비와 증가 카테고리부터 줄이는 전략이 좋습니다.
            <span style="float:right;">예상 절약액 약 {money(expected_saving)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("상세 분석 & 데이터"):
        st.subheader("월간 분석 JSON")
        st.json(monthly_data)

        st.subheader("월간 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))

else:
    st.info("Member ID와 분석 월을 입력한 뒤, 월간 소비 성적표 생성을 눌러주세요.")