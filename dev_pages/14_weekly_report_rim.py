from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.ui.date_picker import DEFAULT_CALENDAR_DATE

st.set_page_config(page_title="이번 주 소비 습관 리포트", page_icon="🔁", layout="wide")


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def _to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1120px; padding-top: 2rem; }
        .title { font-size: 34px; font-weight: 900; color: #0f172a; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 24px; }

        .hero {
            padding: 34px;
            border-radius: 28px;
            background: linear-gradient(135deg, #581c87 0%, #7c3aed 45%, #2563eb 100%);
            color: white;
            margin: 28px 0 34px 0;
            box-shadow: 0 18px 42px rgba(124,58,237,0.25);
        }
        .hero-label { font-size: 15px; font-weight: 800; margin-bottom: 16px; }
        .hero-main { font-size: 31px; font-weight: 900; line-height: 1.55; }
        .amount { color: #fde047; font-size: 46px; }
        .point { color: #a7f3d0; font-size: 40px; }

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
            min-height: 140px;
        }
        .card-label { color: #64748b; font-weight: 700; font-size: 14px; }
        .card-value { color: #0f172a; font-weight: 900; font-size: 25px; margin-top: 12px; }
        .card-desc { color:#64748b; font-size:13px; margin-top:8px; line-height:1.5; }

        .habit-card {
            padding: 26px;
            border-radius: 24px;
            background: #f5f3ff;
            border: 1px solid #ddd6fe;
            color: #4c1d95;
            min-height: 170px;
        }
        .habit-title {
            font-size: 24px;
            font-weight: 900;
            color: #6d28d9;
            margin-bottom: 12px;
        }

        .danger-card {
            padding: 26px;
            border-radius: 24px;
            background: #fff7ed;
            border: 1px solid #fdba74;
            color: #9a3412;
            min-height: 170px;
        }
        .danger-title {
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
            background: #7c3aed;
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


def make_weekday_chart(weekly_analysis):
    data = _to_dict(weekly_analysis)
    rows = data.get("weekday_pattern", {}).get("weekday_breakdown", [])

    df = pd.DataFrame(
        [
            {
                "weekday": row.get("weekday", "-"),
                "amount": row.get("total_amount", 0),
            }
            for row in rows
        ]
    )

    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]
    df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)
    df = df.sort_values("weekday")

    fig = px.bar(
        df,
        x="weekday",
        y="amount",
        text="amount",
        color="amount",
        color_continuous_scale=["#ede9fe", "#7c3aed"],
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}원",
        textposition="outside",
        marker_line_width=0,
        width=0.55,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=340,
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


def make_repeat_merchant_chart(weekly_analysis):
    data = _to_dict(weekly_analysis)
    rows = data.get("repeat_patterns", {}).get("top_merchants", [])

    df = pd.DataFrame(
        [
            {
                "merchant": row.get("merchant", "-"),
                "visit_count": row.get("visit_count", row.get("count", 0)),
                "amount": row.get("total_amount", 0),
            }
            for row in rows
        ]
    )

    if df.empty:
        df = pd.DataFrame({"merchant": ["반복 가맹점 없음"], "visit_count": [0], "amount": [0]})

    df = df.sort_values("visit_count", ascending=True).tail(7)

    fig = px.bar(
        df,
        x="visit_count",
        y="merchant",
        orientation="h",
        text="visit_count",
        color="amount",
        color_continuous_scale=["#dbeafe", "#2563eb"],
    )

    fig.update_traces(
        texttemplate="%{text}회",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>방문 %{x}회<br>%{marker.color:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=340,
        margin=dict(t=20, b=20, l=20, r=30),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
        font=dict(color="#334155"),
    )

    return fig


def get_top_repeat_merchant(weekly_data):
    rows = weekly_data.get("repeat_patterns", {}).get("top_merchants", [])
    if not rows:
        return "-", 0, 0

    top = max(
        rows,
        key=lambda x: x.get("visit_count") or x.get("count") or 0,
    )

    return (
        top.get("merchant", "-"),
        top.get("visit_count", top.get("count", 0)),
        top.get("total_amount", 0),
    )


def get_peak_weekday(weekly_data):
    pattern = weekly_data.get("weekday_pattern", {})
    peak = pattern.get("peak_weekday")

    rows = pattern.get("weekday_breakdown", [])
    if peak:
        peak_amount = 0
        for row in rows:
            if row.get("weekday") == peak:
                peak_amount = row.get("total_amount", 0)
        return peak, peak_amount

    if not rows:
        return "-", 0

    top = max(rows, key=lambda x: x.get("total_amount", 0))
    return top.get("weekday", "-"), top.get("total_amount", 0)


def get_action_text(feedback):
    action_items = getattr(feedback, "action_items", []) or []

    if action_items:
        first = action_items[0]
        title = getattr(first, "title", "다음 주 소비 규칙 정하기")
        detail = (
            getattr(first, "detail", None)
            or getattr(first, "description", None)
            or "반복 소비를 줄일 수 있는 행동을 하나 정해보세요."
        )
        return title, detail

    return (
        "반복 가맹점 방문 횟수 줄이기",
        "가장 자주 방문한 가맹점의 이용 횟수를 다음 주에 1회 줄여보세요.",
    )


inject_css()

st.markdown('<div class="title">🔁 이번 주 소비 습관 리포트</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">주간 보고서는 금액보다 “이번 주 반복된 소비 습관”을 찾는 데 집중합니다.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    member_id = st.text_input("Member ID", value="1")

with c2:
    start_date = st.date_input("분석 시작일", value=DEFAULT_CALENDAR_DATE)

with c3:
    end_date = st.date_input("분석 종료일", value=DEFAULT_CALENDAR_DATE + timedelta(days=6))

run = st.button("소비 습관 리포트 생성", use_container_width=True)

if run:
    from catcher_llm.config.settings import get_settings
    from catcher_llm.services.consumption_feedback.weekly_feedback import (
        generate_weekly_feedback,
    )

    settings = get_settings()

    with st.spinner("이번 주 소비 습관을 분석하고 있어요..."):
        result = generate_weekly_feedback(
            member_id=int(member_id),
            week_start=start_date,
            week_end=end_date,
            settings=settings,
            chunk_size=800,
            chunk_overlap=120,
            top_k=3,
            max_queries=4,
        )

    if result.error:
        st.error(f"주간 피드백 생성 실패: {result.error}")
        st.stop()

    if result.feedback is None or result.weekly_analysis is None:
        st.warning("주간 분석 또는 피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback
    weekly_analysis = result.weekly_analysis
    weekly_data = _to_dict(weekly_analysis)

    weekly_summary = weekly_data["weekly_summary"]
    total_amount = weekly_summary["this_week_total"]
    prev_rate = weekly_summary["diff_rate_percent"]
    transaction_count = weekly_summary["transaction_count"]

    top_merchant, top_visit_count, top_merchant_amount = get_top_repeat_merchant(weekly_data)
    peak_weekday, peak_weekday_amount = get_peak_weekday(weekly_data)

    category_rows = weekly_data.get("category_summary", [])
    top_category = "-"
    if category_rows:
        top_item = max(category_rows, key=lambda x: x.get("total_amount", 0))
        top_category = top_item.get("category", "-")

    action_title, action_detail = get_action_text(feedback)

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">이번 주 반복 소비 습관</div>
            <div class="hero-main">
                이번 주 가장 반복된 소비는 <span class="amount">{top_merchant}</span>이고,<br>
                총 <span class="point">{top_visit_count}회</span> 방문했습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section">이번 주 습관 소비 요약</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("이번 주 총 소비", money(total_amount), f"전주 대비 {prev_rate:.2f}%")
    with m2:
        metric_card(
            "습관 소비 TOP 1",
            str(top_merchant),
            f"{top_visit_count}회 · {money(top_merchant_amount)}",
        )
    with m3:
        metric_card("가장 위험한 요일", str(peak_weekday), f"{money(peak_weekday_amount)} 사용")
    with m4:
        metric_card("총 결제 건수", f"{transaction_count}건", f"최다 카테고리: {top_category}")

    st.markdown('<div class="section">소비 습관 그래프</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("### 요일별 소비 흐름")
        st.plotly_chart(make_weekday_chart(weekly_analysis), use_container_width=True)

    with g2:
        st.markdown("### 반복 가맹점 TOP")
        st.plotly_chart(make_repeat_merchant_chart(weekly_analysis), use_container_width=True)

    st.markdown('<div class="section">이번 주 판단</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    summary_title = getattr(feedback, "summary_title", "이번 주 반복 소비를 점검해보세요.")
    scolding_message = getattr(
        feedback,
        "scolding_message",
        "이번 주 소비에서 반복적으로 나타난 지출을 줄이는 것이 중요합니다.",
    )

    with p1:
        st.markdown(
            f"""
            <div class="habit-card">
                <div class="habit-title">습관 소비 TOP 1: {top_merchant}</div>
                <b>{top_visit_count}회 반복 방문</b><br><br>
                이번 주에는 같은 가맹점에서 반복적으로 지출이 발생했습니다.
                다음 주에는 이 방문 횟수를 1회만 줄여도 습관 소비를 끊는 시작점이 됩니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            f"""
            <div class="danger-card">
                <div class="danger-title">{summary_title}</div>
                {scolding_message}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">다음 주 행동 규칙</div>', unsafe_allow_html=True)

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

    # 반복 가맹점 1회 줄이면 아낄 수 있는 금액
    if top_visit_count and top_merchant_amount:
        expected_saving = round(top_merchant_amount / top_visit_count)
    else:
        expected_saving = getattr(feedback, "expected_saving_amount", 0)

    st.markdown(
        f"""
        <div class="effect">
            다음 주에는 반복 가맹점 방문 횟수를 줄이는 것부터 시작해보세요.
            <span style="float:right;">예상 절약액 약 {money(expected_saving)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("상세 분석 & 데이터"):
        st.subheader("주간 분석 JSON")
        st.json(weekly_data)

        st.subheader("주간 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))

else:
    st.info("Member ID와 분석 기간을 선택한 뒤, 소비 습관 리포트 생성을 눌러주세요.")
