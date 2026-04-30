from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_week_range,
)

st.set_page_config(page_title="이번 주 소비 습관 리포트", page_icon="🔁", layout="wide")


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def _to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _html_text(value) -> str:
    """리포트 카드에 넣을 LLM 텍스트를 HTML 안전 문자열로 변환한다."""
    return html.escape(str(value or "")).replace("\n", "<br>")


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


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
            color:#64748b;
            font-size:14px;
            margin-bottom:16px;
        }

        .hero {
            padding: 28px 32px;
            border-radius: 30px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,0.24), transparent 28%),
                linear-gradient(135deg, #111827 0%, #7c3aed 48%, #2563eb 100%);
            color:white;
            box-shadow: 0 26px 70px rgba(124,58,237,0.22);
            min-height: 218px;
        }

        .hero-kicker {
            display:inline-flex;
            padding:7px 12px;
            border-radius:999px;
            background:rgba(255,255,255,0.15);
            border:1px solid rgba(255,255,255,0.22);
            font-size:13px;
            font-weight:850;
            margin-bottom:18px;
        }

        .hero-main {
            font-size:32px;
            font-weight:950;
            line-height:1.45;
            letter-spacing:-0.7px;
        }

        .hero-main strong {
            color:#fde68a;
        }

        .hero-desc {
            margin-top:16px;
            color:rgba(255,255,255,0.88);
            font-size:15px;
            line-height:1.65;
            font-weight:650;
        }

        .hero-chip-wrap {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top:20px;
        }

        .hero-chip {
            padding:10px 13px;
            border-radius:999px;
            background:rgba(255,255,255,0.14);
            border:1px solid rgba(255,255,255,0.18);
            font-size:13px;
            font-weight:850;
        }

        .side-panel {
            padding:24px;
            border-radius:30px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 18px 46px rgba(15,23,42,0.07);
            min-height:218px;
        }

        .side-label {
            font-size:13px;
            font-weight:900;
            color:#64748b;
            margin-bottom:10px;
        }

        .side-value {
            font-size:32px;
            font-weight:950;
            color:#0f172a;
            line-height:1.2;
            letter-spacing:-0.7px;
        }

        .side-desc {
            margin-top:14px;
            color:#64748b;
            font-size:14px;
            line-height:1.65;
            font-weight:650;
        }

        .section {
            font-size:19px;
            font-weight:950;
            color:#0f172a;
            margin:22px 0 12px;
            letter-spacing:-0.4px;
        }

        .metric-card {
            padding:20px 22px;
            border-radius:24px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:120px;
        }

        .metric-label {
            color:#64748b;
            font-weight:850;
            font-size:13px;
        }

        .metric-value {
            color:#0f172a;
            font-weight:950;
            font-size:24px;
            margin-top:10px;
            line-height:1.25;
            letter-spacing:-0.5px;
        }

        .metric-desc {
            color:#94a3b8;
            font-size:12.5px;
            margin-top:8px;
            line-height:1.45;
            font-weight:650;
        }

        .habit-card {
            padding:22px;
            border-radius:26px;
            background:#f5f3ff;
            border:1px solid #ddd6fe;
            color:#4c1d95;
            min-height:194px;
        }

        .habit-title {
            font-size:22px;
            font-weight:950;
            color:#6d28d9;
            margin-bottom:12px;
            letter-spacing:-0.5px;
        }

        .danger-card {
            padding:22px;
            border-radius:26px;
            background:#fff7ed;
            border:1px solid #fed7aa;
            color:#9a3412;
            min-height:194px;
        }

        .danger-title {
            font-size:22px;
            font-weight:950;
            color:#ea580c;
            margin-bottom:12px;
            letter-spacing:-0.5px;
        }

        .action-card {
            padding:22px;
            border-radius:26px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:138px;
        }

        .num {
            display:inline-flex;
            width:36px;
            height:36px;
            border-radius:50%;
            background:#3182f6;
            color:white;
            align-items:center;
            justify-content:center;
            font-weight:950;
            margin-right:12px;
        }

        .effect-card {
            padding:22px;
            border-radius:26px;
            background:#ecfdf5;
            border:1px solid #bbf7d0;
            color:#166534;
            min-height:138px;
            font-weight:850;
            line-height:1.65;
        }

        .vote-card {
            padding:22px;
            border-radius:26px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:138px;
        }

        div[data-testid="stButton"] button {
            border-radius:18px;
            height:48px;
            font-weight:900;
        }

        div[data-testid="stTextInput"] input {
            border-radius:14px;
        }

        .st-key-weekly_like_btn_active button {
            background-color:#22c55e !important;
            color:white !important;
            border:1px solid #22c55e !important;
        }

        .st-key-weekly_dislike_btn_active button {
            background-color:#ef4444 !important;
            color:white !important;
            border:1px solid #ef4444 !important;
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


def make_weekday_chart(weekly_analysis):
    data = _to_dict(weekly_analysis)
    rows = data.get("weekday_pattern", {}).get("weekday_breakdown", [])

    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]

    df = pd.DataFrame(
        [
            {
                "weekday": row.get("weekday", "-"),
                "amount": safe_int(row.get("total_amount", 0)),
            }
            for row in rows
        ]
    )

    if df.empty:
        df = pd.DataFrame({"weekday": weekday_order, "amount": [0] * 7})

    for day in weekday_order:
        if day not in df["weekday"].values:
            df = pd.concat(
                [df, pd.DataFrame([{"weekday": day, "amount": 0}])],
                ignore_index=True,
            )

    df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)
    df = df.sort_values("weekday")

    fig = px.bar(df, x="weekday", y="amount", text="amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color="#7c3aed",
        marker_line_width=0,
        width=0.55,
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


def make_repeat_merchant_chart(weekly_analysis):
    data = _to_dict(weekly_analysis)
    rows = data.get("repeat_patterns", {}).get("top_merchants", [])

    parsed = []

    for row in rows:
        merchant = row.get("merchant", "-")
        count = safe_int(row.get("visit_count", row.get("count", 0)))
        amount = safe_int(row.get("total_amount", 0))

        parsed.append(
            {
                "merchant": merchant,
                "visit_count": count,
                "amount": amount,
            }
        )

    df = pd.DataFrame(parsed)

    if df.empty:
        df = pd.DataFrame(
            {
                "merchant": ["반복 가맹점 없음"],
                "visit_count": [0],
                "amount": [0],
            }
        )

    df = df.sort_values("visit_count", ascending=True).tail(7)

    fig = px.bar(df, x="visit_count", y="merchant", orientation="h", text="visit_count")

    fig.update_traces(
        texttemplate="%{text}회",
        textposition="outside",
        marker_color="#60a5fa",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>방문 %{x}회<extra></extra>",
    )

    fig.update_layout(
        height=340,
        margin=dict(t=24, b=8, l=8, r=32),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
        font=dict(color="#334155", size=12),
    )

    return fig


def get_top_repeat_merchant(weekly_data):
    rows = weekly_data.get("repeat_patterns", {}).get("top_merchants", [])

    if not rows:
        return "-", 0, 0

    top = max(rows, key=lambda x: safe_int(x.get("visit_count", x.get("count", 0))))

    return (
        top.get("merchant", "-"),
        safe_int(top.get("visit_count", top.get("count", 0))),
        safe_int(top.get("total_amount", 0)),
    )


def get_peak_weekday(weekly_data):
    pattern = weekly_data.get("weekday_pattern", {})
    peak = pattern.get("peak_weekday")
    rows = pattern.get("weekday_breakdown", [])

    if peak:
        peak_amount = 0
        for row in rows:
            if row.get("weekday") == peak:
                peak_amount = safe_int(row.get("total_amount", 0))
        return peak, peak_amount

    if not rows:
        return "-", 0

    top = max(rows, key=lambda x: safe_int(x.get("total_amount", 0)))
    return top.get("weekday", "-"), safe_int(top.get("total_amount", 0))


def get_top_category(weekly_data):
    rows = weekly_data.get("category_summary", [])

    if not rows:
        return "-", 0

    top = max(rows, key=lambda x: safe_int(x.get("total_amount", 0)))

    return top.get("category", "-"), safe_int(top.get("total_amount", 0))


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


def get_pattern_type(top_visit_count: int, peak_weekday_amount: int, total_amount: int):
    weekday_ratio = (peak_weekday_amount / total_amount * 100) if total_amount else 0

    if top_visit_count >= 2:
        return "반복형", "같은 가맹점에서 반복된 소비가 뚜렷합니다."

    if weekday_ratio >= 60:
        return "집중형", "특정 요일에 소비가 몰린 흐름입니다."

    return "분산형", "여러 항목에 나뉘어 소비가 발생했습니다."


def render_vote_buttons():
    like_active = st.session_state.weekly_report_feedback == "like"
    dislike_active = st.session_state.weekly_report_feedback == "dislike"

    like_key = "weekly_like_btn_active" if like_active else "weekly_like_btn"
    dislike_key = "weekly_dislike_btn_active" if dislike_active else "weekly_dislike_btn"

    like_col, dislike_col = st.columns(2)

    with like_col:
        if st.button("👍 좋아요", use_container_width=True, key=like_key):
            st.session_state.weekly_report_feedback = "like"
            st.rerun()

    with dislike_col:
        if st.button("👎 싫어요", use_container_width=True, key=dislike_key):
            st.session_state.weekly_report_feedback = "dislike"
            st.rerun()


def render_weekly_report(result):
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
    feedback_message = _html_text(feedback.feedback_message)
    next_week_mission = _html_text(feedback.next_week_mission)
    feedback_evidences = feedback.key_evidences
    feedback_action_items = feedback.action_items

    total_amount = safe_int(weekly_summary["this_week_total"])
    prev_rate = float(weekly_summary.get("diff_rate_percent", 0))

    top_merchant, top_visit_count, top_merchant_amount = get_top_repeat_merchant(weekly_data)
    peak_weekday, peak_weekday_amount = get_peak_weekday(weekly_data)
    top_category, top_category_amount = get_top_category(weekly_data)

    action_title, action_detail = get_action_text(feedback)

    pattern_type, pattern_desc = get_pattern_type(
        top_visit_count=top_visit_count,
        peak_weekday_amount=peak_weekday_amount,
        total_amount=total_amount,
    )

    if top_visit_count >= 2 and top_merchant != "-":
        hero_main = (
            f"이번 주 가장 반복된 소비는 <strong>{top_merchant}</strong>이고,<br>"
            f"총 <strong>{top_visit_count}회</strong> 방문했습니다."
        )
        habit_title = f"습관 소비 TOP<br>{top_merchant}"
        habit_body = (
            f"<b>{top_visit_count}회 반복 방문</b><br><br>"
            "이번 주에는 같은 가맹점에서 반복적으로 지출이 발생했습니다.<br>"
            "다음 주에는 이 방문 횟수를 1회만 줄여도 습관 소비를 끊는 시작점이 됩니다."
        )
    else:
        hero_main = (
            f"이번 주 소비는 반복보다 <strong>{peak_weekday}요일</strong>에 몰렸고,<br>"
            f"해당 요일에 <strong>{money(peak_weekday_amount)}</strong>을 사용했습니다."
        )
        habit_title = "반복보다<br>집중 소비"
        habit_body = (
            f"반복 가맹점이 뚜렷하지 않아 <b>{peak_weekday}요일 고액 소비</b>와 "
            f"<b>{top_category}</b> 카테고리를 중심으로 해석했습니다.<br><br>"
            "거래가 대부분 1회라면 반복 분석보다 집중 분석이 더 정확합니다."
        )

    summary_title = getattr(feedback, "summary_title", "이번 주 소비 인사이트")

    if "주간 소비" in summary_title and "피드백" in summary_title:
        summary_title = "LLM 소비 코멘트"

    if top_visit_count and top_merchant_amount:
        expected_saving = round(top_merchant_amount / max(top_visit_count, 1))
    else:
        expected_saving = getattr(feedback, "expected_saving_amount", 0) or round(
            top_category_amount * 0.15
        )

    h1, h2 = st.columns([2.4, 1])

    with h1:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-kicker">🧠 LLM 주간 소비 해석 · {pattern_type}</div>
                <div class="hero-main">
                    {hero_main}
                </div>
                <div class="hero-desc">
                    단순히 많이 쓴 항목보다, 이번 주 소비가 반복된 습관인지
                    특정 요일에 몰린 소비인지 먼저 구분했습니다.
                </div>
                <div class="hero-chip-wrap">
                    <div class="hero-chip">총 소비 · {money(total_amount)}</div>
                    <div class="hero-chip">최대 요일 · {peak_weekday}</div>
                    <div class="hero-chip">핵심 판단 · {pattern_desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
            <div class="side-panel">
                <div class="side-label">이번 주 핵심 신호</div>
                <div class="side-value">{pattern_type}</div>
                <div class="side-desc">
                    가장 큰 소비 카테고리는 <b>{top_category}</b>입니다.<br>
                    해당 카테고리에서 <b>{money(top_category_amount)}</b>을 사용했습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">이번 주 핵심 요약</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_card("이번 주 총 소비", money(total_amount), f"전주 대비 {prev_rate:.2f}%")

    with m2:
        if top_visit_count >= 2 and top_merchant != "-":
            metric_card(
                "반복 소비 TOP", top_merchant, f"{top_visit_count}회 · {money(top_merchant_amount)}"
            )
        else:
            metric_card("소비 집중 요일", f"{peak_weekday}요일", money(peak_weekday_amount))

    with m3:
        metric_card("최대 소비 카테고리", top_category, money(top_category_amount))

    st.markdown('<div class="section">주간 소비 대시보드</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([1.25, 1.25, 1])

    with d1:
        with st.container(border=True):
            st.markdown("### 요일별 소비 흐름")
            st.plotly_chart(make_weekday_chart(weekly_analysis), use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown("### 반복 가맹점 TOP")
            st.plotly_chart(make_repeat_merchant_chart(weekly_analysis), use_container_width=True)

    with d3:
        st.markdown(
            f"""
            <div class="habit-card">
                <div class="habit-title">{habit_title}</div>
                {habit_body}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="danger-card">
                <div class="danger-title">{summary_title}</div>
                {feedback_message}<br><br>
                <b>다음 주 미션</b><br>
                {next_week_mission}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">다음 주 실행 플랜</div>', unsafe_allow_html=True)

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
                다음 주에는 소비 전체를 줄이기보다<br>
                <b>가장 큰 소비 지점 1개</b>부터 조정해보세요.
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
        render_vote_buttons()

    if st.session_state.weekly_report_feedback == "like":
        st.success("좋아요 감사합니다! 다음 리포트도 이 방향으로 개선해볼게요 😊")

    elif st.session_state.weekly_report_feedback == "dislike":
        st.warning("어떤 점이 아쉬웠나요?")

        st.session_state.weekly_feedback_reason = st.text_area(
            "아쉬웠던 점",
            value=st.session_state.weekly_feedback_reason,
            placeholder="예: 피드백이 너무 뻔해요 / 그래프가 이해하기 어려워요 / 행동 규칙이 더 구체적이면 좋겠어요",
            key="weekly_feedback_reason_input",
        )

        if st.button("의견 제출", use_container_width=True, key="weekly_reason_submit_btn"):
            st.success("의견 감사합니다! 다음 리포트 개선에 반영할게요 🙏")

    with st.expander("상세 분석 & 데이터"):
        st.subheader("주간 분석 JSON")
        st.json(weekly_data)

        st.subheader("주간 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("피드백 근거")
        if feedback_evidences:
            st.dataframe(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in feedback_evidences
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("표시할 피드백 근거가 없습니다.")

        st.subheader("다음 주 할 일")
        if feedback_action_items:
            st.dataframe(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in feedback_action_items
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("표시할 행동 항목이 없습니다.")

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))


inject_css()
render_date_picker_styles()

if "weekly_report_generated" not in st.session_state:
    st.session_state.weekly_report_generated = False

if "weekly_result" not in st.session_state:
    st.session_state.weekly_result = None

if "weekly_report_feedback" not in st.session_state:
    st.session_state.weekly_report_feedback = None

if "weekly_feedback_reason" not in st.session_state:
    st.session_state.weekly_feedback_reason = ""

top1, top2 = st.columns([1.3, 1])

with top1:
    st.markdown('<div class="page-title">🔁 이번 주 소비 습관 리포트</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">주간 보고서는 금액보다 “반복된 소비 습관”과 “소비 집중 지점”을 찾는 데 집중합니다.</div>',
        unsafe_allow_html=True,
    )

with top2:
    f1, f2, f3 = st.columns([1, 1.35, 0.9])

    with f1:
        member_id = st.text_input("Member ID", value="1")

    with f2:
        start_date, end_date = select_week_range(
            "분석 주",
            default_start=DEFAULT_CALENDAR_DATE,
            key="weekly_report_week",
        )

    with f3:
        st.write("")
        run = st.button("생성", use_container_width=True)

if run:
    st.session_state.weekly_report_generated = True
    st.session_state.weekly_result = None
    st.session_state.weekly_report_feedback = None
    st.session_state.weekly_feedback_reason = ""

if st.session_state.weekly_report_generated:
    from catcher_llm.config.settings import get_settings
    from catcher_llm.services.consumption_feedback.weekly_feedback import (
        generate_weekly_feedback,
    )

    settings = get_settings()

    if st.session_state.weekly_result is None:
        with st.spinner("이번 주 소비 습관을 분석하고 있어요..."):
            st.session_state.weekly_result = generate_weekly_feedback(
                member_id=int(member_id),
                week_start=start_date,
                week_end=end_date,
                settings=settings,
                chunk_size=800,
                chunk_overlap=120,
                top_k=3,
                max_queries=4,
            )

    render_weekly_report(st.session_state.weekly_result)

else:
    st.info("Member ID와 분석 기간을 선택한 뒤, 생성을 눌러주세요.")
