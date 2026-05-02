from __future__ import annotations

import html
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_week_range,
)

st.set_page_config(
    page_title="이번 주 소비 습관 리포트",
    page_icon="🔁",
    layout="wide",
)

PURPLE_MAIN = "#7c3aed"
PURPLE_LIGHT = "#8b5cf6"
PURPLE_DARK = "#6d28d9"
USER_SCORE_COLUMN = "personal_score"


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def _to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _html_text(value) -> str:
    return html.escape(str(value or "")).replace("\n", "<br>")


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def calc_diff_rate(current: int | float, base: int | float) -> float:
    if not base:
        return 0.0
    return (current - base) / base * 100


def quote_col(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "sqlite" / "app.sqlite3").exists():
            return parent
    return current.parents[1]


PROJECT_ROOT = find_project_root()
APP_SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite" / "app.sqlite3"


# =========================
# DB: weekly action plan / point
# =========================

def ensure_weekly_action_plan_table() -> None:
    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_action_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                action_title TEXT NOT NULL,
                action_detail TEXT,
                target_type TEXT,
                target_name TEXT,
                target_amount INTEGER DEFAULT 0,
                reward_given INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, week_start, week_end)
            )
            """
        )
        conn.commit()


def save_weekly_action_plan(
    member_id: int,
    week_start: date,
    week_end: date,
    action_title: str,
    action_detail: str,
    target_type: str,
    target_name: str,
    target_amount: int,
) -> None:
    ensure_weekly_action_plan_table()

    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO weekly_action_plans (
                user_id, week_start, week_end,
                action_title, action_detail,
                target_type, target_name, target_amount
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, week_start, week_end)
            DO UPDATE SET
                action_title = excluded.action_title,
                action_detail = excluded.action_detail,
                target_type = excluded.target_type,
                target_name = excluded.target_name,
                target_amount = excluded.target_amount
            """,
            (
                member_id,
                week_start.isoformat(),
                week_end.isoformat(),
                action_title,
                action_detail,
                target_type,
                target_name,
                target_amount,
            ),
        )
        conn.commit()


def get_previous_week_action_plan(member_id: int, start_date: date):
    ensure_weekly_action_plan_table()

    prev_start = start_date - timedelta(days=7)

    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM weekly_action_plans
            WHERE user_id = ?
              AND week_start = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            conn,
            params=[member_id, prev_start.isoformat()],
        )

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def get_category_total(member_id: int, week_start: date, week_end: date, category: str) -> int:
    try:
        with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
            df = pd.read_sql_query(
                """
                SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) AS total
                FROM transactions
                WHERE user_id = ?
                  AND category = ?
                  AND date(substr(used_at, 1, 10)) >= date(?)
                  AND date(substr(used_at, 1, 10)) <= date(?)
                """,
                conn,
                params=[member_id, category, week_start.isoformat(), week_end.isoformat()],
            )
        return safe_int(df.iloc[0]["total"])
    except Exception:
        return 0


def get_merchant_total(member_id: int, week_start: date, week_end: date, merchant: str) -> int:
    try:
        with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
            df = pd.read_sql_query(
                """
                SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) AS total
                FROM transactions
                WHERE user_id = ?
                  AND merchant_name = ?
                  AND date(substr(used_at, 1, 10)) >= date(?)
                  AND date(substr(used_at, 1, 10)) <= date(?)
                """,
                conn,
                params=[member_id, merchant, week_start.isoformat(), week_end.isoformat()],
            )
        return safe_int(df.iloc[0]["total"])
    except Exception:
        return 0


def check_previous_plan_success(member_id: int, start_date: date, end_date: date):
    plan = get_previous_week_action_plan(member_id, start_date)

    if not plan:
        return None

    target_type = str(plan.get("target_type", ""))
    target_name = str(plan.get("target_name", ""))

    prev_start = start_date - timedelta(days=7)
    prev_end = end_date - timedelta(days=7)

    if target_type == "가맹점":
        prev_amount = get_merchant_total(member_id, prev_start, prev_end, target_name)
        this_amount = get_merchant_total(member_id, start_date, end_date, target_name)
    else:
        prev_amount = get_category_total(member_id, prev_start, prev_end, target_name)
        this_amount = get_category_total(member_id, start_date, end_date, target_name)

    success = prev_amount > 0 and this_amount < prev_amount

    return {
        "plan": plan,
        "prev_amount": prev_amount,
        "this_amount": this_amount,
        "success": success,
    }


def give_weekly_plan_reward(plan_id: int, member_id: int) -> bool:
    ensure_weekly_action_plan_table()

    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        plan = conn.execute(
            "SELECT reward_given FROM weekly_action_plans WHERE id = ?",
            (plan_id,),
        ).fetchone()

        if not plan or safe_int(plan[0]) == 1:
            return False

        user_cols = pd.read_sql_query("PRAGMA table_info(users)", conn)["name"].tolist()

        if USER_SCORE_COLUMN in user_cols:
            conn.execute(
                f"""
                UPDATE users
                SET {quote_col(USER_SCORE_COLUMN)} =
                    CAST(COALESCE(NULLIF({quote_col(USER_SCORE_COLUMN)}, ''), '0') AS INTEGER) + 50
                WHERE id = ?
                """,
                (member_id,),
            )

        conn.execute(
            """
            UPDATE weekly_action_plans
            SET reward_given = 1
            WHERE id = ?
            """,
            (plan_id,),
        )

        conn.commit()

    return True


# =========================
# weekly comparison
# =========================

def get_week_total_from_sqlite(member_id: int, week_start: date, week_end: date) -> int:
    if not APP_SQLITE_PATH.exists():
        return 0

    try:
        with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
            query = """
            SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) AS total
            FROM transactions
            WHERE user_id = ?
              AND date(substr(used_at, 1, 10)) >= date(?)
              AND date(substr(used_at, 1, 10)) <= date(?)
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=[member_id, week_start.isoformat(), week_end.isoformat()],
            )
            return safe_int(df.iloc[0]["total"])
    except Exception:
        return 0


def get_recent_4_week_average(member_id: int, week_start: date, week_end: date) -> int:
    values = []

    for i in range(1, 5):
        start = week_start - timedelta(days=7 * i)
        end = week_end - timedelta(days=7 * i)
        total = get_week_total_from_sqlite(member_id, start, end)

        if total > 0:
            values.append(total)

    if not values:
        return 0

    return round(sum(values) / len(values))


def get_last_month_same_week_total(member_id: int, week_start: date, week_end: date) -> int:
    return get_week_total_from_sqlite(
        member_id,
        week_start - timedelta(days=28),
        week_end - timedelta(days=28),
    )


# =========================
# CSS
# =========================

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
            font-size: 31px;
            font-weight: 950;
            color: #0f172a;
            letter-spacing: -0.7px;
            margin-bottom: 4px;
        }

        .page-subtitle {
            color:#64748b;
            font-size:14px;
            margin-bottom:16px;
            font-weight:650;
        }

        .hero {
            padding: 30px 34px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,0.26), transparent 28%),
                linear-gradient(135deg, #0f172a 0%, #7c3aed 48%, #2563eb 100%);
            color:white;
            box-shadow: 0 26px 70px rgba(124,58,237,0.22);
            min-height: 230px;
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

        .hero-main strong { color:#fde68a; }

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

        .side-panel-up {
            padding:24px;
            border-radius:30px;
            background:#fff5f5;
            border:1px solid #fecaca;
            box-shadow:0 18px 46px rgba(239,68,68,0.10);
            min-height:230px;
        }

        .side-panel-down {
            padding:24px;
            border-radius:30px;
            background:#eff6ff;
            border:1px solid #bfdbfe;
            box-shadow:0 18px 46px rgba(59,130,246,0.10);
            min-height:230px;
        }

        .side-panel-neutral {
            padding:24px;
            border-radius:30px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 18px 46px rgba(15,23,42,0.07);
            min-height:230px;
        }

        .side-label {
            font-size:14px;
            font-weight:900;
            color:#64748b;
            margin-bottom:12px;
        }

        .side-value {
            font-size:38px;
            font-weight:950;
            line-height:1.15;
            letter-spacing:-0.9px;
        }

        .side-desc {
            margin-top:18px;
            color:#475569;
            font-size:15px;
            line-height:1.75;
            font-weight:750;
        }

        .up-color { color:#dc2626; }
        .down-color { color:#2563eb; }
        .neutral-color { color:#475569; }

        .section {
            font-size:20px;
            font-weight:950;
            color:#0f172a;
            margin:24px 0 12px;
            letter-spacing:-0.4px;
        }

        .metric-card, .compare-card {
            padding:20px 22px;
            border-radius:24px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:122px;
        }

        .metric-label, .compare-label {
            color:#64748b;
            font-weight:850;
            font-size:13px;
        }

        .metric-value, .compare-value {
            color:#0f172a;
            font-weight:950;
            font-size:24px;
            margin-top:10px;
            line-height:1.25;
            letter-spacing:-0.5px;
        }

        .metric-desc, .compare-desc {
            color:#94a3b8;
            font-size:12.5px;
            margin-top:8px;
            line-height:1.45;
            font-weight:650;
        }

        .purple-card {
            padding:22px;
            border-radius:26px;
            background:#f5f3ff;
            border:1px solid #ddd6fe;
            color:#4c1d95;
            min-height:194px;
        }

        .purple-title {
            font-size:22px;
            font-weight:950;
            color:#6d28d9;
            margin-bottom:12px;
            letter-spacing:-0.5px;
        }

        .orange-card {
            padding:22px;
            border-radius:26px;
            background:#fff7ed;
            border:1px solid #fed7aa;
            color:#9a3412;
            min-height:194px;
        }

        .orange-title {
            font-size:22px;
            font-weight:950;
            color:#ea580c;
            margin-bottom:12px;
            letter-spacing:-0.5px;
        }

        .insight-card, .action-card, .vote-card {
            padding:22px;
            border-radius:26px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:142px;
        }

        .insight-title {
            font-size:19px;
            font-weight:950;
            color:#0f172a;
            margin-bottom:12px;
            letter-spacing:-0.4px;
        }

        .insight-body {
            color:#475569;
            font-size:14px;
            line-height:1.7;
            font-weight:650;
        }

        .num {
            display:inline-flex;
            width:36px;
            height:36px;
            border-radius:50%;
            background:#7c3aed;
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
            min-height:142px;
            font-weight:850;
            line-height:1.65;
        }

        .reward-card {
            padding:22px;
            border-radius:26px;
            background:#ecfdf5;
            border:1px solid #86efac;
            color:#166534;
            box-shadow:0 10px 28px rgba(34,197,94,0.12);
            min-height:130px;
            font-weight:850;
            line-height:1.65;
        }

        .fail-card {
            padding:22px;
            border-radius:26px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            color:#475569;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:130px;
            font-weight:750;
            line-height:1.65;
        }

        .point-pop {
            display:inline-block;
            margin-top:14px;
            padding:14px 24px;
            border-radius:999px;
            background:linear-gradient(135deg, #22c55e, #16a34a);
            color:white;
            font-size:28px;
            font-weight:950;
            box-shadow:0 16px 34px rgba(34,197,94,0.35);
            animation: pointPop 1.05s ease-out;
        }

        @keyframes pointPop {
            0% { opacity:0; transform:translateY(18px) scale(0.72); }
            45% { opacity:1; transform:translateY(-8px) scale(1.12); }
            75% { transform:translateY(0px) scale(0.98); }
            100% { opacity:1; transform:translateY(0px) scale(1); }
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


# =========================
# UI helpers
# =========================

def metric_card(label: str, value: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_html_text(label)}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compare_card(label: str, current: int, base: int):
    if base <= 0:
        value = "비교 데이터 없음"
        desc = "해당 기준 주차의 소비 데이터가 없습니다."
        color = "neutral-color"
    else:
        diff = current - base
        rate = calc_diff_rate(current, base)

        if diff > 0:
            value = f"+{money(abs(diff))} 증가"
            desc = f"기준 {money(base)} 대비 +{rate:.1f}%"
            color = "up-color"
        elif diff < 0:
            value = f"-{money(abs(diff))} 감소"
            desc = f"기준 {money(base)} 대비 {rate:.1f}%"
            color = "down-color"
        else:
            value = "변화 없음"
            desc = f"기준 {money(base)}와 동일"
            color = "neutral-color"

    st.markdown(
        f"""
        <div class="compare-card">
            <div class="compare-label">{_html_text(label)}</div>
            <div class="compare-value {color}">{value}</div>
            <div class="compare-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Data extractors
# =========================

def get_weekly_summary(weekly_data: dict) -> dict:
    return weekly_data.get("weekly_summary", {}) or {}


def get_category_rows(weekly_data: dict) -> list[dict]:
    rows = weekly_data.get("category_summary", []) or []
    return [
        {
            "category": row.get("category", row.get("category_name", "-")),
            "amount": safe_int(row.get("total_amount", row.get("amount", 0))),
            "count": safe_int(row.get("transaction_count", row.get("count", 0))),
        }
        for row in rows
    ]


def get_merchant_rows(weekly_data: dict) -> list[dict]:
    repeat_rows = (
        weekly_data.get("repeat_patterns", {}).get("top_merchants", [])
        if isinstance(weekly_data.get("repeat_patterns", {}), dict)
        else []
    )

    merchant_rows = weekly_data.get("merchant_summary", []) or []
    rows = repeat_rows or merchant_rows

    return [
        {
            "merchant": row.get("merchant_name", row.get("merchant", row.get("name", "-"))),
            "amount": safe_int(row.get("total_amount", row.get("amount", 0))),
            "count": safe_int(row.get("visit_count", row.get("transaction_count", row.get("count", 0)))),
        }
        for row in rows
    ]


def get_weekday_rows(weekly_data: dict) -> list[dict]:
    pattern = weekly_data.get("weekday_pattern", {}) or {}
    rows = pattern.get("weekday_breakdown", []) or []

    return [
        {
            "weekday": row.get("weekday", "-"),
            "amount": safe_int(row.get("total_amount", row.get("amount", 0))),
        }
        for row in rows
    ]


def get_top_category(weekly_data: dict):
    rows = get_category_rows(weekly_data)

    if not rows:
        return "-", 0, 0

    top = max(rows, key=lambda x: x["amount"])
    return top["category"], top["amount"], top["count"]


def get_top_merchant(weekly_data: dict):
    rows = get_merchant_rows(weekly_data)

    if not rows:
        return "-", 0, 0

    top = max(rows, key=lambda x: (x["count"], x["amount"]))
    return top["merchant"], top["count"], top["amount"]


def get_peak_weekday(weekly_data: dict):
    pattern = weekly_data.get("weekday_pattern", {}) or {}
    peak = pattern.get("peak_weekday")
    rows = get_weekday_rows(weekly_data)

    if peak:
        for row in rows:
            if row["weekday"] == peak:
                return peak, row["amount"]

    if not rows:
        return "-", 0

    top = max(rows, key=lambda x: x["amount"])
    return top["weekday"], top["amount"]


def get_pattern_type(top_visit_count: int, peak_weekday_amount: int, total_amount: int):
    weekday_ratio = (peak_weekday_amount / total_amount * 100) if total_amount else 0

    if top_visit_count >= 2:
        return "반복형", "같은 가맹점에서 반복된 소비가 뚜렷합니다."

    if weekday_ratio >= 45:
        return "집중형", "특정 요일에 소비가 몰린 흐름입니다."

    return "분산형", "여러 항목에 나뉘어 소비가 발생했습니다."


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
        return _html_text(title), _html_text(detail)

    return (
        "가장 큰 소비 지점 1개만 줄이기",
        "다음 주에는 전체 소비를 줄이기보다 TOP 가맹점 또는 TOP 카테고리 소비를 1회만 줄여보세요.",
    )


def infer_plan_target(
    top_merchant: str,
    top_merchant_amount: int,
    top_category: str,
    top_category_amount: int,
) -> tuple[str, str, int]:
    if top_merchant and top_merchant != "-" and top_merchant_amount > 0:
        return "가맹점", top_merchant, top_merchant_amount

    return "카테고리", top_category, top_category_amount


# =========================
# Charts
# =========================

def make_weekday_chart(weekly_data: dict):
    rows = get_weekday_rows(weekly_data)
    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]

    df = pd.DataFrame(rows)

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
        marker_color=PURPLE_MAIN,
        marker_line_width=0,
        width=0.55,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=330,
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


def make_top_category_chart(weekly_data: dict):
    rows = get_category_rows(weekly_data)
    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame({"category": ["데이터 없음"], "amount": [0], "count": [0]})

    df = df.sort_values("amount", ascending=False).head(5)
    df = df.sort_values("amount", ascending=True)

    fig = px.bar(df, x="amount", y="category", orientation="h", text="amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color=PURPLE_LIGHT,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}원<extra></extra>",
    )

    fig.update_layout(
        height=330,
        margin=dict(t=16, b=8, l=8, r=38),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
        font=dict(color="#334155", size=12),
    )

    return fig


def make_top_merchant_chart(weekly_data: dict):
    rows = get_merchant_rows(weekly_data)
    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame({"merchant": ["데이터 없음"], "amount": [0], "count": [0]})

    df = df.sort_values("amount", ascending=False).head(5)
    df = df.sort_values("amount", ascending=True)

    fig = px.bar(df, x="amount", y="merchant", orientation="h", text="amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color=PURPLE_DARK,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}원<br>%{customdata}회<extra></extra>",
        customdata=df["count"],
    )

    fig.update_layout(
        height=330,
        margin=dict(t=16, b=8, l=8, r=38),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
        font=dict(color="#334155", size=12),
    )

    return fig


# =========================
# Vote
# =========================

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
        if st.button("👎 아쉬워요", use_container_width=True, key=dislike_key):
            st.session_state.weekly_report_feedback = "dislike"
            st.rerun()


# =========================
# Main report
# =========================

def render_weekly_report(result, member_id: int, start_date: date, end_date: date):
    if result.error:
        st.error(f"주간 피드백 생성 실패: {result.error}")
        st.stop()

    if result.feedback is None or result.weekly_analysis is None:
        st.warning("주간 분석 또는 피드백 결과가 비어 있습니다.")
        st.stop()

    feedback = result.feedback
    weekly_data = _to_dict(result.weekly_analysis)
    weekly_summary = get_weekly_summary(weekly_data)

    total_amount = safe_int(weekly_summary.get("this_week_total", 0))
    prev_amount = safe_int(weekly_summary.get("last_week_total", 0))

    sqlite_this_week_total = get_week_total_from_sqlite(member_id, start_date, end_date)
    if total_amount <= 0 and sqlite_this_week_total > 0:
        total_amount = sqlite_this_week_total

    sqlite_prev_week_total = get_week_total_from_sqlite(
        member_id,
        start_date - timedelta(days=7),
        end_date - timedelta(days=7),
    )

    if prev_amount <= 0 and sqlite_prev_week_total > 0:
        prev_amount = sqlite_prev_week_total

    recent_4_week_average = get_recent_4_week_average(member_id, start_date, end_date)
    last_month_same_week_total = get_last_month_same_week_total(member_id, start_date, end_date)

    diff_amount = total_amount - prev_amount
    prev_rate = calc_diff_rate(total_amount, prev_amount)

    top_merchant, top_visit_count, top_merchant_amount = get_top_merchant(weekly_data)
    peak_weekday, peak_weekday_amount = get_peak_weekday(weekly_data)
    top_category, top_category_amount, top_category_count = get_top_category(weekly_data)

    pattern_type, pattern_desc = get_pattern_type(
        top_visit_count=top_visit_count,
        peak_weekday_amount=peak_weekday_amount,
        total_amount=total_amount,
    )

    action_title, action_detail = get_action_text(feedback)
    target_type, target_name, target_amount = infer_plan_target(
        top_merchant=top_merchant,
        top_merchant_amount=top_merchant_amount,
        top_category=top_category,
        top_category_amount=top_category_amount,
    )

    save_weekly_action_plan(
        member_id=member_id,
        week_start=start_date,
        week_end=end_date,
        action_title=action_title,
        action_detail=action_detail,
        target_type=target_type,
        target_name=target_name,
        target_amount=target_amount,
    )

    previous_plan_check = check_previous_plan_success(member_id, start_date, end_date)

    summary_title = _html_text(getattr(feedback, "summary_title", "LLM 소비 코멘트"))
    feedback_message = _html_text(
        getattr(
            feedback,
            "feedback_message",
            getattr(feedback, "scolding_message", "이번 주 소비에서 반복되는 패턴을 줄이는 것이 중요합니다."),
        )
    )
    next_week_mission = _html_text(getattr(feedback, "next_week_mission", ""))

    if top_visit_count >= 2 and top_merchant != "-":
        hero_main = (
            f"이번 주 돈이 샌 지점은<br>"
            f"<strong>{_html_text(top_merchant)}</strong> 반복 소비였어요."
        )
        hero_desc = (
            f"{_html_text(top_merchant)}에서 총 {_html_text(top_visit_count)}회, "
            f"{money(top_merchant_amount)}을 사용했습니다. "
            "주간 리포트는 금액보다 반복된 습관을 먼저 봅니다."
        )
    else:
        hero_main = (
            f"이번 주 소비는<br>"
            f"<strong>{_html_text(peak_weekday)}요일</strong>에 가장 몰렸어요."
        )
        hero_desc = (
            f"{_html_text(peak_weekday)}요일에 {money(peak_weekday_amount)}을 사용했습니다. "
            "반복 가맹점이 뚜렷하지 않을 때는 특정 요일과 카테고리 집중도를 중심으로 해석합니다."
        )

    expected_saving = (
        round(top_merchant_amount / max(top_visit_count, 1))
        if top_visit_count and top_merchant_amount
        else round(top_category_amount * 0.15)
    )

    h1, h2 = st.columns([2.35, 1])

    with h1:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-kicker">🧠 LLM 주간 소비 해석 · {pattern_type}</div>
                <div class="hero-main">{hero_main}</div>
                <div class="hero-desc">{hero_desc}</div>
                <div class="hero-chip-wrap">
                    <div class="hero-chip">총 소비 · {money(total_amount)}</div>
                    <div class="hero-chip">TOP 카테고리 · {_html_text(top_category)}</div>
                    <div class="hero-chip">핵심 판단 · {_html_text(pattern_desc)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        if diff_amount > 0:
            card_class = "side-panel-up"
            text_class = "up-color"
            change_word = "증가"
            icon = "📈"
        elif diff_amount < 0:
            card_class = "side-panel-down"
            text_class = "down-color"
            change_word = "감소"
            icon = "📉"
        else:
            card_class = "side-panel-neutral"
            text_class = "neutral-color"
            change_word = "변화 없음"
            icon = "➖"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="side-label">전주 대비 소비 변화</div>
                <div class="side-value {text_class}">{icon} {change_word}</div>
                <div class="side-desc">
                    지난주 대비
                    <b class="{text_class}">{money(abs(diff_amount))}</b>
                    {change_word}했습니다.<br>
                    변화율은
                    <b class="{text_class}">{abs(prev_rate):.1f}%</b>
                    입니다.<br><br>
                    이번 주는 <b>{_html_text(top_category)}</b> 소비가 가장 컸습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">이번 주 핵심 요약</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("이번 주 총 소비", money(total_amount), f"전주 대비 {prev_rate:.1f}%")

    with m2:
        metric_card("TOP 가맹점", _html_text(top_merchant), f"{top_visit_count}회 · {money(top_merchant_amount)}")

    with m3:
        metric_card("TOP 카테고리", _html_text(top_category), f"{top_category_count}건 · {money(top_category_amount)}")

    with m4:
        metric_card("소비 집중 요일", f"{_html_text(peak_weekday)}요일", money(peak_weekday_amount))

    st.markdown('<div class="section">비교 기준으로 보기</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        compare_card("전주 대비", total_amount, prev_amount)

    with c2:
        compare_card("최근 4주 평균 대비", total_amount, recent_4_week_average)

    with c3:
        compare_card("지난달 같은 주차 대비", total_amount, last_month_same_week_total)

    st.markdown('<div class="section">주간 소비 대시보드</div>', unsafe_allow_html=True)

    d1, d2 = st.columns([1.1, 1])

    with d1:
        with st.container(border=True):
            st.markdown("### 요일별 소비 흐름")
            st.plotly_chart(make_weekday_chart(weekly_data), use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown("### TOP 5 가맹점")
            st.plotly_chart(make_top_merchant_chart(weekly_data), use_container_width=True)

    d3, d4 = st.columns([1.1, 1])

    with d3:
        with st.container(border=True):
            st.markdown("### TOP 5 카테고리")
            st.plotly_chart(make_top_category_chart(weekly_data), use_container_width=True)

    with d4:
        st.markdown(
            f"""
            <div class="purple-card">
                <div class="purple-title">이번 주 소비 패턴</div>
                <b>{pattern_type}</b><br><br>
                {_html_text(peak_weekday)}요일에 가장 많은 소비가 발생했고,<br>
                가장 큰 소비 카테고리는 <b>{_html_text(top_category)}</b>였습니다.<br><br>
                총 <b>{money(top_category_amount)}</b> 사용
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">LLM 소비 해석</div>', unsafe_allow_html=True)

    i1, i2 = st.columns([1.25, 1])

    with i1:
        mission_html = ""
        if next_week_mission:
            mission_html = f"<br><br><b>다음 주 미션</b><br>{next_week_mission}"

        st.markdown(
            f"""
            <div class="orange-card">
                <div class="orange-title">{summary_title}</div>
                {feedback_message}
                {mission_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">왜 이 지점을 봐야 할까?</div>
                <div class="insight-body">
                    주간 리포트의 핵심은 단순 합계가 아니라 <b>반복성</b>입니다.<br><br>
                    같은 가맹점, 같은 카테고리, 특정 요일에 소비가 반복되면
                    사용자는 스스로 인식하지 못한 소비 습관을 만들 가능성이 높습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">지난주 실행 플랜 달성 여부</div>', unsafe_allow_html=True)

    if previous_plan_check is None:
        st.markdown(
            """
            <div class="fail-card">
                <b>아직 비교할 지난주 실행 플랜이 없어요.</b><br><br>
                이번 주 실행 플랜부터 저장되고, 다음 주 리포트에서 달성 여부를 확인할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif previous_plan_check["success"]:
        plan = previous_plan_check["plan"]
        rewarded = give_weekly_plan_reward(safe_int(plan["id"]), member_id)

        st.markdown(
            f"""
            <div class="reward-card">
                🎉 지난주 실행 플랜을 달성했어요!<br><br>
                <b>{_html_text(plan["target_type"])} · {_html_text(plan["target_name"])}</b> 소비가
                지난주 <b>{money(previous_plan_check["prev_amount"])}</b>에서
                이번 주 <b>{money(previous_plan_check["this_amount"])}</b>로 줄었습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if rewarded:
            st.markdown('<div class="point-pop">+50P 🎉</div>', unsafe_allow_html=True)
        else:
            st.info("이미 이 실행 플랜 보상은 지급되었습니다.")

    else:
        plan = previous_plan_check["plan"]

        st.markdown(
            f"""
            <div class="fail-card">
                <b>지난주 실행 플랜은 아직 달성하지 못했어요.</b><br><br>
                지난주 목표: <b>{_html_text(plan["action_title"])}</b><br>
                점검 대상: <b>{_html_text(plan["target_type"])} · {_html_text(plan["target_name"])}</b><br><br>
                지난주 {money(previous_plan_check["prev_amount"])} →
                이번 주 {money(previous_plan_check["this_amount"])} 입니다.
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
                이번 주 실행 플랜은 다음 주 리포트에서 자동 점검됩니다.<br><br>
                점검 대상: <b>{_html_text(target_type)} · {_html_text(target_name)}</b><br>
                현재 소비액: <b>{money(target_amount)}</b><br><br>
                예상 절약액 약 <b>{money(expected_saving)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:
        st.markdown(
            """
            <div class="vote-card">
                <b>이 리포트는 어땠나요?</b><br><br>
                <span style="color:#64748b; font-weight:650;">
                다음 리포트 개선에 반영할게요.
                </span>
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

    with st.expander("상세 분석 데이터 보기"):
        st.subheader("주간 분석 JSON")
        st.json(weekly_data)

        st.subheader("주간 피드백 JSON")
        st.json(feedback.model_dump() if hasattr(feedback, "model_dump") else feedback)

        st.subheader("비교 데이터 디버그")
        st.json(
            {
                "db_path": str(APP_SQLITE_PATH),
                "this_week": [start_date.isoformat(), end_date.isoformat()],
                "this_week_total": total_amount,
                "prev_week_total": prev_amount,
                "recent_4_week_average": recent_4_week_average,
                "last_month_same_week_total": last_month_same_week_total,
                "saved_plan_target": {
                    "target_type": target_type,
                    "target_name": target_name,
                    "target_amount": target_amount,
                },
                "previous_plan_check": previous_plan_check,
            }
        )

        feedback_evidences = getattr(feedback, "key_evidences", []) or []
        feedback_action_items = getattr(feedback, "action_items", []) or []

        st.subheader("피드백 근거")
        if feedback_evidences:
            st.dataframe(
                [item.model_dump() if hasattr(item, "model_dump") else item for item in feedback_evidences],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("표시할 피드백 근거가 없습니다.")

        st.subheader("다음 주 할 일")
        if feedback_action_items:
            st.dataframe(
                [item.model_dump() if hasattr(item, "model_dump") else item for item in feedback_action_items],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("표시할 행동 항목이 없습니다.")

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))


# =========================
# Page
# =========================

inject_css()
render_date_picker_styles()
ensure_weekly_action_plan_table()

if "weekly_report_generated" not in st.session_state:
    st.session_state.weekly_report_generated = False

if "weekly_result" not in st.session_state:
    st.session_state.weekly_result = None

if "weekly_report_feedback" not in st.session_state:
    st.session_state.weekly_report_feedback = None

if "weekly_feedback_reason" not in st.session_state:
    st.session_state.weekly_feedback_reason = ""

if "weekly_report_params" not in st.session_state:
    st.session_state.weekly_report_params = None

top1, top2 = st.columns([1.35, 1])

with top1:
    st.markdown(
        '<div class="page-title">🔁 이번 주 소비 습관 리포트</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">지난주 실행 플랜 달성 여부를 확인하고, 달성 시 50P를 지급합니다.</div>',
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
    st.session_state.weekly_report_params = {
        "member_id": int(member_id),
        "start_date": start_date,
        "end_date": end_date,
    }

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

    params = st.session_state.weekly_report_params or {
        "member_id": int(member_id),
        "start_date": start_date,
        "end_date": end_date,
    }

    render_weekly_report(
        st.session_state.weekly_result,
        member_id=params["member_id"],
        start_date=params["start_date"],
        end_date=params["end_date"],
    )

else:
    st.info("Member ID와 분석 기간을 선택한 뒤, 생성을 눌러주세요.")
