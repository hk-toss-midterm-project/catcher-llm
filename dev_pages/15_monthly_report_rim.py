from __future__ import annotations

import html
import inspect
import re
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)

st.set_page_config(page_title="월간 소비 리포트", page_icon="🏆", layout="wide")

USER_SCORE_COLUMN = "personal_score"


def money(value: int | float) -> str:
    return f"{value:,.0f}원"


def to_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def _html_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]*>", "", text)
    return html.escape(text).replace("\n", "<br>")


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


def shift_month(month: str, offset: int) -> str:
    year, month_num = map(int, month.split("-"))
    month_num += offset

    while month_num <= 0:
        year -= 1
        month_num += 12

    while month_num > 12:
        year += 1
        month_num -= 12

    return f"{year}-{month_num:02d}"


def month_range(month: str) -> tuple[str, str]:
    year, month_num = map(int, month.split("-"))
    start_date = f"{year}-{month_num:02d}-01"

    if month_num == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month_num + 1:02d}-01"

    return start_date, end_date


def get_month_total_from_sqlite(member_id: int, month: str) -> int:
    if not APP_SQLITE_PATH.exists():
        return 0

    start_date, end_date = month_range(month)

    try:
        with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
            query = """
            SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) AS total
            FROM transactions
            WHERE user_id = ?
              AND date(substr(used_at, 1, 10)) >= date(?)
              AND date(substr(used_at, 1, 10)) < date(?)
            """
            df = pd.read_sql_query(query, conn, params=[member_id, start_date, end_date])
            return safe_int(df.iloc[0]["total"])
    except Exception:
        return 0


def get_recent_3_month_average(member_id: int, month: str) -> int:
    values = []

    for i in range(1, 4):
        target_month = shift_month(month, -i)
        total = get_month_total_from_sqlite(member_id, target_month)
        if total > 0:
            values.append(total)

    if not values:
        return 0

    return round(sum(values) / len(values))


def calc_saving_rate_and_point(recent_3_month_average: int, total_amount: int) -> tuple[float, int]:
    if recent_3_month_average <= 0:
        return 0.0, 0

    saving_rate = (recent_3_month_average - total_amount) / recent_3_month_average * 100

    if saving_rate <= 0:
        return saving_rate, 0

    point = int(saving_rate * 20)
    point = min(point, 1000)

    return saving_rate, point


def ensure_monthly_saving_reward_table() -> None:
    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_saving_rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                month TEXT NOT NULL,
                recent_3_month_average INTEGER NOT NULL,
                this_month_total INTEGER NOT NULL,
                saving_rate REAL NOT NULL,
                reward_point INTEGER NOT NULL,
                reward_given INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, month)
            )
            """
        )
        conn.commit()


def give_monthly_saving_reward(
    member_id: int,
    month: str,
    recent_3_month_average: int,
    total_amount: int,
    saving_rate: float,
    reward_point: int,
) -> bool:
    if reward_point <= 0:
        return False

    ensure_monthly_saving_reward_table()

    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        already = conn.execute(
            """
            SELECT id
            FROM monthly_saving_rewards
            WHERE user_id = ?
              AND month = ?
            """,
            (member_id, month),
        ).fetchone()

        if already:
            return False

        user_cols = pd.read_sql_query("PRAGMA table_info(users)", conn)["name"].tolist()

        if USER_SCORE_COLUMN in user_cols:
            conn.execute(
                f"""
                UPDATE users
                SET {quote_col(USER_SCORE_COLUMN)} =
                    CAST(COALESCE(NULLIF({quote_col(USER_SCORE_COLUMN)}, ''), '0') AS INTEGER) + ?
                WHERE id = ?
                """,
                (reward_point, member_id),
            )

        conn.execute(
            """
            INSERT INTO monthly_saving_rewards (
                user_id, month, recent_3_month_average,
                this_month_total, saving_rate, reward_point, reward_given
            )
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                member_id,
                month,
                recent_3_month_average,
                total_amount,
                saving_rate,
                reward_point,
            ),
        )

        conn.commit()

    return True


def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background:#f8fafc; }

        .block-container {
            max-width:1500px;
            padding-top:1.3rem;
            padding-bottom:2rem;
        }

        .page-title {
            font-size:30px;
            font-weight:950;
            color:#0f172a;
            letter-spacing:-0.7px;
            margin-bottom:4px;
        }

        .page-subtitle {
            color:#64748b;
            font-size:14px;
            margin-bottom:16px;
            font-weight:650;
        }

        .hero {
            padding:28px 32px;
            border-radius:30px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,0.24), transparent 28%),
                linear-gradient(135deg, #0f172a 0%, #059669 48%, #2563eb 100%);
            color:white;
            box-shadow:0 26px 70px rgba(5,150,105,0.22);
            min-height:218px;
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
            min-height:218px;
        }

        .side-panel-down {
            padding:24px;
            border-radius:30px;
            background:#eff6ff;
            border:1px solid #bfdbfe;
            box-shadow:0 18px 46px rgba(59,130,246,0.10);
            min-height:218px;
        }

        .side-panel-neutral {
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
            font-size:34px;
            font-weight:950;
            line-height:1.2;
            letter-spacing:-0.7px;
        }

        .side-desc {
            margin-top:14px;
            color:#475569;
            font-size:14px;
            line-height:1.65;
            font-weight:700;
        }

        .up-color { color:#dc2626; }
        .down-color { color:#2563eb; }
        .neutral-color { color:#475569; }

        .section {
            font-size:19px;
            font-weight:950;
            color:#0f172a;
            margin:22px 0 12px;
            letter-spacing:-0.4px;
        }

        .metric-card, .compare-card {
            padding:20px 22px;
            border-radius:24px;
            background:#ffffff;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 28px rgba(15,23,42,0.055);
            min-height:120px;
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

        .saving-point-card {
            padding:24px;
            border-radius:28px;
            background:#ecfdf5;
            border:1px solid #86efac;
            color:#166534;
            box-shadow:0 16px 38px rgba(34,197,94,0.12);
        }

        .saving-point-title {
            font-size:22px;
            font-weight:950;
            color:#047857;
            margin-bottom:12px;
        }

        .saving-point-value {
            font-size:36px;
            font-weight:950;
            color:#16a34a;
            letter-spacing:-0.8px;
            margin:8px 0;
        }

        .saving-point-desc {
            font-size:14px;
            line-height:1.65;
            font-weight:750;
            color:#166534;
        }

        .point-wrap {
            margin-top:16px;
            text-align:center;
        }

        .point-pop {
            display:inline-block;
            padding:16px 30px;
            border-radius:999px;
            background:linear-gradient(135deg, #22c55e, #16a34a);
            color:white;
            font-size:34px;
            font-weight:950;
            box-shadow:0 18px 42px rgba(34,197,94,0.35);
            animation:pointPop 1.15s ease-out;
        }

        .point-sub {
            margin-top:10px;
            color:#166534;
            font-weight:850;
            animation:fadeIn 1.25s ease-out;
        }

        @keyframes pointPop {
            0% { opacity:0; transform:translateY(24px) scale(0.55) rotate(-8deg); }
            45% { opacity:1; transform:translateY(-10px) scale(1.18) rotate(4deg); }
            72% { transform:translateY(3px) scale(0.96) rotate(-2deg); }
            100% { opacity:1; transform:translateY(0) scale(1) rotate(0deg); }
        }

        @keyframes fadeIn {
            0% { opacity:0; transform:translateY(8px); }
            100% { opacity:1; transform:translateY(0); }
        }

        .score-card {
            padding:22px;
            border-radius:26px;
            background:#ecfdf5;
            border:1px solid #bbf7d0;
            color:#065f46;
            min-height:194px;
        }

        .score-title {
            font-size:22px;
            font-weight:950;
            margin-bottom:12px;
            color:#047857;
            letter-spacing:-0.5px;
        }

        .strategy-card {
            padding:22px;
            border-radius:26px;
            background:#fff7ed;
            border:1px solid #fed7aa;
            color:#9a3412;
            min-height:194px;
        }

        .strategy-title {
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
            background:#10b981;
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

        div[data-testid="stTextInput"] input { border-radius:14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        desc = "해당 기준 월의 소비 데이터가 없습니다."
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


def render_saving_point_card(
    member_id: int,
    month: str,
    total_amount: int,
    recent_3_month_average: int,
    saving_rate: float,
    reward_point: int,
):
    rewarded = give_monthly_saving_reward(
        member_id=member_id,
        month=month,
        recent_3_month_average=recent_3_month_average,
        total_amount=total_amount,
        saving_rate=saving_rate,
        reward_point=reward_point,
    )

    st.markdown('<div class="section">월간 절약 포인트</div>', unsafe_allow_html=True)

    if recent_3_month_average <= 0:
        st.info("최근 3개월 평균 데이터가 부족해서 절약 포인트를 계산할 수 없습니다.")
        return

    if reward_point <= 0:
        st.markdown(
            f"""
            <div class="saving-point-card" style="background:#ffffff; border:1px solid #e5e7eb; color:#475569;">
                <div class="saving-point-title" style="color:#475569;">이번 달 절약 포인트 없음</div>
                <div class="saving-point-desc">
                    최근 3개월 평균은 <b>{money(recent_3_month_average)}</b>이고,
                    이번 달 실제 지출은 <b>{money(total_amount)}</b>입니다.<br>
                    절약률은 <b>{saving_rate:.1f}%</b>로, 포인트 적립 기준에 도달하지 못했습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="saving-point-card">
            <div class="saving-point-title">이번 달 절약 보상</div>
            <div class="saving-point-desc">
                최근 3개월 평균 <b>{money(recent_3_month_average)}</b> 대비
                이번 달 지출은 <b>{money(total_amount)}</b>입니다.
            </div>
            <div class="saving-point-value">{saving_rate:.1f}% 절약 · {reward_point:,}P</div>
            <div class="saving-point-desc">
                포인트 산식: 절약률 × 20, 최대 1,000P까지 적립됩니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rewarded:
        st.markdown(
            f"""
            <div class="point-wrap">
                <div class="point-pop">+{reward_point:,}P 🎉</div>
                <div class="point-sub">월간 절약 보상이 적립되었습니다</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("이번 달 절약 포인트는 이미 지급되었습니다.")


def get_category_rows(monthly_data):
    return (
        monthly_data.get("category_deep", [])
        or monthly_data.get("category_summary", [])
        or monthly_data.get("category_changes", [])
        or []
    )


def get_top_category(monthly_data):
    rows = get_category_rows(monthly_data)
    if not rows:
        return "-", 0
    top = max(rows, key=lambda x: safe_int(x.get("total_amount", x.get("amount", 0))))
    return top.get("category", "-"), safe_int(top.get("total_amount", top.get("amount", 0)))


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
            improved.append({"category": row.get("category", "-"), "diff_amount": diff_amount})

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
        or monthly_data.get("merchant_summary", [])
        or monthly_data.get("top_merchants", [])
    )

    if not rows:
        return "-", 0, 0

    top = max(
        rows,
        key=lambda x: safe_int(x.get("visit_count", x.get("count", x.get("transaction_count", 0)))),
    )

    return (
        top.get("merchant", top.get("merchant_name", top.get("name", "-"))),
        safe_int(top.get("visit_count", top.get("count", top.get("transaction_count", 0)))),
        safe_int(top.get("total_amount", top.get("amount", 0))),
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

    fig = px.bar(df, x="diff_amount", y="category", orientation="h", text="diff_amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_color="#22c55e",
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


def get_top5_merchants_from_sqlite(member_id: int, month: str) -> pd.DataFrame:
    start_date, end_date = month_range(month)

    with sqlite3.connect(str(APP_SQLITE_PATH)) as conn:
        query = """
        SELECT
            merchant_name AS merchant_name,
            COUNT(*) AS payment_count,
            SUM(CAST(amount AS REAL)) AS total_amount
        FROM transactions
        WHERE user_id = ?
          AND merchant_name IS NOT NULL
          AND merchant_name != ''
          AND date(substr(used_at, 1, 10)) >= date(?)
          AND date(substr(used_at, 1, 10)) < date(?)
        GROUP BY merchant_name
        HAVING total_amount IS NOT NULL
        ORDER BY total_amount DESC
        LIMIT 5
        """
        return pd.read_sql_query(query, conn, params=[member_id, start_date, end_date])


def make_top5_merchant_chart_from_sqlite(member_id: int, month: str):
    df = get_top5_merchants_from_sqlite(member_id, month)

    if df.empty:
        return None

    df["total_amount"] = df["total_amount"].fillna(0).astype(float)
    df["payment_count"] = df["payment_count"].fillna(0).astype(int)
    df = df.sort_values("total_amount", ascending=True)

    fig = px.bar(df, x="total_amount", y="merchant_name", orientation="h", text="total_amount")

    fig.update_traces(
        texttemplate="%{text:,.0f}원",
        textposition="outside",
        marker_color="#16a34a",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}원<br>%{customdata}회<extra></extra>",
        customdata=df["payment_count"],
    )

    fig.update_layout(
        height=340,
        margin=dict(t=24, b=8, l=8, r=40),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False, tickformat=","),
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
        return _html_text(title), _html_text(detail)

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
        if st.button("👎 아쉬워요", use_container_width=True, type=dislike_type):
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

    feedback_message = _html_text(getattr(feedback, "feedback_message", ""))
    next_month_mission = _html_text(getattr(feedback, "next_month_mission", ""))

    feedback_evidences = getattr(feedback, "key_evidences", []) or []
    feedback_action_items = getattr(feedback, "action_items", []) or []

    member_id_int = int(member_id)

    total_amount = safe_int(monthly_summary["this_month_total"])
    prev_amount = safe_int(monthly_summary.get("prev_month_total", 0))
    transaction_count = safe_int(monthly_summary.get("transaction_count", 0))

    sqlite_this_month_total = get_month_total_from_sqlite(member_id_int, month)
    if total_amount <= 0 and sqlite_this_month_total > 0:
        total_amount = sqlite_this_month_total

    sqlite_prev_month_total = get_month_total_from_sqlite(member_id_int, shift_month(month, -1))
    if prev_amount <= 0 and sqlite_prev_month_total > 0:
        prev_amount = sqlite_prev_month_total

    recent_3_month_average = get_recent_3_month_average(member_id_int, month)
    saving_rate, reward_point = calc_saving_rate_and_point(recent_3_month_average, total_amount)

    saved_amount = prev_amount - total_amount
    diff_amount = total_amount - prev_amount
    diff_rate = calc_diff_rate(total_amount, prev_amount)

    top_category, top_category_amount = get_top_category(monthly_data)
    improved = get_improved_category(monthly_data)
    worst_category, worst_amount = get_worst_category(monthly_data)
    repeat_merchant, repeat_count, repeat_amount = get_repeat_target(monthly_data)

    action_title, action_detail = get_action_text(feedback)

    if diff_amount > 0:
        status_text = "전월보다 소비가 늘어난 증가형 흐름"
        hero_result = "증가형"
        side_class = "side-panel-up"
        side_color = "up-color"
        side_icon = "📈"
        side_word = "증가"
    elif diff_amount < 0:
        status_text = "전월보다 소비가 줄어든 절약형 흐름"
        hero_result = "절약형"
        side_class = "side-panel-down"
        side_color = "down-color"
        side_icon = "📉"
        side_word = "감소"
    else:
        status_text = "전월과 비슷한 유지형 흐름"
        hero_result = "유지형"
        side_class = "side-panel-neutral"
        side_color = "neutral-color"
        side_icon = "➖"
        side_word = "변화 없음"

    if repeat_count > 0 and repeat_merchant != "-":
        repeat_text = f"반복 가맹점: {_html_text(repeat_merchant)} · {repeat_count}회"
    else:
        repeat_text = f"우선 점검 카테고리: {_html_text(worst_category)}"

    expected_saving = getattr(feedback, "expected_saving_amount", 0)

    if not expected_saving:
        if repeat_count and repeat_amount:
            expected_saving = round(repeat_amount / max(repeat_count, 1) * 4)
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
                    최근 3개월 평균 대비 절약률을 계산해 월간 절약 포인트를 산정했습니다.
                    절약률이 높을수록 더 많은 포인트가 적립됩니다.
                </div>
                <div class="hero-chip-wrap">
                    <div class="hero-chip">최다 소비 · {_html_text(top_category)}</div>
                    <div class="hero-chip">절약률 · {saving_rate:.1f}%</div>
                    <div class="hero-chip">예상 포인트 · {reward_point:,}P</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
            <div class="{side_class}">
                <div class="side-label">전월 대비 소비 변화</div>
                <div class="side-value {side_color}">{side_icon} {side_word}</div>
                <div class="side-desc">
                    전월 대비
                    <b class="{side_color}">{money(abs(diff_amount))}</b>
                    {side_word}했습니다.<br>
                    변화율은
                    <b class="{side_color}">{abs(diff_rate):.1f}%</b>
                    입니다.<br><br>
                    이번 달은 <b>{_html_text(top_category)}</b> 소비가 가장 컸습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">월간 핵심 성과</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_card("총 소비", money(total_amount), f"전월 대비 {diff_rate:.1f}%")

    with m2:
        metric_card("최근 3개월 평균", money(recent_3_month_average), f"절약률 {saving_rate:.1f}%")

    with m3:
        metric_card("이번 달 적립 포인트", f"{reward_point:,}P", "절약률 × 20, 최대 1,000P")

    st.markdown('<div class="section">비교 기준으로 보기</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        compare_card("전월 대비", total_amount, prev_amount)

    with c2:
        compare_card("최근 3개월 평균 대비", total_amount, recent_3_month_average)

    render_saving_point_card(
        member_id=member_id_int,
        month=month,
        total_amount=total_amount,
        recent_3_month_average=recent_3_month_average,
        saving_rate=saving_rate,
        reward_point=reward_point,
    )

    st.markdown('<div class="section">월간 소비 대시보드</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([1.15, 1.15, 1.15])

    with d1:
        with st.container(border=True):
            st.markdown("### 주차별 소비 흐름")
            st.plotly_chart(make_weekly_trend_chart(monthly_analysis), use_container_width=True)

    with d2:
        with st.container(border=True):
            st.markdown("### 전월 대비 카테고리 증감")
            st.plotly_chart(make_category_change_chart(monthly_data), use_container_width=True)

    with d3:
        with st.container(border=True):
            st.markdown("### TOP 5 가맹점")

            top5_merchant_fig = make_top5_merchant_chart_from_sqlite(member_id_int, month)

            if top5_merchant_fig is not None:
                st.plotly_chart(top5_merchant_fig, use_container_width=True)
            else:
                st.info("가맹점 데이터가 없습니다.")

    st.markdown('<div class="section">이번 달 판단</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([1, 1])

    with p1:
        if improved:
            improved_category, improved_amount = improved
            score_title = f"좋아진 소비<br>{_html_text(improved_category)}"
            score_body = (
                f"{_html_text(improved_category)} 지출이 전월보다 "
                f"<b>{money(improved_amount)}</b> 줄었습니다."
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

    with p2:
        summary_title = getattr(feedback, "summary_title", f"다음 달 줄일 1순위: {worst_category}")

        if "월간 소비 피드백" in str(summary_title):
            summary_title = "LLM 소비 코멘트"

        mission_html = ""
        if next_month_mission:
            mission_html = f"<br><br><b>다음 달 미션</b><br>{next_month_mission}"

        st.markdown(
            f"""
            <div class="strategy-card">
                <div class="strategy-title">{_html_text(summary_title)}</div>
                {feedback_message}
                {mission_html}<br><br>
                {repeat_text}
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
                <b>{_html_text(worst_category)}</b>부터 조정하는 전략이 좋습니다.
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
                <b>이 리포트는 어땠나요?</b><br><br>
                <span style="color:#64748b; font-weight:650;">
                다음 리포트 개선에 반영할게요.
                </span>
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

        st.subheader("절약 포인트 디버그")
        st.json(
            {
                "db_path": str(APP_SQLITE_PATH),
                "month": month,
                "this_month_total": total_amount,
                "recent_3_month_average": recent_3_month_average,
                "saving_rate": saving_rate,
                "reward_point": reward_point,
                "formula": "(최근 3개월 평균 - 이달 실제 지출) / 최근 3개월 평균 × 100, 포인트 = 절약률 × 20, 최대 1000P",
            }
        )

        st.subheader("피드백 근거")
        if feedback_evidences:
            st.dataframe(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in feedback_evidences
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("표시할 피드백 근거가 없습니다.")

        st.subheader("다음 달 할 일")
        if feedback_action_items:
            st.dataframe(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in feedback_action_items
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("표시할 행동 항목이 없습니다.")

        st.subheader("RAG 검색 질의")
        st.write(getattr(result, "retrieval_queries", []))

        st.subheader("사용자 반응 로그")
        st.json(st.session_state.get("monthly_report_vote_log", {}))


inject_css()
render_date_picker_styles()
ensure_monthly_saving_reward_table()

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
        '<div class="page-subtitle">최근 3개월 평균 대비 절약률을 계산해 월간 절약 포인트를 적립합니다.</div>',
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
