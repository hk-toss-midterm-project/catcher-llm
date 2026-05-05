"""2026 희망사다리 복지 문서 RAG RAGAS 평가 페이지.

2026_hope_ladder_selected.pdf 를 welfare RAG 로 검색하고
transactions_v5.csv 소비 패턴에서 도출한 질문으로 RAGAS 4대 메트릭을 평가한다.

사용 방법:
  uv run streamlit run dev_app.py -> 사이드바에서 'RAGAS 평가' 선택
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from catcher_llm.config.settings import Settings, configure_langsmith_env, get_settings
from catcher_llm.llm.models import get_chat_model, get_embeddings_model
from catcher_llm.services.rag.welfare import generate_welfare_rag_reply

# =============================================================================
_V5_CSV_REL = Path("data") / "raw" / "csv" / "transactions_v5.csv"
_SESSION_KEY = "ragas_result_v5"

_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
_METRIC_KO = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
_METRIC_DESC = {
    "faithfulness": "답변이 검색 문맥에만 근거하는가",
    "answer_relevancy": "답변이 질문과 직접 관련 있는가",
    "context_precision": "검색 문맥이 정답 도출에 유용한가",
    "context_recall": "정답 근거 정보가 빠짐없이 검색됐는가",
}

# -- 사전 정의 Q&A : 희망사다리 문서 기반 (reference 포함) --------------------
_BASELINE_QA: list[tuple[str, str]] = [
    (
        "여성청소년 생리용품 지원의 월 지원금은 얼마인가?",
        "여성청소년 생리용품 지원의 월 지원금은 1만 4,000원이다.",
    ),
    (
        "임신 사전건강관리 지원사업에서 여성에게 지원하는 최대 금액은 얼마인가?",
        "임신 사전건강관리 지원사업에서 여성에게 지원하는 최대 금액은 13만 원이다.",
    ),
    (
        "저소득 청소년부모 아동양육비 지원의 월 지원금은 얼마인가?",
        "저소득 청소년부모 아동양육비 지원의 월 지원금은 25만 원이다.",
    ),
    (
        "3~5세 유치원 학비 중 국공립유치원 교육비는 월 얼마인가?",
        "3~5세 유치원 학비 중 국공립유치원 교육비는 월 10만 원이다.",
    ),
]

# -- 소비 패턴 기반 Q&A : v5 과소비 스토리에서 도출 (reference 자동생성) ------
_PATTERN_QA: dict[str, list[str]] = {
    "의료": [
        "병원비나 치과 치료비 등 의료비가 부담될 때 이용할 수 있는 복지 지원 제도는?",
        "건강검진이나 피부과 진료 비용을 지원받을 수 있는 정부 복지 프로그램이 있나요?",
        "의료비 과부담 가구를 위한 복지 혜택이나 지원금은 어떤 것들이 있나요?",
    ],
    "쇼핑": [
        "청소년이나 저소득 가구가 생활용품 구매 시 받을 수 있는 바우처나 지원금은?",
        "소비 지출이 많아 가계 부담이 클 때 활용할 수 있는 복지 정책은?",
    ],
    "교육": [
        "자녀 교육비 부담을 줄여주는 복지 지원 프로그램에는 어떤 것이 있나요?",
        "유아 및 초중등 학생의 교육비를 지원하는 제도와 지원 금액은?",
    ],
    "식비": [
        "식비 부담이 높은 가구를 위한 식품 급식 지원 복지 제도가 있나요?",
    ],
    "여가": [
        "청소년 문화 여가 활동을 지원하는 복지 프로그램과 지원 금액은?",
    ],
}


# =============================================================================
@dataclass
class QAPair:
    question: str
    reference: str = ""
    tag: str = "baseline"  # "baseline" | category name


@dataclass
class RAGRecord:
    question: str
    answer: str
    contexts: list[str]
    reference: str
    tag: str
    error: str | None = None


@dataclass
class EvalResult:
    records: list[RAGRecord] = field(default_factory=list)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, float] = field(default_factory=dict)
    ran_at: str = ""


# =============================================================================
def _top_cats(settings: Settings, user_id: int = 1) -> list[str]:
    try:
        from catcher_llm.config.settings import PROJECT_ROOT

        path = PROJECT_ROOT / _V5_CSV_REL
        df = pd.read_csv(str(path), encoding="utf-8-sig")
        df["transaction_time"] = pd.to_datetime(df["transaction_time"])
        mask = (df["user_id"] == user_id) & (df["transaction_time"].dt.month == 5)
        return list(df[mask].groupby("category")["amount"].sum().nlargest(5).index)
    except Exception:
        return list(_PATTERN_QA.keys())


def _build_pairs(
    top_categories: list[str],
    with_baseline: bool,
    cat_filter: list[str],
) -> list[QAPair]:
    pairs: list[QAPair] = []
    if with_baseline:
        for q, ref in _BASELINE_QA:
            pairs.append(QAPair(question=q, reference=ref, tag="baseline"))
    for cat in top_categories:
        if cat not in cat_filter:
            continue
        for q in _PATTERN_QA.get(cat, []):
            pairs.append(QAPair(question=q, reference="", tag=cat))
    return pairs


def _extract_answer(rag: Any) -> str:
    return str(getattr(rag, "answer", getattr(rag, "content", rag)))


def _extract_contexts(rag: Any) -> list[str]:
    if hasattr(rag, "contexts"):
        return [getattr(d, "content", getattr(d, "page_content", str(d))) for d in rag.contexts]
    if hasattr(rag, "source_documents"):
        return [getattr(d, "page_content", str(d)) for d in rag.source_documents]
    return []


def _run_eval(pairs: list[QAPair], settings: Settings) -> EvalResult:
    records: list[RAGRecord] = []
    prog = st.progress(0, text="RAG 실행 중...")
    n = len(pairs)

    for i, p in enumerate(pairs):
        prog.progress(int(i / n * 60), text=f"[{i + 1}/{n}] {p.question[:40]}...")
        try:
            rag = generate_welfare_rag_reply(p.question, settings=settings)
            answer = _extract_answer(rag)
            contexts = _extract_contexts(rag)
            err = str(getattr(rag, "error", "") or "") or None
        except Exception as exc:
            answer, contexts, err = "", [], str(exc)[:200]

        records.append(
            RAGRecord(
                question=p.question,
                answer=answer,
                contexts=contexts,
                reference=p.reference if p.reference else answer,
                tag=p.tag,
                error=err,
            )
        )

    prog.progress(65, text="RAGAS 채점 중...")

    valid = [r for r in records if r.answer and r.contexts]
    if not valid:
        prog.progress(100, text="완료")
        return EvalResult(records=records, ran_at=_now())

    dataset = Dataset.from_dict(
        {
            "user_input": [r.question for r in valid],
            "response": [r.answer for r in valid],
            "retrieved_contexts": [r.contexts for r in valid],
            "reference": [r.reference for r in valid],
        }
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=3),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=get_chat_model(settings),
        embeddings=get_embeddings_model(settings),
    )
    df = result.to_pandas()
    df["tag"] = [r.tag for r in valid]
    summary = {m: float(df[m].mean()) for m in _METRICS if m in df.columns}

    prog.progress(100, text="완료!")
    return EvalResult(records=records, df=df, summary=summary, ran_at=_now())


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 시각화
# =============================================================================
def _show_summary_cards(summary: dict[str, float]) -> None:
    cols = st.columns(4)
    for i, m in enumerate(_METRICS):
        score = summary.get(m)
        if score is None:
            continue
        color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        cols[i].metric(
            label=f"{color} {_METRIC_KO[m]}",
            value=f"{score:.3f}",
            help=_METRIC_DESC[m],
        )


def _show_bar(summary: dict[str, float]) -> None:
    labels = [_METRIC_KO[m] for m in _METRICS if m in summary]
    values = [summary[m] for m in _METRICS if m in summary]
    colors = ["#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.4 else "#e74c3c" for v in values]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            width=0.5,
        )
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.15], title="점수", tickformat=".2f"),
        xaxis_title="메트릭",
        height=360,
        margin=dict(t=30, b=20, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, width="stretch")


def _show_heatmap(df: pd.DataFrame) -> None:
    cols = [m for m in _METRICS if m in df.columns]
    if not cols:
        return
    q_labels = [textwrap.shorten(q, 35, placeholder="...") for q in df["user_input"]]
    y_labels = [_METRIC_KO[c] for c in cols]
    z = [[float(df.iloc[ri][c]) for ri in range(len(df))] for c in cols]
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=q_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="점수", thickness=14),
        )
    )
    fig.update_layout(
        height=max(260, 58 * len(cols) + 120),
        xaxis_tickangle=-35,
        margin=dict(t=10, b=140, l=170, r=10),
    )
    st.plotly_chart(fig, width="stretch")


def _show_per_question(df: pd.DataFrame, records: list[RAGRecord]) -> None:
    cols = [m for m in _METRICS if m in df.columns]
    rec_map = {r.question: r for r in records}

    for i, row in df.iterrows():
        q = row["user_input"]
        rec = rec_map.get(q)
        scores = "  ".join(f"{_METRIC_KO[c]}={row[c]:.2f}" for c in cols)
        tag_icon = "🔵" if row.get("tag") == "baseline" else "🟢"
        label = f"{tag_icon} Q{int(i) + 1} | {scores}"
        with st.expander(label, expanded=False):
            st.markdown(f"**질문:** {q}")
            if rec:
                st.markdown("**RAG 답변**")
                st.info(rec.answer or "*(없음)*")
                if rec.reference and rec.reference != rec.answer:
                    st.markdown("**정답 (reference)**")
                    st.success(rec.reference)
                if rec.contexts:
                    st.markdown(f"**검색 문맥 ({len(rec.contexts)}개)**")
                    for j, ctx in enumerate(rec.contexts):
                        with st.expander(f"문맥 {j + 1}", expanded=False):
                            st.text(ctx[:600] + ("..." if len(ctx) > 600 else ""))
                if rec.error:
                    st.error(f"RAG 오류: {rec.error}")


# =============================================================================
# 페이지
# =============================================================================
settings = get_settings()
configure_langsmith_env(settings)

st.title("RAGAS 평가 — 희망사다리 RAG")
st.caption(
    "**2026_hope_ladder_selected.pdf** 기반 welfare RAG를 "
    "**transactions_v5.csv** 소비 패턴 질문으로 평가합니다."
)

# -- 사이드바 ------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    if settings.has_langsmith_key:
        st.success("LangSmith 연동 활성화")
    else:
        st.warning("LangSmith API Key 없음")
    st.markdown("---")
    st.caption(
        "질문 입력란에 평가할 질문을 한 줄에 하나씩 입력하세요.\n\n"
        "희망사다리 문서의 정답이 알려진 질문은 자동으로 reference가 매칭되고,\n"
        "그 외 질문은 RAG 답변을 reference로 사용합니다."
    )

# -- 예시 질문 레퍼런스 (접기) ------------------------------------------------
_ALL_EXAMPLES = [q for q, _ in _BASELINE_QA] + [q for qs in _PATTERN_QA.values() for q in qs]
_EXAMPLE_TEXT = "\n".join(_ALL_EXAMPLES)

with st.expander("📋 예시 질문 보기 (복사해서 사용)", expanded=False):
    st.code(_EXAMPLE_TEXT, language=None)

# -- 자유 입력 -----------------------------------------------------------------
_PLACEHOLDER = (
    "여성청소년 생리용품 지원의 월 지원금은 얼마인가?\n"
    "병원비나 치과 치료비 등 의료비가 부담될 때 이용할 수 있는 복지 지원 제도는?\n"
    "자녀 교육비 부담을 줄여주는 복지 지원 프로그램에는 어떤 것이 있나요?"
)

raw_input = st.text_area(
    "평가할 질문 입력 (한 줄에 하나씩)",
    placeholder=_PLACEHOLDER,
    height=180,
    key="question_input",
)

# 입력 파싱 → QAPair 변환
_KNOWN_REF: dict[str, str] = {q: ref for q, ref in _BASELINE_QA}

final_pairs: list[QAPair] = []
for line in raw_input.splitlines():
    q = line.strip()
    if not q:
        continue
    ref = _KNOWN_REF.get(q, "")  # 알려진 질문이면 reference 자동 매칭
    tag = "baseline" if ref else "custom"
    final_pairs.append(QAPair(question=q, reference=ref, tag=tag))

if final_pairs:
    n_ref = sum(1 for p in final_pairs if p.reference)
    st.caption(
        f"총 **{len(final_pairs)}개** 질문 | "
        f"🔵 정답 자동매칭 {n_ref}개 | "
        f"🟢 reference 자동생성 {len(final_pairs) - n_ref}개"
    )

# -- 실행 버튼 -----------------------------------------------------------------
if not final_pairs:
    st.warning("질문이 없습니다.")
    st.stop()

if st.button("▶ RAGAS 평가 실행", width="stretch", type="primary"):
    st.session_state.pop(_SESSION_KEY, None)
    try:
        ev = _run_eval(final_pairs, settings)
        st.session_state[_SESSION_KEY] = ev
        if ev.df.empty:
            st.error("평가 가능한 응답이 없습니다.")
        else:
            st.success(f"완료 | {len(ev.df)}개 질문 | {ev.ran_at}")
    except Exception as exc:
        st.error(f"평가 실패: {exc}")

# -- 결과 표시 -----------------------------------------------------------------
ev: EvalResult | None = st.session_state.get(_SESSION_KEY)

if ev is None or ev.df.empty:
    st.info("위 버튼을 눌러 평가를 시작하세요.")
    st.stop()

st.caption(f"평가 시각: {ev.ran_at} | 질문 수: {len(ev.df)}")

# 메트릭 카드
_show_summary_cards(ev.summary)

st.markdown("---")

# 탭으로 차트 분리 — 클릭해서 전환
tab_bar, tab_heat = st.tabs(["📊 메트릭별 점수 (막대)", "🔥 질문 × 메트릭 히트맵"])

with tab_bar:
    _show_bar(ev.summary)

with tab_heat:
    st.caption("🔴 낮음 → 🟡 중간 → 🟢 높음")
    _show_heatmap(ev.df)

# 질문별 상세
st.markdown("---")
st.subheader("질문별 RAG 답변 상세")
_show_per_question(ev.df, ev.records)

# CSV 저장 (접기)
with st.expander("CSV 저장", expanded=False):
    fname = st.text_input(
        "파일명", value=f"welfare_ragas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    )
    if st.button("저장"):
        out = Path("evaluate")
        out.mkdir(exist_ok=True)
        ev.df.to_csv(str(out / fname), index=False, encoding="utf-8-sig")
        st.success(f"저장 완료: evaluate/{fname}")
    st.dataframe(
        ev.df[[c for c in ["user_input", "tag"] + _METRICS if c in ev.df.columns]],
        hide_index=True,
        width="stretch",
    )
