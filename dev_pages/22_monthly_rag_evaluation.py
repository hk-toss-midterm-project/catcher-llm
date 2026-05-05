"""월간 피드백 RAG 평가 개발 페이지.

월간 피드백 파이프라인을 실행한 뒤 RAGAS, 프로젝트 전용 규칙 기반 평가,
선택형 LLM Judge 평가를 한 화면에서 확인한다.

사용 방법:
  uv run streamlit run dev_app.py -> 사이드바에서 '월간 피드백 RAG 평가' 선택
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from catcher_llm.config.settings import Settings, configure_langsmith_env, get_settings
from catcher_llm.evaluation.monthly_rag import (
    RAGAS_METRIC_LABELS,
    RAGAS_METRICS,
    MonthlyLangSmithEvaluationResult,
    MonthlyLlmJudgeResult,
    MonthlyRagasRecord,
    MonthlyRuleSummary,
    build_monthly_rag_query_specs,
    evaluate_monthly_feedback_rules,
    run_monthly_langsmith_evaluation,
    run_monthly_llm_judge,
    run_monthly_rag_query_generation,
    run_monthly_ragas_evaluation,
    summarize_ragas_scores,
)
from catcher_llm.schemas.consumption_feedback import MonthlyFeedbackServiceResult
from catcher_llm.services.consumption_feedback.monthly_feedback import generate_monthly_feedback
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)
from catcher_llm.ui.feedback_progress import (
    MONTHLY_FEEDBACK_PROGRESS_STEPS,
    create_feedback_progress_callback,
)

_SESSION_KEY = "monthly_rag_evaluation_result"


@dataclass
class MonthlyRagEvaluationPageResult:
    """Streamlit 세션에 저장할 월간 RAG 평가 실행 결과를 묶는다."""

    service_result: MonthlyFeedbackServiceResult
    rule_summary: MonthlyRuleSummary
    ragas_records: list[MonthlyRagasRecord] = field(default_factory=list)
    ragas_frame: pd.DataFrame | None = None
    ragas_summary: dict[str, float] = field(default_factory=dict)
    ragas_error: str | None = None
    langsmith_result: MonthlyLangSmithEvaluationResult | None = None
    langsmith_error: str | None = None
    llm_judge_result: MonthlyLlmJudgeResult | None = None
    llm_judge_error: str | None = None
    ran_at: str = ""


def _format_score(value: float | None) -> str:
    """점수 값을 화면 표시용 문자열로 변환한다."""
    if value is None:
        return "-"
    return f"{value:.3f}"


def _score_color(value: float | None) -> str:
    """점수 크기에 따라 Streamlit 텍스트용 상태 색상을 반환한다."""
    if value is None:
        return "gray"
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "orange"
    return "red"


def _run_monthly_rag_evaluation(
    *,
    member_id: int,
    analysis_month: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    max_queries: int,
    run_ragas: bool,
    run_langsmith: bool,
    run_llm_judge: bool,
    langsmith_dataset_name: str,
    langsmith_experiment_prefix: str,
    feedback_temperature: float,
    settings: Settings,
) -> MonthlyRagEvaluationPageResult:
    """월간 피드백 생성과 선택된 RAG 평가들을 순서대로 실행한다."""
    progress_callback = create_feedback_progress_callback(MONTHLY_FEEDBACK_PROGRESS_STEPS)
    service_result = generate_monthly_feedback(
        member_id=member_id,
        analysis_month=analysis_month,
        settings=settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        max_queries=max_queries,
        feedback_temperature=feedback_temperature,
        timing_callback=progress_callback,
    )
    rule_summary = evaluate_monthly_feedback_rules(service_result)
    page_result = MonthlyRagEvaluationPageResult(
        service_result=service_result,
        rule_summary=rule_summary,
        ran_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    specs = build_monthly_rag_query_specs(service_result) if run_ragas or run_langsmith else []

    if run_ragas:
        try:
            if not specs:
                page_result.ragas_error = "평가할 문서 RAG 검색 쿼리가 없습니다."
            else:
                page_result.ragas_records = run_monthly_rag_query_generation(
                    specs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    settings=settings,
                )
                page_result.ragas_frame = run_monthly_ragas_evaluation(
                    page_result.ragas_records,
                    settings=settings,
                )
                page_result.ragas_summary = summarize_ragas_scores(page_result.ragas_frame)
        except Exception as exc:
            page_result.ragas_error = str(exc)

    if run_langsmith:
        try:
            if not specs:
                page_result.langsmith_error = "LangSmith에 보낼 문서 RAG 검색 쿼리가 없습니다."
            else:
                page_result.langsmith_result = run_monthly_langsmith_evaluation(
                    specs,
                    settings=settings,
                    dataset_name=langsmith_dataset_name,
                    experiment_prefix=langsmith_experiment_prefix,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
        except Exception as exc:
            page_result.langsmith_error = str(exc)

    if run_llm_judge:
        try:
            page_result.llm_judge_result = run_monthly_llm_judge(
                service_result,
                settings=settings,
            )
        except Exception as exc:
            page_result.llm_judge_error = str(exc)

    return page_result


def _render_score_cards(result: MonthlyRagEvaluationPageResult) -> None:
    """규칙, RAGAS, LLM Judge, 검색 문서 수를 요약 카드로 표시한다."""
    ragas_average = (
        sum(result.ragas_summary.values()) / len(result.ragas_summary)
        if result.ragas_summary
        else None
    )
    llm_score = result.llm_judge_result.total_score if result.llm_judge_result is not None else None
    context_count = len(result.service_result.retrieved_contexts)

    columns = st.columns(4)
    columns[0].metric(
        "Rule Pass Rate",
        f"{result.rule_summary.pass_rate:.1%}",
        help="프로젝트 월간 피드백 규칙 기반 평가 통과율",
    )
    columns[1].metric(
        "RAGAS Avg",
        _format_score(ragas_average),
        help="Context Precision/Recall, Faithfulness, Answer Relevancy 평균",
    )
    columns[2].metric(
        "LLM Judge",
        _format_score(llm_score),
        help="선택 실행한 월간 피드백 RAG 커스텀 LLM Judge 총점",
    )
    columns[3].metric("검색 문서", f"{context_count}개")


def _render_ragas_chart(summary: dict[str, float]) -> None:
    """RAGAS 메트릭 평균 점수를 막대 그래프로 표시한다."""
    if not summary:
        st.info("표시할 RAGAS 점수가 없습니다.")
        return
    labels = [RAGAS_METRIC_LABELS[metric] for metric in RAGAS_METRICS if metric in summary]
    values = [summary[metric] for metric in RAGAS_METRICS if metric in summary]
    colors = [
        "#2f9e44" if value >= 0.8 else "#f08c00" if value >= 0.5 else "#c92a2a" for value in values
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=340,
        yaxis=dict(range=[0, 1.1], title="점수"),
        xaxis_title="메트릭",
        margin=dict(t=24, b=30, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def _render_ragas_records(records: list[MonthlyRagasRecord]) -> None:
    """RAGAS 평가 전 generate_rag_reply로 생성한 쿼리별 문서 RAG 답변을 표시한다."""
    if not records:
        return

    st.subheader("쿼리별 문서 RAG 답변")
    for index, record in enumerate(records, start=1):
        label = f"{index}. {record.document_kind or 'unknown'} | {record.user_input[:80]}"
        with st.expander(label, expanded=index == 1):
            if record.error:
                st.error(record.error)
            st.markdown("**generate_rag_reply 답변**")
            st.write(record.response or "응답 없음")
            st.markdown("**Reference**")
            st.write(record.reference)
            st.caption(f"검색 문서 수: {record.source_count}")


def _render_rule_table(summary: MonthlyRuleSummary) -> None:
    """월간 피드백 전용 규칙 평가 결과를 표로 표시한다."""
    frame = pd.DataFrame(
        [
            {
                "status": "PASS" if item.passed else "FAIL",
                "key": item.key,
                "label": item.label,
                "detail": item.detail,
            }
            for item in summary.rule_evaluations
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")


def _render_feedback_preview(result: MonthlyFeedbackServiceResult) -> None:
    """생성된 월간 피드백의 핵심 본문과 미션을 표시한다."""
    if result.error:
        st.error(f"월간 피드백 생성 실패: {result.error}")
    if result.feedback is None:
        st.warning("피드백 결과가 비어 있습니다.")
        return

    feedback = result.feedback
    st.subheader(feedback.summary_title)
    st.write(feedback.feedback_message)
    st.info(feedback.next_month_mission)


def _render_contexts(result: MonthlyFeedbackServiceResult) -> None:
    """월간 피드백에 사용된 RAG 검색 질의와 문서 컨텍스트를 표시한다."""
    st.subheader("RAG 검색 질의")
    if result.retrieval_queries:
        st.write(result.retrieval_queries)
    else:
        st.info("생성된 검색 질의가 없습니다.")

    st.subheader("검색된 문서 근거")
    if not result.retrieved_contexts:
        st.info("검색된 문서 근거가 없습니다.")
        return

    for index, context in enumerate(result.retrieved_contexts, start=1):
        page_label = f" | p.{context.page_number}" if context.page_number is not None else ""
        kind_label = f" | {context.document_kind}" if context.document_kind else ""
        with st.expander(
            f"{index}. {context.query} | {context.source}{kind_label}{page_label}",
            expanded=index <= 2,
        ):
            if context.usefulness_score is not None:
                st.caption(
                    f"usefulness={context.usefulness_score:.2f} | {context.usefulness_reason or ''}"
                )
            st.write(context.content)


def _render_llm_judge(result: MonthlyLlmJudgeResult | None, error: str | None) -> None:
    """LLM Judge 실행 결과나 오류를 화면에 표시한다."""
    if error:
        st.error(f"LLM Judge 실패: {error}")
    if result is None:
        st.info("LLM Judge를 실행하지 않았습니다.")
        return

    score_columns = st.columns(4)
    score_columns[0].markdown(
        f":{_score_color(result.total_score)}[**Total** {_format_score(result.total_score)}]"
    )
    score_columns[1].markdown(
        f":{_score_color(result.groundedness_score)}"
        f"[**Groundedness** {_format_score(result.groundedness_score)}]"
    )
    score_columns[2].markdown(
        f":{_score_color(result.retrieval_use_score)}"
        f"[**Retrieval Use** {_format_score(result.retrieval_use_score)}]"
    )
    score_columns[3].markdown(
        f":{_score_color(result.product_fit_score)}"
        f"[**Product Fit** {_format_score(result.product_fit_score)}]"
    )
    if result.reason:
        st.write(result.reason)
    with st.expander("LLM Judge 원문"):
        st.code(result.raw_output, language="json")


def _render_langsmith_result(
    result: MonthlyLangSmithEvaluationResult | None,
    error: str | None,
) -> None:
    """LangSmith Experiment 실행 결과나 오류를 화면에 표시한다."""
    if error:
        st.error(f"LangSmith 평가 실패: {error}")
    if result is None:
        st.info("LangSmith 평가를 실행하지 않았습니다.")
        return

    columns = st.columns(3)
    columns[0].metric("Dataset", result.dataset_name)
    columns[1].metric("Examples", f"{result.example_count}개")
    columns[2].metric("Experiment", result.experiment_name or "-")
    if result.experiment_url:
        st.link_button("LangSmith Experiment 열기", result.experiment_url)


def _save_ragas_csv(frame: pd.DataFrame | None) -> None:
    """RAGAS 결과 CSV를 evaluate 디렉터리에 저장하는 UI를 렌더링한다."""
    if frame is None or frame.empty:
        return

    with st.expander("RAGAS CSV 저장"):
        default_name = f"monthly_feedback_ragas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        filename = st.text_input("파일명", value=default_name)
        if st.button("RAGAS CSV 저장"):
            output_dir = Path("evaluate")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / filename
            frame.to_csv(output_path, index=False, encoding="utf-8-sig")
            st.success(f"저장 완료: {output_path}")


settings = configure_langsmith_env(get_settings())

with st.sidebar:
    st.title("Catcher Dev")
    st.caption("월간 피드백 RAG를 RAGAS, 커스텀 규칙, LLM Judge로 평가합니다.")
    st.write(f"LLM: `{settings.chat_model_label}`")
    st.write(f"Embedding: `{settings.embedding_model_label}`")
    if settings.has_langsmith_key:
        st.success("LangSmith tracing 활성화")
    else:
        st.warning("LangSmith API Key 없음")

st.title("월간 피드백 RAG 평가")
st.caption(
    "RAGAS는 월간 피드백 생성 중 나온 검색 쿼리별 문서 RAG 답변을 평가하고, "
    "최종 피드백은 프로젝트 전용 규칙과 LLM Judge로 점검합니다."
)

render_date_picker_styles()
top_controls = st.columns(2)
member_id = int(top_controls[0].number_input("Member ID", min_value=1, value=1, step=1))
with top_controls[1]:
    analysis_month = select_month(
        "분석 월",
        default_month=DEFAULT_CALENDAR_MONTH,
        key="monthly_rag_eval_month",
    )

retrieval_controls = st.columns(4)
chunk_size = int(
    retrieval_controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
)
chunk_overlap = int(
    retrieval_controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
)
top_k = int(retrieval_controls[2].number_input("Top K", min_value=1, value=6, step=1))
max_queries = int(retrieval_controls[3].number_input("Max queries", min_value=1, value=4, step=1))

eval_controls = st.columns(4)
run_ragas = eval_controls[0].checkbox("RAGAS 실행", value=True)
run_langsmith = eval_controls[1].checkbox("LangSmith 평가 실행", value=False)
run_llm_judge = eval_controls[2].checkbox("LLM Judge 실행", value=False)
feedback_temperature = float(
    eval_controls[3].number_input(
        "Feedback temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )
)

langsmith_controls = st.columns(2)
langsmith_dataset_name = langsmith_controls[0].text_input(
    "LangSmith dataset",
    value=f"catcher-monthly-feedback-rag-eval-{analysis_month}",
)
langsmith_experiment_prefix = langsmith_controls[1].text_input(
    "LangSmith experiment prefix",
    value="monthly-feedback-rag",
)

if st.button("월간 피드백 RAG 평가 실행", type="primary", width="stretch"):
    with st.spinner("월간 피드백 생성 및 RAG 평가 실행 중..."):
        st.session_state[_SESSION_KEY] = _run_monthly_rag_evaluation(
            member_id=member_id,
            analysis_month=analysis_month,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_queries=max_queries,
            run_ragas=run_ragas,
            run_langsmith=run_langsmith,
            run_llm_judge=run_llm_judge,
            langsmith_dataset_name=langsmith_dataset_name,
            langsmith_experiment_prefix=langsmith_experiment_prefix,
            feedback_temperature=feedback_temperature,
            settings=settings,
        )

page_result: MonthlyRagEvaluationPageResult | None = st.session_state.get(_SESSION_KEY)
if page_result is None:
    st.info("위 버튼을 눌러 월간 피드백 RAG 평가를 시작하세요.")
    st.stop()

st.caption(f"평가 시각: {page_result.ran_at}")
_render_score_cards(page_result)

tab_overview, tab_ragas, tab_rules, tab_langsmith, tab_judge, tab_contexts, tab_raw = st.tabs(
    [
        "요약",
        "RAGAS",
        "커스텀 규칙",
        "LangSmith",
        "LLM Judge",
        "검색/근거",
        "원본 JSON",
    ]
)

with tab_overview:
    _render_feedback_preview(page_result.service_result)
    st.subheader("실패 규칙")
    failed_rules = [item for item in page_result.rule_summary.rule_evaluations if not item.passed]
    if failed_rules:
        for item in failed_rules:
            st.error(f"{item.label}: {item.detail}")
    else:
        st.success("프로젝트 월간 피드백 규칙을 모두 통과했습니다.")

with tab_ragas:
    if page_result.ragas_error:
        st.error(f"RAGAS 실패: {page_result.ragas_error}")
    _render_ragas_chart(page_result.ragas_summary)
    _render_ragas_records(page_result.ragas_records)
    if page_result.ragas_frame is not None:
        visible_columns = [
            column
            for column in ["document_kind", "user_input", "source_count", *RAGAS_METRICS]
            if column in page_result.ragas_frame.columns
        ]
        st.dataframe(
            page_result.ragas_frame[visible_columns],
            hide_index=True,
            width="stretch",
        )
        _save_ragas_csv(page_result.ragas_frame)

with tab_rules:
    st.metric(
        "Rule Pass Rate",
        f"{page_result.rule_summary.pass_rate:.1%}",
        help=f"{page_result.rule_summary.pass_count}/{page_result.rule_summary.total_count} 통과",
    )
    _render_rule_table(page_result.rule_summary)


with tab_langsmith:
    _render_langsmith_result(page_result.langsmith_result, page_result.langsmith_error)


with tab_judge:
    _render_llm_judge(page_result.llm_judge_result, page_result.llm_judge_error)

with tab_contexts:
    _render_contexts(page_result.service_result)

with tab_raw:
    st.subheader("월간 피드백 서비스 결과")
    st.json(page_result.service_result.model_dump())
    if page_result.ragas_frame is not None:
        st.subheader("RAGAS 결과")
        st.json(page_result.ragas_frame.to_dict(orient="records"))
    if page_result.ragas_records:
        st.subheader("쿼리별 문서 RAG 답변")
        st.json([asdict(record) for record in page_result.ragas_records])
    if page_result.llm_judge_result is not None:
        st.subheader("LLM Judge 결과")
        st.json(page_result.llm_judge_result.model_dump())
    if page_result.langsmith_result is not None:
        st.subheader("LangSmith 결과")
        st.json(page_result.langsmith_result.model_dump())
