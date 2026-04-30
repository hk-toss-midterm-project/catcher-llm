from __future__ import annotations

from pathlib import Path

import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.schemas.consumption_feedback import RetrievedAdviceContext
from catcher_llm.services.consumption_feedback.daily_feedback import retrieve_feedback_contexts
from catcher_llm.services.ingestion_service import discover_source_files
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.rag_service import generate_rag_reply

settings = get_settings()
configure_langsmith_env(settings)
_FEEDBACK_DOCUMENT_KINDS: tuple[DocumentKind, ...] = (
    DocumentKind.SAVING_TIPS,
    DocumentKind.SELF_REPORT,
    DocumentKind.USER_REPORT,
    DocumentKind.WELFARE,
    DocumentKind.KCA_REPORT,
)
_DOCUMENT_KIND_LABELS: dict[DocumentKind, str] = {
    DocumentKind.SAVING_TIPS: "saving_tips | 절약 팁",
    DocumentKind.SELF_REPORT: "self_report | 소비 자기진단",
    DocumentKind.USER_REPORT: "user_report | 사용자 동향",
    DocumentKind.WELFARE: "welfare | 복지 정책",
    DocumentKind.KCA_REPORT: "kca_report | KCA 리포트",
}


def _format_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(settings.raw_data_dir))
    except ValueError:
        return str(path)


def _get_default_source_index(source_files: list[Path]) -> int:
    """기본 단일 문서 선택값으로 복지 정책 PDF 위치를 반환한다."""
    preferred_path = settings.raw_data_dir / "pdf" / "welfare" / "2026_hope_ladder_selected.pdf"
    preferred_resolved = preferred_path.resolve()

    for index, path in enumerate(source_files):
        if path.resolve() == preferred_resolved:
            return index

    return 0


def _format_document_kind(document_kind: DocumentKind) -> str:
    """문서 종류 enum을 개발 화면 선택지 라벨로 변환한다."""
    return _DOCUMENT_KIND_LABELS[document_kind]


def _format_score(score: float | None) -> str:
    """유용성 점수를 화면에 표시할 문자열로 변환한다."""
    if score is None:
        return "-"
    return f"{score:.2f}"


def _render_feedback_contexts(contexts: list[RetrievedAdviceContext]) -> None:
    """문서 종류별 피드백 RAG 검색 결과를 표와 원문 expander로 표시한다."""
    if not contexts:
        st.info("유용하다고 판단된 문서 근거가 없습니다.")
        return

    st.success(f"{len(contexts)}개의 문서 근거를 찾았습니다.")
    st.dataframe(
        [
            {
                "document_kind": context.document_kind or "-",
                "score": _format_score(context.usefulness_score),
                "source": context.source,
                "page": context.page_number or "-",
                "reason": context.usefulness_reason or "-",
            }
            for context in contexts
        ],
        width="stretch",
        hide_index=True,
    )
    for index, context in enumerate(contexts, start=1):
        page_label = f" | p.{context.page_number}" if context.page_number is not None else ""
        title = (
            f"{index}. {context.document_kind or '-'} | "
            f"score {_format_score(context.usefulness_score)} | {context.source}{page_label}"
        )
        with st.expander(title, expanded=index == 1):
            st.caption(context.query)
            st.write(context.content)


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("문서 종류별 피드백 RAG와 단일 문서 RAG를 확인합니다.")


st.title("📄 문서 RAG")
st.caption("문서 종류별 피드백 RAG 검색과 단일 문서 기반 RAG 답변을 확인합니다.")

feedback_tab, single_doc_tab = st.tabs(["문서 종류별 피드백 RAG", "단일 문서 RAG"])

with feedback_tab:
    feedback_question = st.text_area(
        "피드백 RAG 검색 질의",
        value="배달 주문 지출 줄이는 방법",
        height=100,
        key="document_feedback_rag_query",
    )
    selected_document_kinds = st.multiselect(
        "검색할 문서 종류",
        options=list(_FEEDBACK_DOCUMENT_KINDS),
        default=list(_FEEDBACK_DOCUMENT_KINDS),
        format_func=_format_document_kind,
        key="document_feedback_rag_document_kinds",
    )
    feedback_controls = st.columns(4)
    feedback_chunk_size = feedback_controls[0].number_input(
        "Feedback chunk size",
        min_value=100,
        value=800,
        step=50,
        key="document_feedback_chunk_size",
    )
    feedback_chunk_overlap = feedback_controls[1].number_input(
        "Feedback chunk overlap",
        min_value=0,
        value=120,
        step=10,
        key="document_feedback_chunk_overlap",
    )
    feedback_top_k = feedback_controls[2].number_input(
        "Feedback top K",
        min_value=1,
        value=3,
        step=1,
        key="document_feedback_top_k",
    )
    usefulness_threshold = feedback_controls[3].number_input(
        "Usefulness threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="document_feedback_usefulness_threshold",
    )

    if st.button("문서 종류별 RAG 검색 실행", width="stretch"):
        if not feedback_question.strip():
            st.warning("검색 질의를 입력하세요.")
        elif not selected_document_kinds:
            st.warning("검색할 문서 종류를 하나 이상 선택하세요.")
        else:
            with st.spinner("문서 종류별 RAG 검색 중..."):
                feedback_contexts = retrieve_feedback_contexts(
                    [feedback_question.strip()],
                    chunk_size=int(feedback_chunk_size),
                    chunk_overlap=int(feedback_chunk_overlap),
                    top_k=int(feedback_top_k),
                    document_kinds=selected_document_kinds,
                    usefulness_threshold=float(usefulness_threshold),
                    settings=settings,
                )
            _render_feedback_contexts(feedback_contexts)

with single_doc_tab:
    source_files = discover_source_files(settings)
    if not source_files:
        st.info("data/raw 아래에 검색할 문서가 없습니다.")
    else:
        selected_source = st.selectbox(
            "조회할 문서",
            options=source_files,
            index=_get_default_source_index(source_files),
            format_func=_format_source_path,
        )

        question = st.text_area(
            "질문",
            value="임산부 지원 정책을 알려줘",
            height=100,
            key="single_document_rag_query",
        )

        controls = st.columns(3)
        chunk_size = controls[0].number_input(
            "Chunk size",
            min_value=100,
            value=800,
            step=50,
        )
        chunk_overlap = controls[1].number_input(
            "Chunk overlap",
            min_value=0,
            value=120,
            step=10,
        )
        top_k = controls[2].number_input("Top K", min_value=1, value=4, step=1)

        if st.button("문서 RAG 실행", width="stretch"):
            if not question.strip():
                st.warning("질문을 입력하세요.")
            else:
                with st.spinner("RAG 답변 생성 중..."):
                    result = generate_rag_reply(
                        question=question.strip(),
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        top_k=int(top_k),
                        source_files=[selected_source],
                        settings=settings,
                    )

                st.write(f"문서 경로: `{_format_source_path(selected_source)}`")

                if result.error:
                    st.error(result.answer)
                else:
                    st.subheader("답변")
                    st.write(result.answer)

                    if result.sources:
                        st.caption("Sources")
                        for source in result.sources:
                            st.write(f"- {source}")

                st.subheader("검색 청크")
                if not result.contexts:
                    st.info("검색된 청크가 없습니다.")
                else:
                    for index, context in enumerate(result.contexts, start=1):
                        page_label = (
                            f" | p.{context.page_number}" if context.page_number is not None else ""
                        )
                        with st.expander(
                            f"Chunk {index}{page_label} | {_format_source_path(Path(context.source))}",
                            expanded=index == 1,
                        ):
                            st.write(context.content)
