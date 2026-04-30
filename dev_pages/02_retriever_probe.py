from __future__ import annotations

from pathlib import Path

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import RetrievedAdviceContext
from catcher_llm.services.consumption_feedback.daily_feedback import retrieve_feedback_contexts
from catcher_llm.services.rag.config import DocumentKind
from catcher_llm.services.test_service import invoke_retriever_question

settings = get_settings()
_FEEDBACK_DOCUMENT_KINDS: tuple[DocumentKind, ...] = (
    DocumentKind.SAVING_TIPS,
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK,
    DocumentKind.USER_REPORT,
    DocumentKind.WELFARE,
    DocumentKind.KCA_REPORT,
)
_DOCUMENT_KIND_LABELS: dict[DocumentKind, str] = {
    DocumentKind.SAVING_TIPS: "saving_tips | 절약 팁",
    DocumentKind.CATCHER_CONSUMPTION_BENCHMARK: (
        "catcher_consumption_benchmark | Catcher 2018.07-12 소비 벤치마크"
    ),
    DocumentKind.USER_REPORT: "user_report | 사용자 동향",
    DocumentKind.WELFARE: "welfare | 복지 정책",
    DocumentKind.KCA_REPORT: "kca_report | KCA 리포트",
}


def discover_pdf_corpus_dirs() -> list[Path]:
    """data/raw/pdf 아래의 문서군 디렉터리 목록을 반환한다."""
    pdf_root = settings.raw_data_dir / "pdf"
    if not pdf_root.exists():
        return []
    return sorted(path for path in pdf_root.iterdir() if path.is_dir())


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
    st.caption("개발용 RAG retriever 점검 페이지")


st.title("🔎 Retriever 테스트")
st.caption("문서 종류별 피드백 RAG와 기존 PDF 코퍼스 직접 검색 결과를 확인한다.")

feedback_tab, corpus_tab = st.tabs(["문서 종류별 피드백 RAG", "PDF 코퍼스 직접 검색"])

with feedback_tab:
    feedback_question = st.text_area(
        "피드백 RAG 검색 질의",
        value="배달 주문 지출 줄이는 방법",
        height=100,
        key="feedback_rag_query",
    )
    selected_document_kinds = st.multiselect(
        "검색할 문서 종류",
        options=list(_FEEDBACK_DOCUMENT_KINDS),
        default=list(_FEEDBACK_DOCUMENT_KINDS),
        format_func=_format_document_kind,
        key="feedback_rag_document_kinds",
    )
    feedback_controls = st.columns(4)
    feedback_chunk_size = feedback_controls[0].number_input(
        "Feedback chunk size",
        min_value=100,
        value=800,
        step=50,
    )
    feedback_chunk_overlap = feedback_controls[1].number_input(
        "Feedback chunk overlap",
        min_value=0,
        value=120,
        step=10,
    )
    feedback_top_k = feedback_controls[2].number_input(
        "Feedback top K",
        min_value=1,
        value=3,
        step=1,
    )
    usefulness_threshold = feedback_controls[3].number_input(
        "Usefulness threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
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

with corpus_tab:
    corpus_dirs = discover_pdf_corpus_dirs()
    if not corpus_dirs:
        st.info("data/raw/pdf 아래에 코퍼스 디렉터리가 없습니다.")
        st.stop()

    selected_corpus = st.selectbox(
        "조회할 PDF 코퍼스",
        options=corpus_dirs,
        format_func=lambda path: str(path.relative_to(settings.raw_data_dir)),
    )

    question = st.text_area(
        "질문",
        value="보험료 할인 방법에 대해 알려줘",
        height=100,
        key="corpus_retriever_query",
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

    if st.button("Retriever 실행", width="stretch"):
        if not question.strip():
            st.warning("질문을 입력하세요.")
        else:
            with st.spinner("Retriever invoking..."):
                result = invoke_retriever_question(
                    question=question.strip(),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    top_k=int(top_k),
                    raw_data_dir=selected_corpus,
                    settings=settings,
                )

            st.write(f"코퍼스 경로: `{selected_corpus}`")

            if result.error == "missing_documents":
                st.error("선택한 코퍼스에 검색할 문서가 없습니다.")
            elif result.error:
                st.error(f"Retriever 호출에 실패했습니다: {result.error}")
            else:
                st.success(f"{len(result.contexts)}개의 청크를 찾았습니다.")

            for index, context in enumerate(result.contexts, start=1):
                with st.expander(f"Match {index} | {context.source}", expanded=index == 1):
                    st.write(context.content)
