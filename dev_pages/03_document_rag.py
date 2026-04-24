from __future__ import annotations

from pathlib import Path

import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.services.ingestion_service import discover_source_files
from catcher_llm.services.rag_service import generate_rag_reply

settings = get_settings()
configure_langsmith_env(settings)


def _format_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(settings.raw_data_dir))
    except ValueError:
        return str(path)


def _get_default_source_index(source_files: list[Path]) -> int:
    preferred_path = settings.raw_data_dir / "pdf" / "welfare" / "2026_hope_ladder_selected.pdf"
    preferred_resolved = preferred_path.resolve()

    for index, path in enumerate(source_files):
        if path.resolve() == preferred_resolved:
            return index

    return 0


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("선택한 문서 하나만 기준으로 RAG 답변을 확인합니다.")


st.title("📄 문서 RAG")
st.caption("단일 문서를 선택하고, 해당 문서만 근거로 RAG 답변과 검색 청크를 확인합니다.")

source_files = discover_source_files(settings)
if not source_files:
    st.info("data/raw 아래에 검색할 문서가 없습니다.")
    st.stop()

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
)

controls = st.columns(3)
chunk_size = controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = controls[2].number_input("Top K", min_value=1, value=4, step=1)

if st.button("문서 RAG 실행", use_container_width=True):
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
