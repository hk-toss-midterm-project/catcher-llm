from __future__ import annotations

from pathlib import Path

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.test_service import invoke_retriever_question

settings = get_settings()


def discover_pdf_corpus_dirs() -> list[Path]:
    pdf_root = settings.raw_data_dir / "pdf"
    if not pdf_root.exists():
        return []
    return sorted(path for path in pdf_root.iterdir() if path.is_dir())


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("개발용 retriever 점검 페이지")


st.title("🔎 Retriever 테스트")
st.caption("선택한 PDF 코퍼스 디렉터리를 기준으로 retriever.invoke() 결과를 확인한다.")

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
)

controls = st.columns(3)
chunk_size = controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
chunk_overlap = controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
top_k = controls[2].number_input("Top K", min_value=1, value=4, step=1)

if st.button("Retriever 실행", use_container_width=True):
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
