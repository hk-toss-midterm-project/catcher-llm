from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.services.test_service import invoke_retriever_question
from catcher_llm.ui.components import render_sidebar

settings = get_settings()
configure_langsmith_env(settings)

st.set_page_config(page_title=f"{settings.app_name} | Retriever Test", layout="wide")

render_sidebar(settings)

st.title("Retriever Test")
st.caption("LLM 답변 생성 없이 get_local_retriever().invoke() 결과만 바로 확인한다.")

with st.form("retriever-test-form"):
    question = st.text_area(
        "질문",
        value="보험료 할인 방법에 대해 알려줘",
        height=100,
    )
    controls = st.columns(3)
    chunk_size = controls[0].number_input("Chunk size", min_value=100, value=800, step=50)
    chunk_overlap = controls[1].number_input("Chunk overlap", min_value=0, value=120, step=10)
    top_k = controls[2].number_input("Top K", min_value=1, value=4, step=1)
    submitted = st.form_submit_button("Run retriever test", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("질문을 입력하세요.")
    else:
        with st.spinner("Retriever invoking..."):
            result = invoke_retriever_question(
                question.strip(),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                top_k=int(top_k),
                settings=settings,
            )

        if result.error == "missing_documents":
            st.error(
                "검색할 문서가 없습니다. data/raw에 문서를 넣고 Docs 페이지에서 적재를 먼저 실행하세요."
            )
        elif result.error:
            st.error(f"Retriever 호출에 실패했습니다: {result.error}")
        else:
            st.success(f"{len(result.contexts)}개의 청크를 찾았습니다.")

        for index, context in enumerate(result.contexts, start=1):
            with st.expander(f"Match {index} | {context.source}", expanded=index == 1):
                st.write(context.content)
