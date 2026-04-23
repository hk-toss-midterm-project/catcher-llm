from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt() -> ChatPromptTemplate:
    """검색 컨텍스트만 근거로 답변하도록 지시하는 RAG 프롬프트를 만든다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the retrieved context. "
                "If the context is insufficient, say that you do not know. "
                "Mention the source path when it helps the answer.",
            ),
            (
                "human",
                "Conversation so far:\n{history}\n\nRetrieved context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )
