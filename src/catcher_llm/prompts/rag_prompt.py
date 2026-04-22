from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt() -> ChatPromptTemplate:
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
