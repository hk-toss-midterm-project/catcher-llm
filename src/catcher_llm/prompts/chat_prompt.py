from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_chat_prompt() -> ChatPromptTemplate:
    """일반 채팅 요청에 사용할 시스템/사용자 메시지 프롬프트를 만든다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a pragmatic AI assistant in a Streamlit application. "
                "Give concise, actionable answers.",
            ),
            ("human", "Conversation so far:\n{history}\n\nUser request:\n{input}"),
        ]
    )
