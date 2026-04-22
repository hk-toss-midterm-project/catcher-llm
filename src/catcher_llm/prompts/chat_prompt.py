from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_chat_prompt() -> ChatPromptTemplate:
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
