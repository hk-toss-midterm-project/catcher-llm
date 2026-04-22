from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_summary_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the input in short, well-structured bullet points."),
            ("human", "{text}"),
        ]
    )
