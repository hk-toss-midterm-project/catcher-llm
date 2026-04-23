from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_summary_prompt() -> ChatPromptTemplate:
    """입력 텍스트를 짧은 bullet point로 요약하는 프롬프트를 만든다."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the input in short, well-structured bullet points."),
            ("human", "{text}"),
        ]
    )
