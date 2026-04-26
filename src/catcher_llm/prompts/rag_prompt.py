from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt(system_instruction: str | None = None) -> ChatPromptTemplate:
    """검색 컨텍스트만 근거로 답변하도록 지시하는 RAG 프롬프트를 만든다."""
    instruction = system_instruction or (
        "검색된 문맥만 근거로 질문에 답하세요. "
        "문맥이 부족하면 알 수 없다고 답하세요. "
        "답변에 도움이 될 때는 출처 경로를 함께 언급하세요."
    )
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                instruction,
            ),
            (
                "human",
                "이전 대화:\n{history}\n\n검색된 문맥:\n{context}\n\n질문:\n{question}",
            ),
        ]
    )
