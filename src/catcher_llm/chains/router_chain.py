from __future__ import annotations

from typing import Literal

RouteName = Literal["chat", "rag", "summary"]
SUMMARY_KEYWORDS = ("summary", "summarize", "tl;dr", "요약")
RAG_KEYWORDS = (
    "document",
    "docs",
    "knowledge base",
    "rag",
    "retrieve",
    "search",
    "source",
    "검색",
    "근거",
    "문서",
    "찾아",
)


def route_request(user_input: str) -> RouteName:
    normalized = user_input.lower()
    if any(keyword in normalized for keyword in SUMMARY_KEYWORDS):
        return "summary"
    if any(keyword in normalized for keyword in RAG_KEYWORDS):
        return "rag"
    return "chat"
