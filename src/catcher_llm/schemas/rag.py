from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    source: str
    content: str
    score: float | None = None
    page_number: int | None = None


@dataclass(slots=True)
class RAGResponse:
    answer: str
    contexts: list[RetrievedChunk]
    sources: list[str]
    error: str | None = None
