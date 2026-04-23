from __future__ import annotations

from dataclasses import dataclass

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.vectorstore import get_local_retriever
from catcher_llm.schemas.rag import RetrievedChunk


@dataclass(slots=True)
class RetrieverTestResult:
    question: str
    contexts: list[RetrievedChunk]
    error: str | None = None


def invoke_retriever_question(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    *,
    settings: Settings | None = None,
) -> RetrieverTestResult:
    """질문으로 로컬 retriever를 직접 호출해 검색 결과 청크를 반환한다."""
    config = settings or get_settings()
    retriever = get_local_retriever(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        settings=config,
    )
    if retriever is None:
        return RetrieverTestResult(
            question=question,
            contexts=[],
            error="missing_documents",
        )

    try:
        documents = retriever.invoke(question)
    except Exception as exc:
        return RetrieverTestResult(
            question=question,
            contexts=[],
            error=str(exc),
        )

    contexts = [
        RetrievedChunk(
            source=str(document.metadata.get("source", "unknown")),
            content=document.page_content,
        )
        for document in documents
    ]
    return RetrieverTestResult(question=question, contexts=contexts)
