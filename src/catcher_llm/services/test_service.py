from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.vectorstore import get_local_retriever
from catcher_llm.schemas.rag import RetrievedChunk
from catcher_llm.utils.helpers import extract_page_number


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
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
) -> RetrieverTestResult:
    """질문으로 로컬 retriever를 직접 호출해 검색 결과 청크를 반환한다."""
    config = settings or get_settings()
    if source_files is None:
        retriever = get_local_retriever(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            settings=config,
        )
    else:
        retriever = get_local_retriever(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            source_files=source_files,
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
            page_number=extract_page_number(document.metadata),
        )
        for document in documents
    ]
    return RetrieverTestResult(question=question, contexts=contexts)
