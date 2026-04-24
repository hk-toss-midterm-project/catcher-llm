from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from catcher_llm.chains.rag_chain import build_rag_chain
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.vectorstore import get_local_retriever
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.utils.helpers import (
    extract_page_number,
    format_chat_history,
    format_serialized_context,
)


def retrieve_context_records(
    question: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    *,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, str | int | None]]:
    """질문과 관련된 로컬 문서 청크를 검색해 출처와 본문 형태로 반환한다."""
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
        return []

    documents = retriever.invoke(question)
    return [
        {
            "source": str(document.metadata.get("source", "unknown")),
            "content": document.page_content,
            "page_number": extract_page_number(document.metadata),
        }
        for document in documents
    ]


def generate_rag_reply(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    *,
    history: Sequence[ChatMessage] | None = None,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
) -> RAGResponse:
    """문서 검색 결과를 컨텍스트로 사용해 RAG 답변과 출처 정보를 생성한다."""
    config = settings or get_settings()
    if not config.has_openai_key:
        return RAGResponse(
            answer="OPENAI_API_KEY is not set. Add it to .env before using the RAG flow.",
            contexts=[],
            sources=[],
            error="missing_openai_api_key",
        )

    try:
        context_records = retrieve_context_records(
            question,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            raw_data_dir=raw_data_dir,
            source_files=source_files,
            settings=config,
        )
    except Exception as exc:
        return RAGResponse(
            answer=f"RAG retrieval failed: {exc}",
            contexts=[],
            sources=[],
            error=str(exc),
        )

    if not context_records:
        return RAGResponse(
            answer="No local source documents are available yet. Add files to data/raw and run the ingestion script.",
            contexts=[],
            sources=[],
            error="missing_documents",
        )

    try:
        chain = build_rag_chain(config)
        answer = chain.invoke(
            {
                "history": format_chat_history(history or []),
                "context": format_serialized_context(context_records),
                "question": question,
            }
        )
    except Exception as exc:
        return RAGResponse(
            answer=f"RAG generation failed: {exc}",
            contexts=[],
            sources=[],
            error=str(exc),
        )

    contexts: list[RetrievedChunk] = []
    for item in context_records:
        page_number = item.get("page_number")
        contexts.append(
            RetrievedChunk(
                source=str(item["source"]),
                content=str(item["content"]),
                page_number=page_number if isinstance(page_number, int) else None,
            )
        )

    sources = list(dict.fromkeys(item.source for item in contexts))
    return RAGResponse(answer=answer, contexts=contexts, sources=sources)


def rag_target(
    inputs: dict[str, str | int],
    *,
    settings: Settings | None = None,
) -> dict[str, object]:
    """LangSmith 평가기가 호출할 수 있도록 RAG 응답을 dict 형태로 변환한다."""
    chunk_size = int(inputs.get("chunk_size", 800))
    chunk_overlap = int(inputs.get("chunk_overlap", 120))
    top_k = int(inputs.get("top_k", 4))

    result = generate_rag_reply(
        str(inputs["question"]),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        settings=settings,
    )
    return {
        "answer": result.answer,
        "sources": result.sources,
        "contexts": [
            {
                "source": chunk.source,
                "content": chunk.content,
                "page_number": chunk.page_number,
            }
            for chunk in result.contexts
        ],
        "error": result.error,
    }
