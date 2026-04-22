from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.chains.rag_chain import build_rag_chain
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.vectorstore import get_local_retriever
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse, RetrievedChunk
from catcher_llm.utils.helpers import format_chat_history, format_serialized_context


def retrieve_context_records(
    question: str,
    *,
    settings: Settings | None = None,
) -> list[dict[str, str]]:
    config = settings or get_settings()
    retriever = get_local_retriever(config)
    if retriever is None:
        return []

    documents = retriever.invoke(question)
    return [
        {
            "source": str(document.metadata.get("source", "unknown")),
            "content": document.page_content,
        }
        for document in documents
    ]


def generate_rag_reply(
    question: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
) -> RAGResponse:
    config = settings or get_settings()
    if not config.has_openai_key:
        return RAGResponse(
            answer="OPENAI_API_KEY is not set. Add it to .env before using the RAG flow.",
            contexts=[],
            sources=[],
            error="missing_openai_api_key",
        )

    try:
        context_records = retrieve_context_records(question, settings=config)
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

    contexts = [
        RetrievedChunk(source=item["source"], content=item["content"]) for item in context_records
    ]
    sources = list(dict.fromkeys(item.source for item in contexts))
    return RAGResponse(answer=answer, contexts=contexts, sources=sources)


def rag_target(
    inputs: dict[str, str],
    *,
    settings: Settings | None = None,
) -> dict[str, object]:
    result = generate_rag_reply(inputs["question"], settings=settings)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "contexts": [
            {"source": chunk.source, "content": chunk.content} for chunk in result.contexts
        ],
        "error": result.error,
    }
