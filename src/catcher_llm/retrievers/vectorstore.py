from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.llm.models import get_embeddings_model
from catcher_llm.retrievers.loaders import iter_source_files, load_split_local_documents

_VECTORSTORE_CACHE: dict[tuple[object, ...], InMemoryVectorStore] = {}


def ensure_vectorstore_dir(settings: Settings | None = None) -> Path:
    """벡터스토어 저장 디렉터리를 생성하고 경로를 반환한다."""
    config = settings or get_settings()
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return config.vectorstore_dir


def _build_cache_key(config: Settings) -> tuple[object, ...]:
    """임베딩 설정과 원본 파일 상태를 기반으로 벡터스토어 캐시 키를 만든다."""
    file_signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in iter_source_files(config.raw_data_dir)
    )
    return (
        config.embedding_model,
        config.rag_chunk_size,
        config.rag_chunk_overlap,
        file_signature,
    )


def build_local_vectorstore(settings: Settings | None = None) -> InMemoryVectorStore | None:
    """로컬 문서 청크로 인메모리 벡터스토어를 만들거나 캐시된 값을 반환한다."""
    config = settings or get_settings()
    chunks = load_split_local_documents(
        config.raw_data_dir,
        chunk_size=config.rag_chunk_size,
        chunk_overlap=config.rag_chunk_overlap,
    )
    if not chunks:
        return None

    cache_key = _build_cache_key(config)
    if cache_key not in _VECTORSTORE_CACHE:
        _VECTORSTORE_CACHE[cache_key] = InMemoryVectorStore.from_documents(
            chunks,
            embedding=get_embeddings_model(config),
        )
    return _VECTORSTORE_CACHE[cache_key]


def get_local_retriever(settings: Settings | None = None):
    """로컬 벡터스토어에서 RAG 검색에 사용할 retriever를 생성한다."""
    config = settings or get_settings()
    vectorstore = build_local_vectorstore(config)
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": config.rag_top_k})
