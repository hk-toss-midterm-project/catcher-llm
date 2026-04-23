from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.loaders import load_split_local_documents


def load_retrieval_seed(settings: Settings | None = None) -> Sequence[Document]:
    """현재 설정의 원본 문서를 로드하고 청킹해 검색 초기 데이터로 반환한다."""
    config = settings or get_settings()
    return load_split_local_documents(
        config.raw_data_dir,
        chunk_size=config.rag_chunk_size,
        chunk_overlap=config.rag_chunk_overlap,
    )
