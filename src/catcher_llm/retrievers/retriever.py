from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.loaders import load_split_local_documents


def load_retrieval_seed(
    chunk_size: int,
    chunk_overlap: int,
    *,
    raw_data_dir: Path | None = None,
    settings: Settings | None = None,
) -> Sequence[Document]:
    """지정한 청킹 설정으로 원본 문서를 로드해 검색 초기 데이터를 반환한다."""
    config = settings or get_settings()
    return load_split_local_documents(
        raw_data_dir or config.raw_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
