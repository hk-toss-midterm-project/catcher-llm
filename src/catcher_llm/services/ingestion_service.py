from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.loaders import (
    iter_source_files,
    load_local_documents,
    load_split_local_documents,
)
from catcher_llm.retrievers.vectorstore import build_local_vectorstore, ensure_vectorstore_dir


def discover_source_files(settings: Settings | None = None) -> list[Path]:
    """설정된 원본 데이터 디렉터리에서 적재 가능한 파일을 찾는다."""
    config = settings or get_settings()
    return iter_source_files(config.raw_data_dir)


def _build_manifest(
    documents: Sequence[Document],
    chunks: Sequence[Document],
) -> list[dict[str, str | int]]:
    """원본 문서별 문자 수와 청크 수를 집계한 적재 매니페스트를 만든다."""
    chunk_counts = Counter(str(chunk.metadata["source"]) for chunk in chunks)
    source_stats: dict[str, dict[str, Any]] = {}
    for document in documents:
        source = str(document.metadata["source"])
        if source not in source_stats:
            source_stats[source] = {
                "source": source,
                "chars": 0,
                "chunks": chunk_counts.get(source, 0),
            }
        source_stats[source]["chars"] += len(document.page_content)
    return list(source_stats.values())


def _ingest_documents(
    source_files: Sequence[Path],
    *,
    chunk_size: int,
    chunk_overlap: int,
    settings: Settings,
    manifest_name: str,
    selection_only: bool,
) -> dict[str, str | int]:
    """문서를 로드하고 청크 및 매니페스트를 만든 뒤 벡터스토어 생성을 수행한다."""
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    ensure_vectorstore_dir(settings)

    documents = load_local_documents(settings.raw_data_dir, source_files=source_files)
    chunks = load_split_local_documents(
        settings.raw_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_files=source_files,
    )
    manifest = _build_manifest(documents, chunks)

    manifest_path = settings.processed_data_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    build_kwargs: dict[str, object] = {}
    if selection_only:
        build_kwargs["source_files"] = source_files

    build_local_vectorstore(
        settings.raw_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=settings,
        **build_kwargs,
    )

    return {
        "documents": len(manifest),
        "chunks": len(chunks),
        "manifest_path": str(manifest_path),
    }


def ingest_selected_documents(
    source_files: Sequence[Path],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    *,
    settings: Settings | None = None,
    manifest_name: str = "ingestion_manifest.json",
) -> dict[str, str | int]:
    """선택된 원본 문서를 로드, 청킹하고 적재 결과 매니페스트를 저장한다."""
    config = settings or get_settings()
    return _ingest_documents(
        source_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=config,
        manifest_name=manifest_name,
        selection_only=True,
    )


def ingest_local_documents(
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    settings: Settings | None = None,
) -> dict[str, str | int]:
    """설정된 원본 데이터 디렉터리의 모든 지원 문서를 적재한다."""
    config = settings or get_settings()
    return _ingest_documents(
        discover_source_files(config),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=config,
        manifest_name="ingestion_manifest.json",
        selection_only=False,
    )
