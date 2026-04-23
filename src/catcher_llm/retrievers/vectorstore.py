from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.llm.models import get_embeddings_model
from catcher_llm.retrievers.loaders import iter_source_files, load_split_local_documents

_VECTORSTORE_CACHE: dict[tuple[object, ...], FAISS] = {}
_VECTORSTORE_INDEX_NAME = "index"
_VECTORSTORE_STORE_DIRNAME = "local_faiss"
_VECTORSTORE_METADATA_FILENAME = "local_faiss_metadata.json"


def ensure_vectorstore_dir(settings: Settings | None = None) -> Path:
    """벡터스토어 저장 디렉터리를 생성하고 경로를 반환한다."""
    config = settings or get_settings()
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return config.vectorstore_dir


def _get_file_signature(raw_data_dir: Path | str) -> tuple[tuple[str, int, int], ...]:
    actual_data_dir = Path(raw_data_dir)
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in iter_source_files(actual_data_dir)
    )


def _build_cache_key(
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[object, ...]:
    """임베딩 설정과 원본 파일 상태를 기반으로 벡터스토어 캐시 키를 만든다."""
    actual_data_dir = Path(raw_data_dir)
    file_signature = _get_file_signature(actual_data_dir)
    return (
        str(actual_data_dir.resolve()),
        config.embedding_model,
        chunk_size,
        chunk_overlap,
        file_signature,
    )


def _get_vectorstore_artifact_paths(config: Settings) -> tuple[Path, Path]:
    base_dir = ensure_vectorstore_dir(config)
    return (
        base_dir / _VECTORSTORE_STORE_DIRNAME,
        base_dir / _VECTORSTORE_METADATA_FILENAME,
    )


def _build_store_metadata(
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    actual_data_dir = Path(raw_data_dir)
    return {
        "raw_data_dir": str(actual_data_dir.resolve()),
        "embedding_model": config.embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_signature": [list(item) for item in _get_file_signature(actual_data_dir)],
    }


def _load_store_metadata(metadata_path: Path) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _has_saved_store(store_dir: Path) -> bool:
    return (store_dir / f"{_VECTORSTORE_INDEX_NAME}.faiss").exists() and (
        store_dir / f"{_VECTORSTORE_INDEX_NAME}.pkl"
    ).exists()


def _save_vectorstore(
    vectorstore: FAISS,
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    store_dir, metadata_path = _get_vectorstore_artifact_paths(config)
    store_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_dir), index_name=_VECTORSTORE_INDEX_NAME)
    metadata = _build_store_metadata(config, raw_data_dir, chunk_size, chunk_overlap)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_local_vectorstore(
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    settings: Settings | None = None,
) -> FAISS | None:
    """로컬 문서 청크로 FAISS 벡터스토어를 만들거나 저장된 값을 로드한다."""
    config = settings or get_settings()
    actual_data_dir = Path(raw_data_dir)
    cache_key = _build_cache_key(config, actual_data_dir, chunk_size, chunk_overlap)
    if cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    store_dir, metadata_path = _get_vectorstore_artifact_paths(config)
    metadata = _load_store_metadata(metadata_path)
    expected_metadata = _build_store_metadata(config, actual_data_dir, chunk_size, chunk_overlap)
    if metadata == expected_metadata and _has_saved_store(store_dir):
        vectorstore = FAISS.load_local(
            str(store_dir),
            get_embeddings_model(config),
            index_name=_VECTORSTORE_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        _VECTORSTORE_CACHE[cache_key] = vectorstore
        return vectorstore

    chunks = load_split_local_documents(
        actual_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return None

    vectorstore = FAISS.from_documents(
        chunks,
        embedding=get_embeddings_model(config),
    )
    _save_vectorstore(
        vectorstore,
        config,
        actual_data_dir,
        chunk_size,
        chunk_overlap,
    )
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore


def get_local_retriever(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    *,
    raw_data_dir: Path | str | None = None,
    settings: Settings | None = None,
):
    """로컬 벡터스토어에서 RAG 검색에 사용할 retriever를 생성한다."""
    config = settings or get_settings()
    actual_data_dir = raw_data_dir if raw_data_dir is not None else config.raw_data_dir
    vectorstore = build_local_vectorstore(
        actual_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=config,
    )
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
