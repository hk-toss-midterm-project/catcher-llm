from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.llm.models import get_embeddings_model
from catcher_llm.retrievers.loaders import iter_source_files, load_split_local_documents

_VECTORSTORE_CACHE: dict[tuple[object, ...], FAISS] = {}
_VECTORSTORE_METADATA_FILENAME = "metadata.json"
_SELECTED_VECTORSTORE_DIRNAME = "_selected"


def ensure_vectorstore_dir(settings: Settings | None = None) -> Path:
    """벡터스토어 저장 디렉터리를 생성하고 경로를 반환한다."""
    config = settings or get_settings()
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return config.vectorstore_dir


def _normalize_source_files(
    raw_data_dir: Path | str,
    source_files: Sequence[Path] | None = None,
) -> list[Path]:
    """선택된 원본 파일 목록을 절대 경로 기준의 안정적인 순서로 정규화한다."""
    actual_data_dir = Path(raw_data_dir)
    if source_files is None:
        return iter_source_files(actual_data_dir)
    return sorted((Path(path) for path in source_files), key=lambda path: str(path.resolve()))


def _get_file_signature(
    raw_data_dir: Path | str,
    source_files: Sequence[Path] | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """원본 파일 경로, 수정 시각, 크기를 묶어 캐시 비교용 시그니처를 만든다."""
    files = _normalize_source_files(raw_data_dir, source_files)
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)


def _build_cache_key(
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    source_files: Sequence[Path] | None = None,
) -> tuple[object, ...]:
    """임베딩 설정과 원본 파일 상태를 기반으로 벡터스토어 캐시 키를 만든다."""
    actual_data_dir = Path(raw_data_dir)
    normalized_source_files = _normalize_source_files(actual_data_dir, source_files)
    file_signature = _get_file_signature(actual_data_dir, normalized_source_files)
    return (
        str(actual_data_dir.resolve()),
        config.embedding_model_provider,
        config.embedding_model_name,
        chunk_size,
        chunk_overlap,
        tuple(str(path) for path in normalized_source_files),
        file_signature,
    )


def _get_store_relative_path(
    config: Settings,
    raw_data_dir: Path | str,
    source_files: Sequence[Path] | None = None,
) -> Path:
    """벡터스토어를 저장할 상대 경로를 원본 디렉터리나 선택 파일 기준으로 계산한다."""
    if source_files:
        return _get_selected_store_relative_path(config, source_files)

    actual_data_dir = Path(raw_data_dir).resolve()
    raw_root = config.raw_data_dir.resolve()

    try:
        return actual_data_dir.relative_to(raw_root)
    except ValueError:
        relative_parts = (
            actual_data_dir.parts[1:] if actual_data_dir.is_absolute() else actual_data_dir.parts
        )
        return Path("_external").joinpath(*relative_parts)


def _get_selected_store_relative_path(
    config: Settings,
    source_files: Sequence[Path],
) -> Path:
    """선택 적재용 벡터스토어 경로를 파일 조합 기준으로 안정적으로 만든다."""
    normalized_source_files = _normalize_source_files(config.raw_data_dir, source_files)
    if len(normalized_source_files) == 1:
        return Path(_SELECTED_VECTORSTORE_DIRNAME) / _to_store_safe_relative_path(
            config,
            normalized_source_files[0],
        ).with_suffix("")

    selection_key = "\n".join(str(path) for path in normalized_source_files)
    digest = sha256(selection_key.encode("utf-8")).hexdigest()[:12]
    return Path(_SELECTED_VECTORSTORE_DIRNAME) / digest


def _to_store_safe_relative_path(config: Settings, path: Path) -> Path:
    """원본 루트 밖의 파일도 저장 경로로 쓸 수 있게 안전한 상대 경로로 변환한다."""
    raw_root = config.raw_data_dir.resolve()
    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(raw_root)
    except ValueError:
        relative_parts = (
            resolved_path.parts[1:] if resolved_path.is_absolute() else resolved_path.parts
        )
        return Path("_external").joinpath(*relative_parts)


def _to_metadata_relative_path(config: Settings, path: Path | str) -> str:
    """벡터스토어 메타데이터에 저장할 경로를 원본 루트 기준 상대 경로로 변환한다."""
    relative_path = _to_store_safe_relative_path(config, Path(path))
    return relative_path.as_posix()


def _get_vectorstore_artifact_paths(
    config: Settings,
    raw_data_dir: Path | str,
    source_files: Sequence[Path] | None = None,
) -> tuple[Path, Path]:
    """벡터스토어 디렉터리와 메타데이터 파일 경로를 함께 반환한다."""
    base_dir = ensure_vectorstore_dir(config)
    store_dir = base_dir / _get_store_relative_path(config, raw_data_dir, source_files)
    return (store_dir, store_dir / _VECTORSTORE_METADATA_FILENAME)


def _get_metadata_file_signature(
    config: Settings,
    raw_data_dir: Path | str,
    source_files: Sequence[Path] | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """원본 파일 시그니처의 경로 부분을 메타데이터용 상대 경로로 바꾼다."""
    return tuple(
        (
            _to_metadata_relative_path(config, Path(path)),
            modified_at,
            file_size,
        )
        for path, modified_at, file_size in _get_file_signature(raw_data_dir, source_files)
    )


def _build_store_metadata(
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    source_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """저장된 벡터스토어가 현재 설정과 같은지 비교할 메타데이터를 구성한다."""
    actual_data_dir = Path(raw_data_dir)
    metadata = {
        "raw_data_dir": _to_metadata_relative_path(config, actual_data_dir),
        "embedding_provider": config.embedding_model_provider,
        "embedding_model": config.embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_signature": [
            list(item)
            for item in _get_metadata_file_signature(config, actual_data_dir, source_files)
        ],
    }
    if source_files is not None:
        metadata["source_files"] = [
            _to_metadata_relative_path(config, path) for path in source_files
        ]
    return metadata


def _load_store_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """메타데이터 JSON을 읽어오고 형식이 올바르지 않으면 `None`을 반환한다."""
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
    """FAISS 저장 결과에 필요한 인덱스 파일 두 개가 모두 있는지 확인한다."""
    return (store_dir / "index.faiss").exists() and (store_dir / "index.pkl").exists()


def _save_vectorstore(
    vectorstore: FAISS,
    config: Settings,
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    source_files: Sequence[Path] | None = None,
) -> None:
    """벡터스토어와 현재 생성 조건 메타데이터를 디스크에 함께 저장한다."""
    store_dir, metadata_path = _get_vectorstore_artifact_paths(
        config,
        raw_data_dir,
        source_files,
    )
    store_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_dir))
    metadata = _build_store_metadata(
        config,
        raw_data_dir,
        chunk_size,
        chunk_overlap,
        source_files,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_local_vectorstore(
    raw_data_dir: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
) -> FAISS | None:
    """로컬 문서 청크로 FAISS 벡터스토어를 만들거나 저장된 값을 로드한다."""
    config = settings or get_settings()
    actual_data_dir = Path(raw_data_dir)
    selected_source_files = (
        [Path(path) for path in source_files] if source_files is not None else None
    )
    normalized_source_files = (
        _normalize_source_files(actual_data_dir, selected_source_files)
        if selected_source_files is not None
        else None
    )
    cache_key = _build_cache_key(
        config,
        actual_data_dir,
        chunk_size,
        chunk_overlap,
        source_files=normalized_source_files,
    )
    if cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    store_dir, metadata_path = _get_vectorstore_artifact_paths(
        config,
        actual_data_dir,
        normalized_source_files,
    )
    metadata = _load_store_metadata(metadata_path)
    expected_metadata = _build_store_metadata(
        config,
        actual_data_dir,
        chunk_size,
        chunk_overlap,
        normalized_source_files,
    )
    if metadata == expected_metadata and _has_saved_store(store_dir):
        vectorstore = FAISS.load_local(
            str(store_dir),
            get_embeddings_model(config),
            allow_dangerous_deserialization=True,
        )
        _VECTORSTORE_CACHE[cache_key] = vectorstore
        return vectorstore

    chunks = load_split_local_documents(
        actual_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_files=selected_source_files,
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
        normalized_source_files,
    )
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore


def get_local_retriever(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    *,
    raw_data_dir: Path | str | None = None,
    source_files: Sequence[Path] | None = None,
    settings: Settings | None = None,
):
    """로컬 벡터스토어에서 RAG 검색에 사용할 retriever를 생성한다."""
    config = settings or get_settings()
    actual_data_dir = raw_data_dir if raw_data_dir is not None else config.raw_data_dir
    vectorstore = build_local_vectorstore(
        actual_data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_files=source_files,
        settings=config,
    )
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
