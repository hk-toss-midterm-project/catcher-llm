from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"
from langchain_core.documents import Document
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def iter_source_files(raw_dir: Path) -> list[Path]:
    """원본 문서 디렉터리에서 지원하는 확장자의 파일 목록을 정렬해 반환한다."""
    if not raw_dir.exists():
        return []

    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_source_documents(path: Path) -> list[Document]:
    """단일 원본 파일을 LangChain Document 목록으로 변환한다."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [
            Document(
                page_content=path.read_text(encoding="utf-8", errors="ignore"),
                metadata={"source": str(path)},
            )
        ]
    if suffix == ".pdf":
        documents = OpenDataLoaderPDFLoader(file_path=str(path)).load()
        for document in documents:
            document.metadata["source"] = str(path)
        return documents
    return []


def load_local_documents(
    raw_dir: Path,
    *,
    source_files: Sequence[Path] | None = None,
) -> list[Document]:
    """로컬 원본 파일들을 순회하며 Document 목록으로 로드한다."""
    documents: list[Document] = []
    files = source_files or iter_source_files(raw_dir)
    for path in files:
        documents.extend(load_source_documents(path))
    return documents


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Document 목록을 RAG 검색에 사용할 청크 단위로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def load_split_local_documents(
    raw_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    source_files: Sequence[Path] | None = None,
) -> list[Document]:
    """로컬 문서를 로드한 뒤 지정한 크기와 중복 범위로 청킹한다."""
    return split_documents(
        load_local_documents(raw_dir, source_files=source_files),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
