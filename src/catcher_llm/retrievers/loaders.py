from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def iter_source_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_source_documents(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [
            Document(
                page_content=path.read_text(encoding="utf-8", errors="ignore"),
                metadata={"source": str(path)},
            )
        ]
    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
        for document in documents:
            document.metadata["source"] = str(path)
        return documents
    return []


def load_local_documents(
    raw_dir: Path,
    *,
    source_files: Sequence[Path] | None = None,
) -> list[Document]:
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
    return split_documents(
        load_local_documents(raw_dir, source_files=source_files),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
