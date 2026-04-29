from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}

_MARKDOWN_HEADERS = [
    ("#", "header1"),
    ("##", "header2"),
    ("###", "header3"),
    ("####", "header4"),
]


def load_markdown_file(path: Path) -> list[Document]:
    """마크다운 파일을 헤더 계층 구조 기준으로 분할해 Document 목록으로 변환한다.

    각 섹션은 상위 헤더 정보를 metadata에 포함하므로 RAG 검색 시
    어느 절에서 나온 청크인지 추적할 수 있다.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )
    docs = splitter.split_text(text)
    source = str(path)
    for doc in docs:
        doc.metadata.setdefault("source", source)
        doc.metadata["source"] = source
    return docs


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
    if suffix == ".md":
        return load_markdown_file(path)
    if suffix == ".txt":
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
