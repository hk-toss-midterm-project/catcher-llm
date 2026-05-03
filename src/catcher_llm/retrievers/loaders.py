from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import opendataloader_pdf
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
_PDF_MARKDOWN_PAGE_SEPARATOR = "\n\n<!-- catcher-page:%page-number% -->\n\n"
_PDF_MARKDOWN_PAGE_SEPARATOR_PATTERN = re.compile(r"<!-- catcher-page:(\d+) -->")


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


def _find_converted_markdown_file(output_dir: Path, source_path: Path) -> Path | None:
    """opendataloader-pdf 출력 디렉터리에서 변환된 마크다운 파일을 찾는다."""
    markdown_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )
    if not markdown_files:
        return None

    preferred_names = {
        f"{source_path.stem}.md",
        f"{source_path.stem}.markdown",
    }
    for path in markdown_files:
        if path.name in preferred_names:
            return path
    return markdown_files[0]


def _build_pdf_document(source_path: Path, page_content: str, page_number: int) -> Document:
    """PDF 한 페이지의 텍스트를 검색용 Document로 변환한다."""
    return Document(
        page_content=page_content.strip(),
        metadata={
            "source": str(source_path),
            "page": page_number - 1,
            "page_number": page_number,
            "loader": "opendataloader_pdf",
        },
    )


def _documents_from_pdf_markdown(markdown_text: str, source_path: Path) -> list[Document]:
    """opendataloader-pdf 마크다운 결과를 페이지 구분자 기준으로 Document 목록화한다."""
    parts = _PDF_MARKDOWN_PAGE_SEPARATOR_PATTERN.split(markdown_text)
    documents: list[Document] = []

    first_page_content = parts[0].strip()
    if first_page_content:
        documents.append(_build_pdf_document(source_path, first_page_content, page_number=1))

    part_index = 1
    while part_index < len(parts) - 1:
        page_number = int(parts[part_index])
        page_content = parts[part_index + 1].strip()
        if page_content:
            documents.append(
                _build_pdf_document(source_path, page_content, page_number=page_number)
            )
        part_index += 2

    if not documents and markdown_text.strip():
        documents.append(_build_pdf_document(source_path, markdown_text.strip(), page_number=1))
    return documents


def _load_pdf_file_with_pypdf(path: Path) -> list[Document]:
    """opendataloader-pdf 변환 실패 시 기존 PyPDF 로더로 PDF를 읽는다."""
    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        document.metadata["source"] = str(path)
        document.metadata.setdefault("loader", "pypdf")
        page = document.metadata.get("page")
        if isinstance(page, int):
            document.metadata.setdefault("page_number", page + 1)
    return documents


def load_pdf_file(path: Path) -> list[Document]:
    """PDF 파일을 opendataloader-pdf로 마크다운 변환한 뒤 페이지별 Document로 읽는다."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            opendataloader_pdf.convert(
                str(path),
                output_dir=str(output_dir),
                format="markdown",
                quiet=True,
                markdown_page_separator=_PDF_MARKDOWN_PAGE_SEPARATOR,
                image_output="off",
            )
            markdown_path = _find_converted_markdown_file(output_dir, path)
            if markdown_path is None:
                return _load_pdf_file_with_pypdf(path)
            documents = _documents_from_pdf_markdown(
                markdown_path.read_text(encoding="utf-8", errors="ignore"),
                path,
            )
            if documents:
                return documents
    except Exception:
        return _load_pdf_file_with_pypdf(path)
    return _load_pdf_file_with_pypdf(path)


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
        return load_pdf_file(path)
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
