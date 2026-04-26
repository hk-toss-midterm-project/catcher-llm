from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.prompts.rag_prompt import build_rag_prompt
from catcher_llm.retrievers.loaders import iter_source_files


class DocumentKind(StrEnum):
    KCA_REPORT = "kca_report"
    SAVING_TIPS = "saving_tips"
    SELF_REPORT = "self_report"
    WELFARE = "welfare"


@dataclass(frozen=True, slots=True)
class RAGPipelineConfig:
    document_kind: DocumentKind
    raw_data_dir: Path
    source_files: list[Path]
    prompt: ChatPromptTemplate


_DOCUMENT_KIND_DIRS: dict[DocumentKind, Path] = {
    DocumentKind.KCA_REPORT: Path("pdf") / "kca_report",
    DocumentKind.SAVING_TIPS: Path("pdf") / "saving_tips",
    DocumentKind.SELF_REPORT: Path("pdf") / "self_report",
    DocumentKind.WELFARE: Path("pdf") / "welfare",
}

_DOCUMENT_KIND_SYSTEM_INSTRUCTIONS: dict[DocumentKind, str] = {
    DocumentKind.KCA_REPORT: (
        "검색된 문맥 중 KCA 소비 리포트 내용만 근거로 답하세요. "
        "소비 행태에 대한 근거, 주의할 해석, 출처로 확인되는 인사이트를 중심으로 설명하세요."
    ),
    DocumentKind.SAVING_TIPS: (
        "검색된 문맥 중 절약 팁 내용만 근거로 답하세요. "
        "문맥이 뒷받침하는 범위에서 구체적이고 실행 가능한 절약 방법을 우선 제안하세요."
    ),
    DocumentKind.SELF_REPORT: (
        "검색된 문맥 중 소비 자기진단 리포트 내용만 근거로 답하세요. "
        "문맥에 없는 데이터를 지어내지 말고 소비 패턴과 리포트 인사이트를 설명하세요."
    ),
    DocumentKind.WELFARE: (
        "검색된 문맥 중 복지 정책 내용만 근거로 답하세요. "
        "지원 대상, 혜택, 신청 조건은 문맥에 명시된 경우에만 언급하세요."
    ),
}


def _coerce_document_kind(document_kind: DocumentKind | str) -> DocumentKind:
    """문자열 또는 enum 입력을 표준 문서 종류 enum으로 변환한다."""
    if isinstance(document_kind, DocumentKind):
        return document_kind
    return DocumentKind(document_kind)


def get_rag_pipeline_config(
    document_kind: DocumentKind | str,
    *,
    settings: Settings | None = None,
) -> RAGPipelineConfig:
    """문서 종류에 맞는 원본 파일 목록과 RAG 프롬프트 설정을 반환한다."""
    config = settings or get_settings()
    normalized_kind = _coerce_document_kind(document_kind)
    document_dir = config.raw_data_dir / _DOCUMENT_KIND_DIRS[normalized_kind]
    return RAGPipelineConfig(
        document_kind=normalized_kind,
        raw_data_dir=config.raw_data_dir,
        source_files=iter_source_files(document_dir),
        prompt=build_rag_prompt(
            system_instruction=_DOCUMENT_KIND_SYSTEM_INSTRUCTIONS[normalized_kind]
        ),
    )
