from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.rag_prompt import get_kca_report_prompt
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse
from catcher_llm.services.rag.config import DocumentKind, get_rag_pipeline_config
from catcher_llm.services.rag.core import generate_rag_reply


def generate_kca_report_rag_reply(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 2,
    *,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    """KCA 보고서 전용 프롬프트와 검색 설정으로 사실형 RAG 답변을 생성한다."""
    pipeline_config = get_rag_pipeline_config(DocumentKind.KCA_REPORT, settings=settings)

    prompt = get_kca_report_prompt()

    return generate_rag_reply(
        question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        history=history,
        raw_data_dir=pipeline_config.raw_data_dir,
        source_files=pipeline_config.source_files,
        settings=settings,
        prompt=prompt,
        temperature=temperature,
    )
