from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from catcher_llm.config.settings import Settings
from catcher_llm.prompts.rag_prompt import get_saving_tips_prompt
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.schemas.rag import RAGResponse
from catcher_llm.services.rag.config import DocumentKind, get_rag_pipeline_config
from catcher_llm.services.rag.core import generate_rag_reply


def _get_saving_tips_source_files(raw_data_dir: Path) -> list[Path]:
    """절약 팁 RAG가 우선 사용할 가이드 텍스트 원본 파일 목록을 반환한다."""
    guide_path = raw_data_dir / "txt" / "saving_guides.txt"
    if guide_path.exists():
        return [guide_path]
    return []


def _normalize_saving_tips_answer(question: str, answer: str) -> str:
    """절약 팁 답변을 질문 재진술형 한 문장으로 정규화한다."""
    normalized_answer = answer.strip()
    if not normalized_answer:
        return normalized_answer

    question_stem = question.strip()
    for suffix in ("무엇인가?", "무엇인가", "인가?", "인가"):
        if question_stem.endswith(suffix):
            question_stem = question_stem.removesuffix(suffix).strip()
            break

    if not question_stem:
        return normalized_answer

    if normalized_answer.startswith(question_stem):
        return normalized_answer

    if question_stem.endswith(("은", "는")):
        return f"{question_stem} {normalized_answer}"

    return f"{question_stem}은 {normalized_answer}"


def generate_saving_tips_rag_reply(
    question: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 2,
    *,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    """절약 팁 문서 설정을 주입해 RAG 답변을 생성한다."""
    pipeline_config = get_rag_pipeline_config(DocumentKind.SAVING_TIPS, settings=settings)
    prompt = get_saving_tips_prompt()
    source_files = _get_saving_tips_source_files(pipeline_config.raw_data_dir)
    rag_response = generate_rag_reply(
        question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        history=history,
        raw_data_dir=pipeline_config.raw_data_dir,
        source_files=source_files or pipeline_config.source_files,
        settings=settings,
        prompt=prompt,
        temperature=temperature,
    )
    if isinstance(rag_response, RAGResponse):
        rag_response.answer = _normalize_saving_tips_answer(question, rag_response.answer)
    return rag_response
