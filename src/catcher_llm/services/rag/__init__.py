from __future__ import annotations

from catcher_llm.services.rag.catcher_consumption_benchmark import (
    generate_catcher_consumption_benchmark_rag_reply,
)
from catcher_llm.services.rag.config import DocumentKind, RAGPipelineConfig
from catcher_llm.services.rag.core import generate_rag_reply, rag_target, retrieve_context_records
from catcher_llm.services.rag.kca_report import generate_kca_report_rag_reply
from catcher_llm.services.rag.saving_tips import generate_saving_tips_rag_reply
from catcher_llm.services.rag.welfare import generate_welfare_rag_reply

__all__ = [
    "DocumentKind",
    "RAGPipelineConfig",
    "generate_kca_report_rag_reply",
    "generate_rag_reply",
    "generate_saving_tips_rag_reply",
    "generate_catcher_consumption_benchmark_rag_reply",
    "generate_welfare_rag_reply",
    "rag_target",
    "retrieve_context_records",
]
