from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.rag_prompt import build_rag_prompt


def build_rag_chain(settings: Settings | None = None):
    return build_rag_prompt() | get_chat_model(settings) | StrOutputParser()
