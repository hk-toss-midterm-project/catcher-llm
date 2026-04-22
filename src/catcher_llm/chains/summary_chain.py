from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.summary_prompt import build_summary_prompt


def build_summary_chain(settings: Settings | None = None):
    return build_summary_prompt() | get_chat_model(settings) | StrOutputParser()
