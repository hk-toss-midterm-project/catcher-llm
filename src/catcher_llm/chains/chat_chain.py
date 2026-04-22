from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.chat_prompt import build_chat_prompt


def build_chat_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
):
    chat_model = llm or get_chat_model(settings)
    return build_chat_prompt() | chat_model | StrOutputParser()
