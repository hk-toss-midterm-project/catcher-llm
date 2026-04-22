from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.schemas.chat import ChatMessage


def format_chat_history(history: Sequence[ChatMessage]) -> str:
    if not history:
        return "No previous messages."

    return "\n".join(f"{message.role.title()}: {message.content}" for message in history)


def format_serialized_context(contexts: Sequence[dict[str, str]]) -> str:
    if not contexts:
        return "No retrieved context."

    return "\n\n".join(
        f"[{index}] {item['source']}\n{item['content']}"
        for index, item in enumerate(contexts, start=1)
    )
