from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.schemas.chat import ChatMessage


def format_chat_history(history: Sequence[ChatMessage]) -> str:
    """채팅 메시지 목록을 프롬프트에 넣기 쉬운 문자열로 변환한다."""
    if not history:
        return "No previous messages."

    return "\n".join(f"{message.role.title()}: {message.content}" for message in history)


def format_serialized_context(contexts: Sequence[dict[str, str]]) -> str:
    """검색 컨텍스트 목록을 출처와 본문이 포함된 프롬프트 문자열로 직렬화한다."""
    if not contexts:
        return "No retrieved context."

    return "\n\n".join(
        f"[{index}] {item['source']}\n{item['content']}"
        for index, item in enumerate(contexts, start=1)
    )
