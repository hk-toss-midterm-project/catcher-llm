from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from catcher_llm.schemas.chat import ChatMessage


def format_chat_history(history: Sequence[ChatMessage]) -> str:
    """채팅 메시지 목록을 프롬프트에 넣기 쉬운 문자열로 변환한다."""
    if not history:
        return "No previous messages."

    return "\n".join(f"{message.role.title()}: {message.content}" for message in history)


def extract_page_number(metadata: Mapping[str, Any]) -> int | None:
    """문서 메타데이터에서 사용자 표시용 1-based 페이지 번호를 추출한다."""
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return page_number

    if isinstance(page_number, str) and page_number.isdigit():
        return int(page_number)

    page = metadata.get("page")
    if isinstance(page, int):
        return page + 1

    if isinstance(page, str) and page.isdigit():
        return int(page) + 1

    return None


def format_serialized_context(contexts: Sequence[Mapping[str, object]]) -> str:
    """검색 컨텍스트 목록을 출처와 본문이 포함된 프롬프트 문자열로 직렬화한다."""
    if not contexts:
        return "No retrieved context."

    return "\n\n".join(
        _format_context_item(index, item) for index, item in enumerate(contexts, start=1)
    )


def _format_context_item(index: int, item: Mapping[str, object]) -> str:
    """단일 검색 컨텍스트를 페이지 번호 포함 여부에 맞춰 표시 문자열로 만든다."""
    source = str(item["source"])
    content = str(item["content"])
    page_number = item.get("page_number")
    if isinstance(page_number, int):
        return f"[{index}] {source} (page {page_number})\n{content}"
    return f"[{index}] {source}\n{content}"
