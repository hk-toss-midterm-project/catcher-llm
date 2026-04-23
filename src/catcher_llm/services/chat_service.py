from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.chains.chat_chain import build_chat_chain
from catcher_llm.chains.router_chain import route_request
from catcher_llm.chains.summary_chain import build_summary_chain
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.chat import ChatMessage, ChatTurnResult
from catcher_llm.services.rag_service import generate_rag_reply
from catcher_llm.utils.helpers import format_chat_history


def generate_reply(
    user_input: str,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
) -> ChatTurnResult:
    """사용자 입력을 라우팅한 뒤 채팅, 요약, RAG 중 알맞은 응답을 생성한다."""
    config = settings or get_settings()
    route = route_request(user_input)

    if not config.has_openai_key:
        return ChatTurnResult(
            reply="OPENAI_API_KEY is not set. Add it to .env before using the chat flow.",
            route=route,
            error="missing_openai_api_key",
        )

    payload: dict[str, str]
    if route == "summary":
        chain = build_summary_chain(config)
        payload = {"text": user_input}
    elif route == "rag":
        rag_result = generate_rag_reply(
            user_input,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            history=history,
            settings=config,
        )
        reply = rag_result.answer
        if rag_result.sources and not rag_result.error:
            joined_sources = "\n".join(f"- {source}" for source in rag_result.sources)
            reply = f"{reply}\n\nSources:\n{joined_sources}"
        return ChatTurnResult(reply=reply, route=route, error=rag_result.error)
    else:
        chain = build_chat_chain(config)
        payload = {
            "history": format_chat_history(history or []),
            "input": user_input,
        }

    try:
        reply = chain.invoke(payload)
    except Exception as exc:
        return ChatTurnResult(
            reply=f"Request failed: {exc}",
            route=route,
            error=str(exc),
        )

    return ChatTurnResult(reply=reply, route=route)
