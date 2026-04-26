from __future__ import annotations

from collections.abc import Sequence

from catcher_llm.chains.chat_chain import build_chat_chain, build_session_chat_chain
from catcher_llm.chains.router_chain import route_request
from catcher_llm.chains.summary_chain import build_summary_chain
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.chat import ChatMessage, ChatTurnResult
from catcher_llm.services.rag_service import generate_rag_reply
from catcher_llm.services.session_history_service import (
    append_session_chat_turn,
    load_session_chat_messages,
)
from catcher_llm.utils.helpers import format_chat_history


def _build_session_invoke_config(session_id: str) -> dict[str, dict[str, str]]:
    """LangChain 세션 체인 호출에 필요한 configurable 설정을 만든다."""
    return {"configurable": {"session_id": session_id}}


def _resolve_history(
    history: Sequence[ChatMessage] | None,
    *,
    session_id: str | None,
    settings: Settings,
) -> Sequence[ChatMessage]:
    """세션 ID가 있으면 SQLite 히스토리를 우선 사용하고 없으면 전달받은 히스토리를 쓴다."""
    if session_id is None:
        return history or []

    stored_history = load_session_chat_messages(session_id, settings=settings)
    if stored_history:
        return stored_history
    return history or []


def _persist_non_chat_turn(
    session_id: str | None,
    *,
    user_input: str,
    reply: str,
    settings: Settings,
) -> str | None:
    """세션 채팅 체인이 아닌 흐름의 사용자 입력과 응답을 SQLite에 저장한다."""
    if session_id is None:
        return None

    try:
        append_session_chat_turn(
            session_id,
            user_input=user_input,
            ai_response=reply,
            settings=settings,
        )
    except Exception as exc:
        return str(exc)
    return None


def generate_reply(
    user_input: str,
    history: Sequence[ChatMessage] | None = None,
    settings: Settings | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    chat_temperature: float = 0.0,
    summary_temperature: float = 0.0,
    rag_temperature: float = 0.0,
    session_id: str | None = None,
) -> ChatTurnResult:
    """사용자 입력을 라우팅한 뒤 채팅, 요약, RAG 중 알맞은 응답을 생성한다."""
    config = settings or get_settings()
    route = route_request(user_input)

    chat_model_error = config.chat_model_error
    if chat_model_error is not None:
        return ChatTurnResult(
            reply=config.get_chat_model_error_message("chat flow")
            or "Chat model configuration is invalid.",
            route=route,
            error=chat_model_error,
        )

    payload: dict[str, str]
    if route == "summary":
        chain = build_summary_chain(config, temperature=summary_temperature)
        payload = {"text": user_input}
    elif route == "rag":
        active_history = _resolve_history(history, session_id=session_id, settings=config)
        rag_result = generate_rag_reply(
            user_input,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            history=active_history,
            settings=config,
            temperature=rag_temperature,
        )
        reply = rag_result.answer
        if rag_result.sources and not rag_result.error:
            joined_sources = "\n".join(f"- {source}" for source in rag_result.sources)
            reply = f"{reply}\n\nSources:\n{joined_sources}"
        if rag_result.error is None:
            persistence_error = _persist_non_chat_turn(
                session_id,
                user_input=user_input,
                reply=reply,
                settings=config,
            )
            if persistence_error is not None:
                return ChatTurnResult(reply=reply, route=route, error=persistence_error)
        return ChatTurnResult(reply=reply, route=route, error=rag_result.error)
    elif session_id is not None:
        chain = build_session_chat_chain(config, temperature=chat_temperature)
        payload = {"input": user_input}
    else:
        chain = build_chat_chain(config, temperature=chat_temperature)
        payload = {
            "history": format_chat_history(history or []),
            "input": user_input,
        }

    try:
        if route == "chat" and session_id is not None:
            reply = chain.invoke(payload, config=_build_session_invoke_config(session_id))
        else:
            reply = chain.invoke(payload)
    except Exception as exc:
        return ChatTurnResult(
            reply=f"Request failed: {exc}",
            route=route,
            error=str(exc),
        )

    if route != "chat":
        persistence_error = _persist_non_chat_turn(
            session_id,
            user_input=user_input,
            reply=reply,
            settings=config,
        )
        if persistence_error is not None:
            return ChatTurnResult(reply=reply, route=route, error=persistence_error)

    return ChatTurnResult(reply=reply, route=route)
