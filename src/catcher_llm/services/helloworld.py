from catcher_llm.chains.chat_chain import build_chat_chain
from catcher_llm.config.settings import get_settings


def build_hello_world_chain() -> str:
    settings = get_settings()
    if not settings.has_openai_key:
        return "OPENAI_API_KEY is not set. Add it to .env before using hello world."

    chain = build_chat_chain(settings=settings)
    try:
        return chain.invoke(
            {
                "history": "",
                "input": "안녕하세요!",
            }
        )
    except Exception as exc:
        return f"Request failed: {exc}"
