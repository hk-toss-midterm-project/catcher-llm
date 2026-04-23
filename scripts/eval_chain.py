"""CLI에서 채팅 서비스를 한 번 실행해 라우팅과 응답을 확인한다."""

from __future__ import annotations

import argparse

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.services.chat_service import generate_reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Give me a short overview of this project structure.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_langsmith_env(settings)
    result = generate_reply(args.prompt, history=[], settings=settings)
    print(f"route={result.route}")
    print(result.reply)


if __name__ == "__main__":
    main()
