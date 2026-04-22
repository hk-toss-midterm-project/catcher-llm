from __future__ import annotations

import argparse

from catcher_llm.config.settings import get_settings
from catcher_llm.services.chat_service import generate_reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Give me a short overview of this project structure.",
    )
    args = parser.parse_args()

    result = generate_reply(args.prompt, history=[], settings=get_settings())
    print(f"route={result.route}")
    print(result.reply)


if __name__ == "__main__":
    main()
