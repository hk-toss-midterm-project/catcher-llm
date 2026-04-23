from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class ConsoleDebugCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **_: Any) -> None:
        """체인 시작 시 체인 이름과 입력 키를 디버그 로그로 남긴다."""
        logger.debug("chain_start name=%s keys=%s", serialized.get("name"), list(inputs))

    def on_chain_end(self, outputs: dict[str, Any], **_: Any) -> None:
        """체인 종료 시 출력 키 목록을 디버그 로그로 남긴다."""
        logger.debug("chain_end keys=%s", list(outputs))
