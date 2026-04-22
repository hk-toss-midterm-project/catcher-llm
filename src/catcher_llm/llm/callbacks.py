from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class ConsoleDebugCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **_: Any) -> None:
        logger.debug("chain_start name=%s keys=%s", serialized.get("name"), list(inputs))

    def on_chain_end(self, outputs: dict[str, Any], **_: Any) -> None:
        logger.debug("chain_end keys=%s", list(outputs))
