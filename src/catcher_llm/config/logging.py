from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """루트 로거가 아직 설정되지 않았을 때 기본 로그 포맷과 레벨을 적용한다."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
