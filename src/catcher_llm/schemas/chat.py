from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChatRole = Literal["assistant", "system", "user"]


@dataclass(slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(slots=True)
class ChatTurnResult:
    reply: str
    route: str
    error: str | None = None
