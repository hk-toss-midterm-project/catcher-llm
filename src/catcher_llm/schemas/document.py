from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentRecord:
    source: str
    content: str
