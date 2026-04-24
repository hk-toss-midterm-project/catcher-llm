from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DevPageSpec:
    path: Path
    title: str
    icon: str
    default: bool = False


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_dev_page_specs() -> list[DevPageSpec]:
    root = get_repo_root()

    return [
        DevPageSpec(
            path=root / "dev_pages" / "retriever_probe.py",
            title="Retriever 테스트",
            icon="🔎",
            default=True,
        )
    ]
