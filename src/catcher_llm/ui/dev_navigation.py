from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEV_PAGE_METADATA: dict[str, tuple[str, str]] = {
    "01_chat": ("Chat", "💬"),
    "02_retriever_probe": ("Retriever 테스트", "🔎"),
}


@dataclass(frozen=True)
class DevPageSpec:
    path: Path
    title: str
    icon: str
    default: bool = False


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_dev_pages_dir() -> Path:
    return get_repo_root() / "dev_pages"


def _prettify_title(stem: str) -> str:
    stem = stem.split("_", maxsplit=1)[-1]
    return stem.replace("_", " ").title()


def get_dev_page_specs() -> list[DevPageSpec]:
    specs: list[DevPageSpec] = []

    for index, path in enumerate(sorted(_get_dev_pages_dir().glob("*.py"))):
        title, icon = _DEV_PAGE_METADATA.get(
            path.stem,
            (_prettify_title(path.stem), "🧪"),
        )
        specs.append(
            DevPageSpec(
                path=path,
                title=title,
                icon=icon,
                default=index == 0,
            )
        )

    return specs
