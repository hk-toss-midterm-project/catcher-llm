from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEV_PAGE_METADATA: dict[str, tuple[str, str]] = {
    "01_chat": ("Chat", "💬"),
    "02_retriever_probe": ("Retriever 테스트", "🔎"),
    "03_document_rag": ("문서 RAG", "📄"),
    "04_daily_analysis": ("일일 소비 분석", "📊"),
    "05_consumption_interpretation": ("소비 해석 체인", "🧭"),
}


@dataclass(frozen=True)
class DevPageSpec:
    path: Path
    title: str
    icon: str
    default: bool = False


def get_repo_root() -> Path:
    """현재 모듈 파일 위치를 기준으로 저장소 루트 경로를 계산한다."""
    return Path(__file__).resolve().parents[3]


def _get_dev_pages_dir() -> Path:
    """개발용 Streamlit 페이지 스크립트가 모여 있는 디렉터리 경로를 반환한다."""
    return get_repo_root() / "dev_pages"


def _prettify_title(stem: str) -> str:
    """파일 stem에서 접두 번호를 제거하고 화면 표시용 제목으로 다듬는다."""
    stem = stem.split("_", maxsplit=1)[-1]
    return stem.replace("_", " ").title()


def get_dev_page_specs() -> list[DevPageSpec]:
    """개발용 페이지 파일들을 읽어 사이드바 구성에 쓸 명세 목록으로 변환한다."""
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
