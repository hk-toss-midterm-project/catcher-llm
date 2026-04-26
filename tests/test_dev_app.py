from __future__ import annotations

from catcher_llm.ui.dev_navigation import get_dev_page_specs, get_repo_root


def test_get_repo_root_points_to_project_root() -> None:
    """개발 앱 네비게이션이 저장소 루트를 올바르게 기준점으로 삼는지 검증한다."""
    root = get_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "dev_pages").exists()


def test_get_dev_page_specs_registers_dev_pages_directory() -> None:
    """개발용 Streamlit 페이지들이 사이드바 표시 순서와 메타데이터로 등록되는지 검증한다."""
    specs = get_dev_page_specs()

    assert [spec.path.name for spec in specs] == [
        "01_chat.py",
        "02_retriever_probe.py",
        "03_document_rag.py",
        "04_daily_analysis.py",
        "05_consumption_interpretation.py",
    ]
    assert [spec.title for spec in specs] == [
        "Chat",
        "Retriever 테스트",
        "문서 RAG",
        "일일 소비 분석",
        "소비 해석 체인",
    ]
    assert [spec.icon for spec in specs] == ["💬", "🔎", "📄", "📊", "🧭"]
    assert [spec.default for spec in specs] == [True, False, False, False, False]
    assert all(spec.path.is_file() for spec in specs)
