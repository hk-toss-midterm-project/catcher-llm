from __future__ import annotations

from catcher_llm.ui.dev_navigation import get_dev_page_specs, get_repo_root


def test_get_repo_root_points_to_project_root() -> None:
    root = get_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "dev_pages").exists()


def test_get_dev_page_specs_registers_dev_pages_directory() -> None:
    specs = get_dev_page_specs()

    assert [spec.path.name for spec in specs] == ["01_chat.py", "02_retriever_probe.py"]
    assert [spec.title for spec in specs] == ["Chat", "Retriever 테스트"]
    assert [spec.icon for spec in specs] == ["💬", "🔎"]
    assert [spec.default for spec in specs] == [True, False]
    assert all(spec.path.is_file() for spec in specs)
