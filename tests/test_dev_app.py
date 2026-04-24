from __future__ import annotations

from catcher_llm.ui.dev_navigation import get_dev_page_specs, get_repo_root


def test_get_repo_root_points_to_project_root() -> None:
    root = get_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "dev_pages").exists()


def test_get_dev_page_specs_registers_retriever_probe_page() -> None:
    specs = get_dev_page_specs()

    assert len(specs) == 1

    spec = specs[0]
    assert spec.title == "Retriever 테스트"
    assert spec.icon == "🔎"
    assert spec.default is True
    assert spec.path == get_repo_root() / "dev_pages" / "retriever_probe.py"
    assert spec.path.is_file()
