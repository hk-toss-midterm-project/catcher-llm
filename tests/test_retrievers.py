from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catcher_llm.retrievers.loaders import (
    iter_source_files,
    load_local_documents,
    load_split_local_documents,
)


class RetrieverTests(unittest.TestCase):
    def test_iter_source_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "note.md").write_text("hello", encoding="utf-8")
            (root / "ignore.csv").write_text("a,b", encoding="utf-8")

            files = iter_source_files(root)

            self.assertEqual([path.name for path in files], ["note.md"])

    def test_load_local_documents_reads_page_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "guide.txt").write_text("project guide", encoding="utf-8")

            documents = load_local_documents(root)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "project guide")

    def test_load_split_local_documents_adds_chunk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            text = " ".join(["chunk"] * 400)
            (root / "guide.txt").write_text(text, encoding="utf-8")

            documents = load_split_local_documents(
                root,
                chunk_size=100,
                chunk_overlap=10,
            )

            self.assertGreater(len(documents), 1)
            self.assertIn("chunk_index", documents[0].metadata)


if __name__ == "__main__":
    unittest.main()
