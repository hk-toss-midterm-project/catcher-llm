from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

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
            (root / "guide.pdf").write_bytes(b"%PDF-1.4")
            (root / "ignore.csv").write_text("a,b", encoding="utf-8")

            files = iter_source_files(root)

            self.assertEqual([path.name for path in files], ["guide.pdf", "note.md"])

    def test_load_local_documents_reads_page_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "guide.txt").write_text("project guide", encoding="utf-8")

            documents = load_local_documents(root)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "project guide")

    def test_load_local_documents_uses_pdf_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "guide.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            with patch("catcher_llm.retrievers.loaders.PyPDFLoader") as loader_class:
                loader_class.return_value.load.return_value = [
                    Document(page_content="pdf page", metadata={"page": 0})
                ]

                documents = load_local_documents(root)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "pdf page")
            self.assertEqual(documents[0].metadata["source"], str(pdf_path))
            loader_class.assert_called_once_with(str(pdf_path))

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
