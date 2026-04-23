from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings
from catcher_llm.retrievers.loaders import (
    iter_source_files,
    load_local_documents,
    load_split_local_documents,
)
from catcher_llm.retrievers.vectorstore import (
    _VECTORSTORE_CACHE,
    build_local_vectorstore,
)


class RetrieverTests(unittest.TestCase):
    def tearDown(self) -> None:
        _VECTORSTORE_CACHE.clear()

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

    def test_build_local_vectorstore_saves_index_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir()
            vectorstore_dir.mkdir()
            source_path = raw_dir / "guide.txt"
            source_path.write_text("project guide", encoding="utf-8")
            settings = Settings(raw_data_dir=raw_dir, vectorstore_dir=vectorstore_dir)
            chunks = [Document(page_content="chunk body", metadata={"source": str(source_path)})]
            embeddings = object()
            vectorstore = MagicMock()

            with (
                patch(
                    "catcher_llm.retrievers.vectorstore.load_split_local_documents",
                    return_value=chunks,
                ),
                patch(
                    "catcher_llm.retrievers.vectorstore.get_embeddings_model",
                    return_value=embeddings,
                ),
                patch(
                    "catcher_llm.retrievers.vectorstore.FAISS.from_documents",
                    return_value=vectorstore,
                ) as from_documents,
            ):
                result = build_local_vectorstore(
                    raw_dir,
                    chunk_size=100,
                    chunk_overlap=10,
                    settings=settings,
                )

            self.assertIs(result, vectorstore)
            from_documents.assert_called_once_with(chunks, embedding=embeddings)
            vectorstore.save_local.assert_called_once()

            metadata_path = vectorstore_dir / "local_faiss_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["embedding_model"], settings.embedding_model)
            self.assertEqual(metadata["chunk_size"], 100)
            self.assertEqual(metadata["chunk_overlap"], 10)
            self.assertEqual(metadata["raw_data_dir"], str(raw_dir.resolve()))

    def test_build_local_vectorstore_loads_saved_index_when_metadata_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            vectorstore_dir = root / "vectordb"
            store_dir = vectorstore_dir / "local_faiss"
            raw_dir.mkdir()
            store_dir.mkdir(parents=True)
            source_path = raw_dir / "guide.txt"
            source_path.write_text("project guide", encoding="utf-8")
            settings = Settings(raw_data_dir=raw_dir, vectorstore_dir=vectorstore_dir)
            metadata_path = vectorstore_dir / "local_faiss_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "raw_data_dir": str(raw_dir.resolve()),
                        "embedding_model": settings.embedding_model,
                        "chunk_size": 100,
                        "chunk_overlap": 10,
                        "file_signature": [
                            [
                                str(source_path),
                                source_path.stat().st_mtime_ns,
                                source_path.stat().st_size,
                            ]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (store_dir / "index.faiss").write_bytes(b"faiss")
            (store_dir / "index.pkl").write_bytes(b"pickle")
            embeddings = object()
            vectorstore = MagicMock()

            with (
                patch(
                    "catcher_llm.retrievers.vectorstore.get_embeddings_model",
                    return_value=embeddings,
                ),
                patch(
                    "catcher_llm.retrievers.vectorstore.FAISS.load_local",
                    return_value=vectorstore,
                ) as load_local,
                patch("catcher_llm.retrievers.vectorstore.FAISS.from_documents") as from_documents,
                patch(
                    "catcher_llm.retrievers.vectorstore.load_split_local_documents"
                ) as load_chunks,
            ):
                result = build_local_vectorstore(
                    raw_dir,
                    chunk_size=100,
                    chunk_overlap=10,
                    settings=settings,
                )

            self.assertIs(result, vectorstore)
            load_local.assert_called_once_with(
                str(store_dir),
                embeddings,
                index_name="index",
                allow_dangerous_deserialization=True,
            )
            from_documents.assert_not_called()
            load_chunks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
