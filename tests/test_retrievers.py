from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings
from catcher_llm.retrievers.loaders import (
    iter_source_files,
    load_local_documents,
    load_split_local_documents,
)
from catcher_llm.retrievers.retriever import load_retrieval_seed
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

    def test_load_local_documents_uses_opendataloader_pdf(self) -> None:
        """PDF 원본을 opendataloader-pdf 마크다운 출력으로 페이지별 로드하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "guide.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            def fake_convert(
                input_path: str,
                *,
                output_dir: str | None = None,
                format: str | None = None,
                quiet: bool = False,
                markdown_page_separator: str | None = None,
                image_output: str | None = None,
            ) -> None:
                """opendataloader-pdf 변환 결과 파일을 테스트용으로 생성한다."""
                self.assertEqual(input_path, str(pdf_path))
                self.assertEqual(format, "markdown")
                self.assertTrue(quiet)
                self.assertEqual(image_output, "off")
                self.assertIsNotNone(output_dir)
                self.assertIsNotNone(markdown_page_separator)
                first_separator = markdown_page_separator.replace("%page-number%", "1")
                second_separator = markdown_page_separator.replace("%page-number%", "2")
                Path(output_dir).joinpath("guide.md").write_text(
                    f"{first_separator}첫 페이지 내용{second_separator}둘째 페이지 내용",
                    encoding="utf-8",
                )

            with patch("catcher_llm.retrievers.loaders.opendataloader_pdf.convert") as convert:
                convert.side_effect = fake_convert
                documents = load_local_documents(root)

            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].page_content, "첫 페이지 내용")
            self.assertEqual(documents[0].metadata["page"], 0)
            self.assertEqual(documents[0].metadata["page_number"], 1)
            self.assertEqual(documents[0].metadata["source"], str(pdf_path))
            self.assertEqual(documents[1].page_content, "둘째 페이지 내용")
            self.assertEqual(documents[1].metadata["page"], 1)
            self.assertEqual(documents[1].metadata["page_number"], 2)
            self.assertEqual(documents[1].metadata["source"], str(pdf_path))
            convert.assert_called_once_with(
                str(pdf_path),
                output_dir=ANY,
                format="markdown",
                quiet=True,
                markdown_page_separator=ANY,
                image_output="off",
            )

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

    def test_load_retrieval_seed_uses_explicit_chunk_settings(self) -> None:
        settings = Settings()

        with patch(
            "catcher_llm.retrievers.retriever.load_split_local_documents",
            return_value=[],
        ) as load_documents:
            result = load_retrieval_seed(
                chunk_size=640,
                chunk_overlap=80,
                settings=settings,
            )

        self.assertEqual(result, [])
        load_documents.assert_called_once_with(
            settings.raw_data_dir,
            chunk_size=640,
            chunk_overlap=80,
        )

    def test_build_local_vectorstore_saves_index_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            raw_dir = raw_root / "pdf" / "saving_tips"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir(parents=True)
            vectorstore_dir.mkdir()
            source_path = raw_dir / "guide.txt"
            source_path.write_text("project guide", encoding="utf-8")
            settings = Settings(raw_data_dir=raw_root, vectorstore_dir=vectorstore_dir)
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
            vectorstore.save_local.assert_called_once_with(
                str(vectorstore_dir / "pdf" / "saving_tips")
            )

            metadata_path = vectorstore_dir / "pdf" / "saving_tips" / "metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["embedding_model"], settings.embedding_model)
            self.assertEqual(metadata["chunk_size"], 100)
            self.assertEqual(metadata["chunk_overlap"], 10)
            self.assertEqual(metadata["raw_data_dir"], str(raw_dir.resolve()))

    def test_build_local_vectorstore_loads_saved_index_when_metadata_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            raw_dir = raw_root / "pdf" / "saving_tips"
            vectorstore_dir = root / "vectordb"
            store_dir = vectorstore_dir / "pdf" / "saving_tips"
            raw_dir.mkdir(parents=True)
            store_dir.mkdir(parents=True)
            source_path = raw_dir / "guide.txt"
            source_path.write_text("project guide", encoding="utf-8")
            settings = Settings(raw_data_dir=raw_root, vectorstore_dir=vectorstore_dir)
            metadata_path = store_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "raw_data_dir": str(raw_dir.resolve()),
                        "embedding_provider": settings.embedding_model_provider,
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
                allow_dangerous_deserialization=True,
            )
            from_documents.assert_not_called()
            load_chunks.assert_not_called()

    def test_build_local_vectorstore_saves_selection_specific_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            raw_dir = raw_root / "pdf" / "welfare"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir(parents=True)
            vectorstore_dir.mkdir()
            source_path = raw_dir / "guide.txt"
            other_path = raw_dir / "other.txt"
            source_path.write_text("project guide", encoding="utf-8")
            other_path.write_text("other guide", encoding="utf-8")
            settings = Settings(raw_data_dir=raw_root, vectorstore_dir=vectorstore_dir)
            chunks = [Document(page_content="chunk body", metadata={"source": str(source_path)})]
            embeddings = object()
            vectorstore = MagicMock()

            with (
                patch(
                    "catcher_llm.retrievers.vectorstore.load_split_local_documents",
                    return_value=chunks,
                ) as load_chunks,
                patch(
                    "catcher_llm.retrievers.vectorstore.get_embeddings_model",
                    return_value=embeddings,
                ),
                patch(
                    "catcher_llm.retrievers.vectorstore.FAISS.from_documents",
                    return_value=vectorstore,
                ),
            ):
                result = build_local_vectorstore(
                    raw_root,
                    chunk_size=100,
                    chunk_overlap=10,
                    source_files=[source_path],
                    settings=settings,
                )

            self.assertIs(result, vectorstore)
            load_chunks.assert_called_once_with(
                raw_root,
                chunk_size=100,
                chunk_overlap=10,
                source_files=[source_path],
            )
            vectorstore.save_local.assert_called_once_with(
                str(vectorstore_dir / "_selected" / "pdf" / "welfare" / "guide")
            )

            metadata_path = (
                vectorstore_dir / "_selected" / "pdf" / "welfare" / "guide" / "metadata.json"
            )
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["raw_data_dir"], str(raw_root.resolve()))
            self.assertEqual(metadata["source_files"], [str(source_path.resolve())])
            self.assertEqual(len(metadata["file_signature"]), 1)
            self.assertEqual(metadata["file_signature"][0][0], str(source_path))


if __name__ == "__main__":
    unittest.main()
