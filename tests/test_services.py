from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.services.chat_service import generate_reply
from catcher_llm.services.ingestion_service import ingest_local_documents, ingest_selected_documents
from catcher_llm.services.rag_service import generate_rag_reply, rag_target
from catcher_llm.services.test_service import invoke_retriever_question

logger = logging.getLogger(__name__)


class ServiceTests(unittest.TestCase):
    def test_invoke_retriever_question_returns_serialized_matches(self) -> None:
        settings = Settings()
        retriever_documents = [
            Document(
                page_content="retrieved content",
                metadata={"source": "data/raw/guide.md", "page": 2, "chunk_index": 5},
            )
        ]

        with patch("catcher_llm.services.test_service.get_local_retriever") as retriever_factory:
            retriever_factory.return_value.invoke.return_value = retriever_documents

            result = invoke_retriever_question(
                "보험료 할인 방법이 뭐야?",
                chunk_size=600,
                chunk_overlap=60,
                top_k=3,
                settings=settings,
            )

        self.assertEqual(result.question, "보험료 할인 방법이 뭐야?")
        self.assertIsNone(result.error)
        self.assertEqual(len(result.contexts), 1)
        self.assertEqual(result.contexts[0].source, "data/raw/guide.md")
        self.assertEqual(result.contexts[0].content, "retrieved content")
        retriever_factory.assert_called_once_with(
            chunk_size=600,
            chunk_overlap=60,
            top_k=3,
            settings=settings,
        )

    def test_invoke_retriever_question_handles_missing_documents(self) -> None:
        with patch("catcher_llm.services.test_service.get_local_retriever", return_value=None):
            result = invoke_retriever_question("검색 테스트", settings=Settings())

        self.assertEqual(result.question, "검색 테스트")
        self.assertEqual(result.contexts, [])
        self.assertEqual(result.error, "missing_documents")

    def test_generate_reply_handles_missing_api_key(self) -> None:
        settings = Settings(openai_api_key="")

        result = generate_reply(
            "hello",
            history=[ChatMessage(role="user", content="hi")],
            settings=settings,
        )

        self.assertEqual(result.error, "missing_openai_api_key")

    def test_generate_rag_reply_handles_missing_api_key(self) -> None:
        result = generate_rag_reply("search docs", settings=Settings(openai_api_key=""))

        self.assertEqual(result.error, "missing_openai_api_key")

    def test_rag_target_serializes_error_state(self) -> None:
        result = rag_target({"question": "search docs"}, settings=Settings(openai_api_key=""))

        self.assertEqual(result["error"], "missing_openai_api_key")
        self.assertIn("answer", result)

    def test_ingest_local_documents_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir()
            processed_dir.mkdir()
            vectorstore_dir.mkdir()
            (raw_dir / "source.md").write_text("hello world", encoding="utf-8")

            settings = Settings(
                data_dir=root,
                raw_data_dir=raw_dir,
                processed_data_dir=processed_dir,
                vectorstore_dir=vectorstore_dir,
            )

            with patch(
                "catcher_llm.services.ingestion_service.build_local_vectorstore"
            ) as build_store:
                result = ingest_local_documents(settings=settings)

            self.assertEqual(result["documents"], 1)
            self.assertGreaterEqual(int(result["chunks"]), 1)
            self.assertTrue((processed_dir / "ingestion_manifest.json").exists())
            build_store.assert_called_once_with(
                raw_dir,
                chunk_size=800,
                chunk_overlap=120,
                settings=settings,
            )

    def test_ingest_local_documents_counts_pdf_as_single_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir()
            processed_dir.mkdir()
            vectorstore_dir.mkdir()
            pdf_path = raw_dir / "guide.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            settings = Settings(
                data_dir=root,
                raw_data_dir=raw_dir,
                processed_data_dir=processed_dir,
                vectorstore_dir=vectorstore_dir,
            )

            with patch("catcher_llm.retrievers.loaders.PyPDFLoader") as loader_class:
                loader_class.return_value.load.return_value = [
                    Document(page_content="page one", metadata={"page": 0}),
                    Document(page_content="page two", metadata={"page": 1}),
                ]

                with patch(
                    "catcher_llm.services.ingestion_service.build_local_vectorstore"
                ) as build_store:
                    result = ingest_local_documents(settings=settings)

            self.assertEqual(result["documents"], 1)
            self.assertGreaterEqual(int(result["chunks"]), 1)
            self.assertTrue((processed_dir / "ingestion_manifest.json").exists())
            build_store.assert_called_once_with(
                raw_dir,
                chunk_size=800,
                chunk_overlap=120,
                settings=settings,
            )

    def test_ingest_selected_documents_logs_execution_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            vectorstore_dir = root / "vectordb"
            raw_dir.mkdir()
            processed_dir.mkdir()
            vectorstore_dir.mkdir()
            pdf_path = raw_dir / "guide.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            settings = Settings(
                data_dir=root,
                raw_data_dir=raw_dir,
                processed_data_dir=processed_dir,
                vectorstore_dir=vectorstore_dir,
            )

            with patch("catcher_llm.retrievers.loaders.PyPDFLoader") as loader_class:
                loader_class.return_value.load.return_value = [
                    Document(page_content="page one", metadata={"page": 0}),
                    Document(page_content="page two", metadata={"page": 1}),
                ]

                logger.info("calling ingest_selected_documents with source=%s", pdf_path)
                with patch(
                    "catcher_llm.services.ingestion_service.build_local_vectorstore"
                ) as build_store:
                    result = ingest_selected_documents(
                        [pdf_path],
                        settings=settings,
                        manifest_name="selected_manifest.json",
                    )
                logger.info("ingest_selected_documents returned result=%s", result)

            manifest_path = processed_dir / "selected_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            logger.info("ingestion manifest=%s", manifest)

            self.assertEqual(result["documents"], 1)
            self.assertGreaterEqual(int(result["chunks"]), 1)
            self.assertEqual(result["manifest_path"], str(manifest_path))
            self.assertEqual(manifest[0]["source"], str(pdf_path))
            build_store.assert_called_once_with(
                raw_dir,
                chunk_size=800,
                chunk_overlap=120,
                settings=settings,
            )


if __name__ == "__main__":
    unittest.main()
