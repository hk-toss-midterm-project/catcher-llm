from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        raw_data_dir = Path("data/raw/pdf/saving_tips")
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
                raw_data_dir=raw_data_dir,
                settings=settings,
            )

        self.assertEqual(result.question, "보험료 할인 방법이 뭐야?")
        self.assertIsNone(result.error)
        self.assertEqual(len(result.contexts), 1)
        self.assertEqual(result.contexts[0].source, "data/raw/guide.md")
        self.assertEqual(result.contexts[0].content, "retrieved content")
        self.assertEqual(result.contexts[0].page_number, 3)
        retriever_factory.assert_called_once_with(
            chunk_size=600,
            chunk_overlap=60,
            top_k=3,
            raw_data_dir=raw_data_dir,
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

    def test_generate_reply_requires_anthropic_api_key_for_claude(self) -> None:
        """Claude provider를 선택하면 Anthropic API 키 누락을 별도 오류로 반환한다."""
        settings = Settings(llm_provider="claude", anthropic_api_key="")

        result = generate_reply(
            "hello",
            history=[ChatMessage(role="user", content="hi")],
            settings=settings,
        )

        self.assertEqual(result.error, "missing_anthropic_api_key")

    def test_generate_reply_allows_ollama_without_openai_api_key(self) -> None:
        """Ollama provider는 OpenAI API 키 없이도 일반 채팅 체인을 호출한다."""
        settings = Settings(llm_provider="ollama", openai_api_key="")
        chain = MagicMock()
        chain.invoke.return_value = "로컬 모델 응답"

        with patch("catcher_llm.services.chat_service.build_chat_chain", return_value=chain):
            result = generate_reply(
                "hello",
                history=[ChatMessage(role="user", content="hi")],
                settings=settings,
            )

        self.assertEqual(result.reply, "로컬 모델 응답")
        self.assertIsNone(result.error)
        chain.invoke.assert_called_once()

    def test_generate_reply_passes_chat_temperature_to_chain(self) -> None:
        """일반 채팅 응답 생성 시 호출 옵션 temperature가 채팅 체인에 전달된다."""
        settings = Settings(openai_api_key="test-key")
        chain = MagicMock()
        chain.invoke.return_value = "응답"

        with patch(
            "catcher_llm.services.chat_service.build_chat_chain", return_value=chain
        ) as build:
            result = generate_reply("hello", settings=settings, chat_temperature=0.65)

        self.assertEqual(result.reply, "응답")
        build.assert_called_once_with(settings, temperature=0.65)

    def test_generate_reply_uses_sqlite_session_chain_when_session_id_is_given(self) -> None:
        """세션 ID가 있으면 LangChain 세션 체인을 session_id 설정과 함께 호출한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(
                openai_api_key="test-key",
                sqlite_db_path=Path(tmp_dir) / "sqlite" / "app.sqlite3",
            )
            chain = MagicMock()
            chain.invoke.return_value = "저장된 세션 응답"

            with patch(
                "catcher_llm.services.chat_service.build_session_chat_chain",
                return_value=chain,
            ) as build:
                result = generate_reply(
                    "hello",
                    settings=settings,
                    session_id="user:1:chat",
                    chat_temperature=0.65,
                )

        self.assertEqual(result.reply, "저장된 세션 응답")
        build.assert_called_once_with(settings, temperature=0.65)
        chain.invoke.assert_called_once_with(
            {"input": "hello"},
            config={"configurable": {"session_id": "user:1:chat"}},
        )

    def test_generate_reply_passes_rag_temperature_to_rag_service(self) -> None:
        """RAG 라우팅 시 호출 옵션 temperature가 RAG 서비스에 전달된다."""
        settings = Settings(openai_api_key="test-key")

        with patch("catcher_llm.services.chat_service.generate_rag_reply") as generate_rag:
            generate_rag.return_value.answer = "RAG 응답"
            generate_rag.return_value.sources = []
            generate_rag.return_value.error = None

            result = generate_reply("문서에서 찾아줘", settings=settings, rag_temperature=0.1)

        self.assertEqual(result.reply, "RAG 응답")
        generate_rag.assert_called_once()
        self.assertEqual(generate_rag.call_args.kwargs["temperature"], 0.1)

    def test_generate_rag_reply_handles_missing_api_key(self) -> None:
        result = generate_rag_reply("search docs", settings=Settings(openai_api_key=""))

        self.assertEqual(result.error, "missing_openai_api_key")

    def test_generate_rag_reply_allows_ollama_chat_and_embeddings(self) -> None:
        """Ollama 채팅과 임베딩 provider를 쓰면 OpenAI 키 없이도 RAG 생성을 시도한다."""
        settings = Settings(
            llm_provider="ollama",
            embedding_provider="ollama",
            openai_api_key="",
        )
        retriever_documents = [
            Document(
                page_content="retrieved content",
                metadata={"source": "data/raw/guide.pdf", "page": 2},
            )
        ]
        chain = MagicMock()
        chain.invoke.return_value = "정답"

        with (
            patch("catcher_llm.services.rag_service.get_local_retriever") as retriever_factory,
            patch("catcher_llm.services.rag_service.build_rag_chain", return_value=chain),
        ):
            retriever_factory.return_value.invoke.return_value = retriever_documents

            result = generate_rag_reply("검색 테스트", settings=settings)

        self.assertEqual(result.answer, "정답")
        self.assertIsNone(result.error)
        chain.invoke.assert_called_once()

    def test_generate_rag_reply_passes_temperature_to_rag_chain(self) -> None:
        """RAG 답변 생성 시 호출 옵션 temperature가 RAG 체인에 전달된다."""
        settings = Settings(openai_api_key="test-key")
        retriever_documents = [
            Document(
                page_content="retrieved content",
                metadata={"source": "data/raw/guide.pdf", "page": 2},
            )
        ]
        chain = MagicMock()
        chain.invoke.return_value = "정답"

        with (
            patch("catcher_llm.services.rag_service.get_local_retriever") as retriever_factory,
            patch("catcher_llm.services.rag_service.build_rag_chain", return_value=chain) as build,
        ):
            retriever_factory.return_value.invoke.return_value = retriever_documents

            result = generate_rag_reply("검색 테스트", settings=settings, temperature=0.05)

        self.assertEqual(result.answer, "정답")
        build.assert_called_once_with(settings, temperature=0.05)

    def test_generate_rag_reply_serializes_page_numbers_in_context(self) -> None:
        settings = Settings(openai_api_key="test-key")
        retriever_documents = [
            Document(
                page_content="retrieved content",
                metadata={"source": "data/raw/guide.pdf", "page": 2},
            )
        ]
        chain = MagicMock()
        chain.invoke.return_value = "정답"

        with (
            patch("catcher_llm.services.rag_service.get_local_retriever") as retriever_factory,
            patch("catcher_llm.services.rag_service.build_rag_chain", return_value=chain),
        ):
            retriever_factory.return_value.invoke.return_value = retriever_documents

            result = generate_rag_reply("검색 테스트", settings=settings)

        self.assertEqual(result.answer, "정답")
        self.assertIsNone(result.error)
        self.assertEqual(result.contexts[0].page_number, 3)
        chain.invoke.assert_called_once()
        payload = chain.invoke.call_args.args[0]
        self.assertIn("[1] data/raw/guide.pdf (page 3)\nretrieved content", payload["context"])

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
                source_files=[pdf_path],
            )


if __name__ == "__main__":
    unittest.main()
