from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.chat import ChatMessage
from catcher_llm.services.chat_service import generate_reply
from catcher_llm.services.ingestion_service import ingest_local_documents
from catcher_llm.services.rag_service import generate_rag_reply, rag_target


class ServiceTests(unittest.TestCase):
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

            result = ingest_local_documents(settings)

            self.assertEqual(result["documents"], 1)
            self.assertGreaterEqual(int(result["chunks"]), 1)
            self.assertTrue((processed_dir / "ingestion_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
