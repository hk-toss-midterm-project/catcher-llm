from __future__ import annotations

import unittest
from unittest.mock import patch

from catcher_llm.services.rag.kca_report import generate_kca_report_rag_reply


class KcaReportRagServiceTests(unittest.TestCase):
    def test_generate_kca_report_rag_reply_uses_tighter_default_top_k(self) -> None:
        """KCA 보고서 RAG 서비스가 사실형 평가에 맞게 기본 top_k=2를 사용하는지 검증한다."""
        with (
            patch("catcher_llm.services.rag.kca_report.get_rag_pipeline_config") as get_config,
            patch("catcher_llm.services.rag.kca_report.get_kca_report_prompt") as get_prompt,
            patch("catcher_llm.services.rag.kca_report.generate_rag_reply") as generate_reply,
        ):
            get_config.return_value.raw_data_dir = "raw"
            get_config.return_value.source_files = ["doc.pdf"]
            get_prompt.return_value = object()
            generate_reply.return_value = object()

            generate_kca_report_rag_reply("질문")

        generate_reply.assert_called_once()
        self.assertEqual(generate_reply.call_args.kwargs["top_k"], 2)


if __name__ == "__main__":
    unittest.main()
