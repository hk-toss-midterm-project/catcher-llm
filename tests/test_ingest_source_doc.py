from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ingest_source_doc.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("ingest_source_doc_module", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
ingest_source_doc = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(ingest_source_doc)


class IngestSourceDocScriptTests(unittest.TestCase):
    def test_main_ingests_selected_pdf_and_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "guide.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            resolved_path = pdf_path.resolve()
            settings = object()
            stdout = io.StringIO()

            with (
                patch("sys.argv", ["ingest_source_doc.py", str(pdf_path)]),
                patch.object(ingest_source_doc, "get_settings", return_value=settings),
                patch.object(
                    ingest_source_doc,
                    "ingest_selected_documents",
                    return_value={
                        "documents": 1,
                        "chunks": 3,
                        "manifest_path": "/tmp/guide_ingestion_manifest.json",
                    },
                ) as ingest_mock,
                redirect_stdout(stdout),
            ):
                ingest_source_doc.main()

            ingest_mock.assert_called_once_with(
                [resolved_path],
                settings=settings,
                manifest_name="guide_ingestion_manifest.json",
            )
            output = stdout.getvalue()
            self.assertIn("Indexed 1 file", output)
            self.assertIn("Chunks: 3", output)
            self.assertIn("Manifest: /tmp/guide_ingestion_manifest.json", output)

    def test_main_raises_when_source_document_is_missing(self) -> None:
        missing_path = "/tmp/does-not-exist.pdf"

        with patch("sys.argv", ["ingest_source_doc.py", missing_path]):
            with self.assertRaises(FileNotFoundError):
                ingest_source_doc.main()


if __name__ == "__main__":
    unittest.main()
