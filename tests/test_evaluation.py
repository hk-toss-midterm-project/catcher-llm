from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catcher_llm.evaluation.dataset import load_eval_examples
from catcher_llm.evaluation.evaluators import (
    answer_contains_reference,
    has_retrieved_sources,
    normalize_text,
    retrieved_context_supports_reference,
)


class EvaluationTests(unittest.TestCase):
    def test_normalize_text_collapses_case_and_spacing(self) -> None:
        self.assertEqual(normalize_text(" Streamlit,\nUI "), "streamlit ui")

    def test_load_eval_examples_reads_langsmith_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "examples.json"
            path.write_text(
                '[{"inputs": {"question": "q"}, "outputs": {"answer": "a"}}]',
                encoding="utf-8",
            )

            records = load_eval_examples(path)

            self.assertEqual(records[0]["inputs"]["question"], "q")

    def test_retrieval_evaluators_score_supporting_context(self) -> None:
        outputs = {
            "answer": "The UI uses Streamlit.",
            "sources": ["data/raw/project_overview.md"],
            "contexts": [
                {
                    "source": "data/raw/project_overview.md",
                    "content": "This project uses Streamlit for the user interface.",
                }
            ],
        }
        references = {"answer": "Streamlit"}

        self.assertTrue(answer_contains_reference({}, outputs, references).score)
        self.assertTrue(retrieved_context_supports_reference({}, outputs, references).score)
        self.assertTrue(has_retrieved_sources({}, outputs).score)


if __name__ == "__main__":
    unittest.main()
