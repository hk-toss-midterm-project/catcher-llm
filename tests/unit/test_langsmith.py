from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import get_settings


class TestLangSmith(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()

    def test_get_settings_sets_langchain_env_vars(self):
        # Mock Settings object
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = True
        mock_settings.langsmith_api_key = "test-key"
        mock_settings.langsmith_project = "test-project"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        with patch("catcher_llm.config.settings.Settings", return_value=mock_settings):
            # Clear env vars that might be set
            with patch.dict(os.environ, {}, clear=True):
                get_settings()

                self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
                self.assertEqual(os.environ.get("LANGCHAIN_API_KEY"), "test-key")
                self.assertEqual(os.environ.get("LANGCHAIN_PROJECT"), "test-project")

    def test_get_settings_does_not_set_if_tracing_disabled(self):
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = False

        with patch("catcher_llm.config.settings.Settings", return_value=mock_settings):
            with patch.dict(os.environ, {}, clear=True):
                get_settings()

                self.assertIsNone(os.environ.get("LANGCHAIN_TRACING_V2"))


if __name__ == "__main__":
    unittest.main()
