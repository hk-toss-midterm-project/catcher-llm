from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from catcher_llm.config.settings import configure_langsmith_env, get_settings


class TestLangSmith(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()

    def test_get_settings_does_not_set_langchain_env_vars(self):
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = True
        mock_settings.langsmith_api_key = "test-key"
        mock_settings.langsmith_project = "test-project"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        with patch("catcher_llm.config.settings.Settings", return_value=mock_settings):
            with patch.dict(os.environ, {}, clear=True):
                settings = get_settings()

                self.assertIs(settings, mock_settings)
                self.assertIsNone(os.environ.get("LANGCHAIN_TRACING_V2"))
                self.assertIsNone(os.environ.get("LANGCHAIN_API_KEY"))
                self.assertIsNone(os.environ.get("LANGCHAIN_PROJECT"))

    def test_configure_langsmith_env_sets_langchain_env_vars(self):
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = True
        mock_settings.langsmith_api_key = "test-key"
        mock_settings.langsmith_project = "test-project"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        with patch.dict(os.environ, {}, clear=True):
            configure_langsmith_env(mock_settings)

            self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
            self.assertEqual(os.environ.get("LANGCHAIN_API_KEY"), "test-key")
            self.assertEqual(os.environ.get("LANGCHAIN_PROJECT"), "test-project")
            self.assertEqual(
                os.environ.get("LANGCHAIN_ENDPOINT"),
                "https://api.smith.langchain.com",
            )

    def test_configure_langsmith_env_clears_env_if_tracing_disabled(self):
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = False

        with patch.dict(
            os.environ,
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "test-key",
                "LANGCHAIN_PROJECT": "test-project",
                "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            },
            clear=True,
        ):
            configure_langsmith_env(mock_settings)

            self.assertIsNone(os.environ.get("LANGCHAIN_TRACING_V2"))
            self.assertIsNone(os.environ.get("LANGCHAIN_API_KEY"))
            self.assertIsNone(os.environ.get("LANGCHAIN_PROJECT"))
            self.assertIsNone(os.environ.get("LANGCHAIN_ENDPOINT"))


if __name__ == "__main__":
    unittest.main()
