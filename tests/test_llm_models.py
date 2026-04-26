from __future__ import annotations

import unittest
from dataclasses import fields

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model, get_embeddings_model


class LLMModelTests(unittest.TestCase):
    def test_settings_does_not_expose_global_temperature(self) -> None:
        """모델 temperature는 전역 환경 설정 필드가 아니라 호출 옵션으로 관리한다."""
        setting_field_names = {field.name for field in fields(Settings)}

        self.assertNotIn("model_temperature", setting_field_names)

    def test_settings_does_not_expose_common_chat_model_override(self) -> None:
        """공통 CHAT_MODEL override 대신 provider별 모델 설정만 노출한다."""
        setting_field_names = {field.name for field in fields(Settings)}

        self.assertNotIn("chat_model", setting_field_names)

    def test_settings_normalizes_claude_provider_alias(self) -> None:
        """Claude 별칭을 Anthropic provider로 정규화하고 Anthropic 모델명을 사용한다."""
        settings = Settings(
            llm_provider="claude",
            anthropic_model="claude-custom",
        )

        self.assertEqual(settings.chat_provider, "anthropic")
        self.assertEqual(settings.chat_model_name, "claude-custom")
        self.assertEqual(settings.chat_model_label, "anthropic:claude-custom")

    def test_get_chat_model_builds_openai_model(self) -> None:
        """OpenAI provider 설정으로 ChatOpenAI 인스턴스가 생성되는지 검증한다."""
        model = get_chat_model(
            Settings(
                llm_provider="openai",
                openai_api_key="test-openai-key",
                openai_model="gpt-test",
            )
        )

        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.model_name, "gpt-test")

    def test_get_chat_model_uses_call_temperature(self) -> None:
        """채팅 모델 temperature는 Settings가 아니라 함수 호출 인자로 지정한다."""
        model = get_chat_model(
            Settings(
                llm_provider="openai",
                openai_api_key="test-openai-key",
                openai_model="gpt-test",
            ),
            temperature=0.42,
        )

        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.temperature, 0.42)

    def test_get_chat_model_builds_anthropic_model(self) -> None:
        """Anthropic provider 설정으로 ChatAnthropic 인스턴스가 생성되는지 검증한다."""
        model = get_chat_model(
            Settings(
                llm_provider="anthropic",
                anthropic_api_key="test-anthropic-key",
                anthropic_model="claude-test",
            )
        )

        self.assertIsInstance(model, ChatAnthropic)
        self.assertEqual(model.model, "claude-test")

    def test_get_chat_model_builds_ollama_model(self) -> None:
        """Ollama provider 설정으로 ChatOllama 인스턴스가 생성되는지 검증한다."""
        model = get_chat_model(
            Settings(
                llm_provider="ollama",
                ollama_model="llama-test",
                ollama_base_url="http://localhost:11434",
            )
        )

        self.assertIsInstance(model, ChatOllama)
        self.assertEqual(model.model, "llama-test")
        self.assertEqual(model.base_url, "http://localhost:11434")

    def test_get_embeddings_model_builds_openai_embeddings(self) -> None:
        """OpenAI embedding provider 설정으로 OpenAIEmbeddings가 생성되는지 검증한다."""
        embeddings = get_embeddings_model(
            Settings(
                embedding_provider="openai",
                openai_api_key="test-openai-key",
                embedding_model="text-embedding-test",
            )
        )

        self.assertIsInstance(embeddings, OpenAIEmbeddings)
        self.assertEqual(embeddings.model, "text-embedding-test")

    def test_get_embeddings_model_builds_ollama_embeddings(self) -> None:
        """Ollama embedding provider 설정으로 OllamaEmbeddings가 생성되는지 검증한다."""
        embeddings = get_embeddings_model(
            Settings(
                embedding_provider="ollama",
                ollama_embedding_model="nomic-test",
                ollama_base_url="http://localhost:11434",
            )
        )

        self.assertIsInstance(embeddings, OllamaEmbeddings)
        self.assertEqual(embeddings.model, "nomic-test")
        self.assertEqual(embeddings.base_url, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
