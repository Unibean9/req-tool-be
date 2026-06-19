import pytest

from app.models.llm_provider import ProviderType
from app.services.llm_clients import (
    AnthropicLLMClient,
    BedrockLLMClient,
    DEFAULT_MODEL_BY_PROVIDER,
    GoogleLLMClient,
    LLMClientFactory,
    OpenAILLMClient,
    _extract_bedrock_text,
)


@pytest.mark.parametrize(
    ("provider_type", "client_class"),
    [
        (ProviderType.BEDROCK, BedrockLLMClient),
        (ProviderType.OPENAI, OpenAILLMClient),
        (ProviderType.GOOGLE, GoogleLLMClient),
        (ProviderType.ANTHROPIC, AnthropicLLMClient),
    ],
)
def test_llm_client_factory_supports_current_provider_types(provider_type, client_class):
    client = LLMClientFactory.create(provider_type=provider_type, api_key="key-test")

    assert isinstance(client, client_class)
    assert client.config.api_key == "key-test"
    assert client.config.model == DEFAULT_MODEL_BY_PROVIDER[provider_type]


def test_llm_client_factory_keeps_provider_specific_credentials():
    client = LLMClientFactory.create(
        provider_type=ProviderType.BEDROCK,
        api_key="AKIATEST",
        secret_key="secret-test",
        region="ap-southeast-1",
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    assert isinstance(client, BedrockLLMClient)
    assert client.config.api_key == "AKIATEST"
    assert client.config.secret_key == "secret-test"
    assert client.config.region == "ap-southeast-1"
    assert client.config.model == "anthropic.claude-3-haiku-20240307-v1:0"


def test_extract_bedrock_text_returns_first_content_text():
    data = {"output": {"message": {"content": [{"text": "pong"}]}}}

    assert _extract_bedrock_text(data) == "pong"
