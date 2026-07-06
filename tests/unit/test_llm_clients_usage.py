import httpx
import pytest

from app.services import llm_clients as llm_client_module
from app.services.llm_clients import (
    AnthropicLLMClient,
    BedrockLLMClient,
    CustomLLMClient,
    GoogleLLMClient,
    LLMClientConfig,
    MistralLLMClient,
    OpenAILLMClient,
)

# Payload with usage for each provider — all normalize to {"input": 11, "output": 22, "total": 33}
_PAYLOAD_WITH_USAGE = {
    OpenAILLMClient: {
        "output_text": "raw answer",
        "usage": {"input_tokens": 11, "output_tokens": 22, "total_tokens": 33},
    },
    AnthropicLLMClient: {
        "content": [{"type": "text", "text": "raw answer"}],
        "usage": {"input_tokens": 11, "output_tokens": 22},  # no total → computed
    },
    GoogleLLMClient: {
        "candidates": [{"content": {"parts": [{"text": "raw answer"}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 22, "totalTokenCount": 33},
    },
    BedrockLLMClient: {
        "output": {"message": {"content": [{"text": "raw answer"}]}},
        "usage": {"inputTokens": 11, "outputTokens": 22, "totalTokens": 33},
    },
    CustomLLMClient: {
        "choices": [{"message": {"content": "raw answer"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    },
    MistralLLMClient: {
        "choices": [{"message": {"content": "raw answer"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    },
}

# Payload without usage for each provider
_PAYLOAD_NO_USAGE = {
    OpenAILLMClient: {"output_text": "raw answer"},
    AnthropicLLMClient: {"content": [{"type": "text", "text": "raw answer"}]},
    GoogleLLMClient: {"candidates": [{"content": {"parts": [{"text": "raw answer"}]}}]},
    BedrockLLMClient: {"output": {"message": {"content": [{"text": "raw answer"}]}}},
    CustomLLMClient: {"choices": [{"message": {"content": "raw answer"}}]},
    MistralLLMClient: {"choices": [{"message": {"content": "raw answer"}}]},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_PAYLOAD_WITH_USAGE))
async def test_generate_returns_text_and_normalized_usage(monkeypatch, client_class):
    _install_httpx_recorder(monkeypatch, _PAYLOAD_WITH_USAGE[client_class])
    client = client_class(_client_config(client_class))

    text, usage = await client.generate(
        messages=[{"role": "user", "content": "Ping"}],
        system=None,
        max_tokens=32,
        response_format=None,
    )

    assert text == "raw answer"
    assert usage == {"input": 11, "output": 22, "total": 33}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_PAYLOAD_NO_USAGE))
async def test_generate_returns_none_usage_when_absent(monkeypatch, client_class):
    _install_httpx_recorder(monkeypatch, _PAYLOAD_NO_USAGE[client_class])
    client = client_class(_client_config(client_class))

    text, usage = await client.generate(
        messages=[{"role": "user", "content": "Ping"}],
        system=None,
        max_tokens=32,
        response_format=None,
    )

    assert text == "raw answer"
    assert usage is None


def _install_httpx_recorder(monkeypatch, payload):
    def client_class(*args, **kwargs):
        return _RecordingAsyncClient(payload)

    def openai_client(**_kwargs):
        return _RecordingOpenAIClient(payload)

    def anthropic_client(**_kwargs):
        return _RecordingAnthropicClient(payload)

    def google_client(**_kwargs):
        return _RecordingGoogleClient(payload)

    def mistral_client(**_kwargs):
        return _RecordingMistralClient(payload)

    monkeypatch.setattr(httpx, "AsyncClient", client_class)
    monkeypatch.setattr(llm_client_module, "_create_openai_sdk", openai_client)
    monkeypatch.setattr(llm_client_module, "_create_anthropic_sdk", anthropic_client)
    monkeypatch.setattr(llm_client_module, "_create_google_sdk", google_client)
    monkeypatch.setattr(llm_client_module, "_create_mistral_sdk", mistral_client)


def _client_config(client_class):
    kwargs = {"api_key": "key-test", "model": "model-test"}
    if client_class is CustomLLMClient:
        kwargs["base_url"] = "https://custom.example/v1"
    return LLMClientConfig(**kwargs)


class _RecordingAsyncClient:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, **kwargs):
        return httpx.Response(200, json=self.payload, request=httpx.Request("POST", url))


class _RecordingOpenAIClient:
    def __init__(self, payload):
        self.responses = _RecordingSDKResource(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingAnthropicClient:
    def __init__(self, payload):
        self.messages = _RecordingSDKResource(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingSDKResource:
    def __init__(self, payload):
        self.payload = payload

    async def create(self, **kwargs):
        return self.payload


class _RecordingMistralClient:
    def __init__(self, payload):
        self.chat = _RecordingMistralChat(payload)


class _RecordingMistralChat:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        return self.payload


class _RecordingGoogleClient:
    def __init__(self, payload):
        self.aio = _RecordingGoogleAsyncClient(payload)


class _RecordingGoogleAsyncClient:
    def __init__(self, payload):
        self.models = _RecordingGoogleModels(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingGoogleModels:
    def __init__(self, payload):
        self.payload = payload

    async def generate_content(self, *, model, contents, config=None):
        return self.payload
