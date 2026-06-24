import httpx
import pytest

from app.services.llm_clients import (
    AnthropicLLMClient,
    BedrockLLMClient,
    GoogleLLMClient,
    LLMClientConfig,
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
}

# Payload without usage for each provider
_PAYLOAD_NO_USAGE = {
    OpenAILLMClient: {"output_text": "raw answer"},
    AnthropicLLMClient: {"content": [{"type": "text", "text": "raw answer"}]},
    GoogleLLMClient: {"candidates": [{"content": {"parts": [{"text": "raw answer"}]}}]},
    BedrockLLMClient: {"output": {"message": {"content": [{"text": "raw answer"}]}}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_PAYLOAD_WITH_USAGE))
async def test_generate_returns_text_and_normalized_usage(monkeypatch, client_class):
    _install_httpx_recorder(monkeypatch, _PAYLOAD_WITH_USAGE[client_class])
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

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
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

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

    monkeypatch.setattr(httpx, "AsyncClient", client_class)


class _RecordingAsyncClient:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, **kwargs):
        return httpx.Response(200, json=self.payload, request=httpx.Request("POST", url))
