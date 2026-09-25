"""Per-call LLM deadline: every provider's generate() honours the configured call timeout, and one
hard asyncio deadline bounds the call whatever the SDK does (per-socket timeouts, own retries)."""

import asyncio

import httpx
import pytest

from app.services import llm_clients as llm_client_module
from app.services.llm_clients import (
    AnthropicLLMClient,
    BedrockLLMClient,
    CustomLLMClient,
    DeadlineLLMClient,
    GoogleLLMClient,
    LLMCallTimeoutError,
    LLMClientConfig,
    MistralLLMClient,
    OpenAILLMClient,
)

_TOOL = {"name": "respond", "description": "Reply.", "parameters": {"type": "object", "properties": {}}}


class _FakeHttpx:
    """Stands in for httpx.AsyncClient; records the timeout it was built with."""

    seen: list[float] = []

    def __init__(self, *args, timeout=None, **kwargs):
        _FakeHttpx.seen.append(timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        payload = {"choices": [{"message": {"content": "ok"}}], "output": {"message": {"content": [{"text": "ok"}]}}}
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


@pytest.fixture
def seen_timeouts(monkeypatch):
    seen: list[float] = []
    _FakeHttpx.seen = seen

    async def _record(api_key, timeout, body, **kwargs):
        seen.append(timeout)
        return {}

    monkeypatch.setattr(llm_client_module, "_openai_create_response", _record)
    monkeypatch.setattr(llm_client_module, "_anthropic_create_message", _record)
    monkeypatch.setattr(llm_client_module, "_google_generate_content", _record)
    monkeypatch.setattr(llm_client_module, "_mistral_create_chat_completion", _record)
    monkeypatch.setattr(llm_client_module, "_create_mistral_sdk", lambda **_kwargs: object())
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpx)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class",
    [OpenAILLMClient, AnthropicLLMClient, GoogleLLMClient, MistralLLMClient, CustomLLMClient, BedrockLLMClient],
)
@pytest.mark.parametrize("with_tools", [True, False])
async def test_every_provider_generate_uses_the_configured_call_timeout(seen_timeouts, client_class, with_tools):
    """A drafting call writing a large section takes ~50s; a hard-coded 30s SDK timeout would cut it."""
    config = LLMClientConfig(api_key="k", model="m", base_url="https://custom.example/v1", request_timeout=123.0)
    client = client_class(config)

    await client.generate(
        messages=[{"role": "user", "content": "hi"}],
        system="s",
        max_tokens=10,
        tools=[_TOOL] if with_tools else None,
    )

    assert seen_timeouts == [123.0]


class _SlowClient:
    marker = "inner-attribute"

    async def generate(self, **kwargs):
        await asyncio.sleep(5)


class _FastClient:
    async def generate(self, **kwargs):
        return "done", {"input": 1, "output": 1, "total": 2}


@pytest.mark.asyncio
async def test_deadline_client_stops_a_hung_call_with_a_call_timeout_error():
    client = DeadlineLLMClient(_SlowClient(), 0.01)

    with pytest.raises(LLMCallTimeoutError) as excinfo:
        await client.generate(messages=[], system=None, max_tokens=1)

    # Still a TimeoutError, so the turn-level handler keeps treating it as a resumable timeout.
    assert isinstance(excinfo.value, TimeoutError)
    assert excinfo.value.seconds == 0.01


@pytest.mark.asyncio
async def test_deadline_client_passes_results_and_attributes_through():
    client = DeadlineLLMClient(_FastClient(), 1.0)

    assert await client.generate(messages=[], system=None, max_tokens=1) == (
        "done",
        {"input": 1, "output": 1, "total": 2},
    )
    assert DeadlineLLMClient(_SlowClient(), 1.0).marker == "inner-attribute"
