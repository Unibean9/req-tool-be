import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.models.llm_provider import ProviderType
from app.services.llm_clients import (
    DEFAULT_MODEL_BY_PROVIDER,
    AnthropicLLMClient,
    BedrockLLMClient,
    GoogleLLMClient,
    LLMClientConfig,
    LLMClientFactory,
    OpenAILLMClient,
    _extract_bedrock_text,
    _parse_generate_text,
    _responses_json_schema_format,
)

ANALYSIS_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "next_action": {"type": "string"},
    },
    "required": ["gaps", "contradictions", "risks", "confidence", "next_action"],
}


ANALYSIS_RESULT = {
    "gaps": ["Thiếu persona chính"],
    "contradictions": [],
    "risks": ["Scope chưa rõ"],
    "confidence": 0.82,
    "next_action": "ask_human",
}


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_class", "provider_payload"),
    [
        (OpenAILLMClient, {"output_text": '{"answer": "ok"}'}),
        (AnthropicLLMClient, {"content": [{"type": "text", "text": '{"answer": "ok"}'}]}),
        (
            GoogleLLMClient,
            {"candidates": [{"content": {"parts": [{"text": '{"answer": "ok"}'}]}}]},
        ),
        (BedrockLLMClient, {"output": {"message": {"content": [{"text": '{"answer": "ok"}'}]}}}),
    ],
)
async def test_generate_with_response_format_returns_dict(monkeypatch, client_class, provider_payload):
    recorder = _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Phân tích yêu cầu"}],
        system="Bạn là BA.",
        max_tokens=256,
        response_format={"name": "analysis_result", "schema": {"type": "object"}},
    )

    assert result == {"answer": "ok"}
    assert recorder.requests


@pytest.mark.asyncio
async def test_openai_generate_uses_responses_text_format_for_schema(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, {"output_text": '{"answer": "ok"}'})
    client = OpenAILLMClient(LLMClientConfig(api_key="key-test", model="model-test"))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Phân tích yêu cầu"}],
        system="Bạn là BA.",
        max_tokens=256,
        response_format={"name": "analysis_result", "schema": {"type": "object"}},
    )

    body = recorder.requests[0]["json"]
    assert result == {"answer": "ok"}
    assert "response_format" not in body
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "analysis_result",
        "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    }


def test_openai_responses_schema_makes_optional_fields_nullable_and_required():
    schema = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": ["ask_user", "write_draft"]},
            "message": {"type": "string"},
        },
        "required": ["tool"],
    }

    formatted = _responses_json_schema_format({"name": "tool_selection", "schema": schema})

    assert formatted["strict"] is True
    assert formatted["schema"]["additionalProperties"] is False
    assert formatted["schema"]["required"] == ["tool", "message"]
    assert formatted["schema"]["properties"]["message"]["type"] == ["string", "null"]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Thiếu tool"},
        {"tool": "finalize", "message": "Sai enum"},
        {"tool": "ask_user", "message": "ok", "extra": "không hợp lệ"},
    ],
)
def test_parse_generate_text_rejects_schema_invalid_structured_output(payload):
    response_format = {
        "name": "tool_selection",
        "schema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["ask_user", "write_draft"]},
                "message": {"type": "string"},
            },
            "required": ["tool", "message"],
            "additionalProperties": False,
        },
    }

    with pytest.raises(ValueError, match="không khớp JSON Schema"):
        _parse_generate_text(json.dumps(payload, ensure_ascii=False), response_format)


def test_parse_generate_text_accepts_tool_args_prompt_alias():
    payload = {"tools": [{"name": "ask_user", "args": {"prompt": "Bạn muốn phân tích phần nào?"}}]}
    schema = {"type": "object", "properties": {"tools": {"type": "array"}}, "required": ["tools"]}

    result = _parse_generate_text(json.dumps(payload, ensure_ascii=False), schema)

    assert result == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_class", "provider_payload"),
    [
        (OpenAILLMClient, {"output_text": "not json"}),
        (AnthropicLLMClient, {"content": [{"type": "text", "text": "not json"}]}),
        (GoogleLLMClient, {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}),
        (BedrockLLMClient, {"output": {"message": {"content": [{"text": "not json"}]}}}),
    ],
)
async def test_generate_with_response_format_rejects_invalid_json(monkeypatch, client_class, provider_payload):
    _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    with pytest.raises(ValueError, match="Không parse được JSON"):
        await client.generate(
            messages=[{"role": "user", "content": "Phân tích yêu cầu"}],
            system="Bạn là BA.",
            max_tokens=256,
            response_format={"name": "analysis_result", "schema": {"type": "object"}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_class", "provider_payload"),
    [
        (OpenAILLMClient, {"output_text": "raw answer"}),
        (AnthropicLLMClient, {"content": [{"type": "text", "text": "raw answer"}]}),
        (GoogleLLMClient, {"candidates": [{"content": {"parts": [{"text": "raw answer"}]}}]}),
        (BedrockLLMClient, {"output": {"message": {"content": [{"text": "raw answer"}]}}}),
    ],
)
async def test_generate_without_response_format_returns_raw_text_and_no_extra_params(monkeypatch, client_class, provider_payload):
    recorder = _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Ping"}],
        system=None,
        max_tokens=32,
        response_format=None,
    )

    body = recorder.requests[0]["json"]
    assert result == "raw answer"
    assert "response_format" not in body
    assert "responseMimeType" not in str(body)
    assert "responseSchema" not in str(body)
    assert "JSON Schema" not in str(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [AnthropicLLMClient, BedrockLLMClient])
async def test_generate_injects_schema_for_prompt_based_providers(monkeypatch, client_class):
    recorder = _install_httpx_recorder(
        monkeypatch,
        {"content": [{"text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}]},
    )
    if client_class is BedrockLLMClient:
        recorder = _install_httpx_recorder(
            monkeypatch,
            {"output": {"message": {"content": [{"text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}]}}},
        )
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    await client.generate(
        messages=[{"role": "user", "content": "Phân tích"}],
        system="Bạn là BA.",
        max_tokens=256,
        response_format={"name": "analysis_result", "schema": ANALYSIS_RESULT_SCHEMA},
    )

    body = recorder.requests[0]["json"]
    assert "JSON Schema" in str(body)
    assert "analysis_result" in str(body)
    assert "next_action" in str(body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_class", "provider_payload"),
    [
        (OpenAILLMClient, {"output_text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}),
        (AnthropicLLMClient, {"content": [{"type": "text", "text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}]}),
        (
            GoogleLLMClient,
            {"candidates": [{"content": {"parts": [{"text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}]}}]},
        ),
        (BedrockLLMClient, {"output": {"message": {"content": [{"text": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}]}}}),
    ],
)
async def test_generate_parses_analysis_result_schema_for_each_provider(monkeypatch, client_class, provider_payload):
    _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Phân tích"}],
        system="Bạn là BA.",
        max_tokens=512,
        response_format={"name": "analysis_result", "schema": ANALYSIS_RESULT_SCHEMA},
    )

    assert result == ANALYSIS_RESULT


def _install_httpx_recorder(monkeypatch, payload):
    recorder = _HttpxRecorder(payload)
    monkeypatch.setattr(httpx, "AsyncClient", recorder.client_class)
    return recorder


# ---------------------------------------------------------------------------
# Native tool calling — generate(tools=...) returns an AIMessage(tool_calls=[...])
# Deterministic parser coverage; per-provider LIVE smoke tests are a separate exit gate.
# ---------------------------------------------------------------------------

_ASK_TOOL = {
    "name": "ask_user",
    "description": "Ask the user a question.",
    "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
}

_TOOL_RESPONSE_BY_PROVIDER = {
    OpenAILLMClient: {
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "ask_user", "arguments": "{\"message\": \"hi\"}"},
            {"type": "message", "content": [{"type": "output_text", "text": "draft text"}]},
        ]
    },
    AnthropicLLMClient: {
        "content": [
            {"type": "text", "text": "draft text"},
            {"type": "tool_use", "id": "tu1", "name": "ask_user", "input": {"message": "hi"}},
        ]
    },
    GoogleLLMClient: {
        "candidates": [
            {"content": {"parts": [{"text": "draft text"}, {"functionCall": {"name": "ask_user", "args": {"message": "hi"}}}]}}
        ]
    },
    BedrockLLMClient: {
        "output": {"message": {"content": [
            {"text": "draft text"},
            {"toolUse": {"toolUseId": "b1", "name": "ask_user", "input": {"message": "hi"}}},
        ]}}
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_TOOL_RESPONSE_BY_PROVIDER))
async def test_generate_with_tools_returns_ai_message(monkeypatch, client_class):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[client_class])
    # Bedrock api-key path (no secret_key) uses httpx — the recorder covers it.
    client = client_class(LLMClientConfig(api_key="key-test", model="model-test"))

    # Call exactly like analyze_node does: tools set, response_format OMITTED. Guards the regression
    # where the real clients declared response_format without a default and raised TypeError.
    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Phân tích"}],
        system="Bạn là BA.",
        max_tokens=256,
        tools=[_ASK_TOOL],
    )

    assert isinstance(result, AIMessage)
    assert [tc["name"] for tc in result.tool_calls] == ["ask_user"]
    assert result.tool_calls[0]["args"] == {"message": "hi"}
    assert result.content == "draft text"  # client surfaces the text verbatim; analyze_node treats it as reasoning, not a draft
    # The request carried the tool config (so a real provider would force the call).
    assert "ask_user" in str(recorder.requests[0]["json"])


class _HttpxRecorder:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def client_class(self, *args, **kwargs):
        return _RecordingAsyncClient(self)


class _RecordingAsyncClient:
    def __init__(self, recorder):
        self.recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, **kwargs):
        self.recorder.requests.append({"url": url, **kwargs})
        return httpx.Response(200, json=self.recorder.payload, request=httpx.Request("POST", url))
