import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.models.llm_provider import ProviderType
from app.services import llm_clients as llm_client_module
from app.services.llm_clients import (
    DEFAULT_MODEL_BY_PROVIDER,
    AnthropicLLMClient,
    BedrockLLMClient,
    CustomLLMClient,
    GoogleLLMClient,
    LLMClientConfig,
    LLMClientFactory,
    MistralLLMClient,
    OpenAILLMClient,
    _extract_bedrock_text,
    _google_contents,
    _parse_generate_text,
    _parse_google_tool_response,
    _plain_dict,
    _responses_json_schema_format,
    _to_anthropic_tool,
    _to_openai_chat_tool,
    _to_openai_tool,
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
    "gaps": ["Thieu persona chinh"],
    "contradictions": [],
    "risks": ["Scope chua ro"],
    "confidence": 0.82,
    "next_action": "ask_human",
}


CHAT_JSON_OK = {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}
CHAT_RAW_ANSWER = {"choices": [{"message": {"content": "raw answer"}}]}
CHAT_INVALID_JSON = {"choices": [{"message": {"content": "not json"}}]}


def _client_config(client_class, **overrides):
    values = {"api_key": "key-test", "model": "model-test"}
    if client_class is CustomLLMClient:
        values["base_url"] = "https://custom.example/v1"
    values.update(overrides)
    return LLMClientConfig(**values)


@pytest.mark.parametrize(
    ("provider_type", "client_class"),
    [
        (ProviderType.BEDROCK, BedrockLLMClient),
        (ProviderType.OPENAI, OpenAILLMClient),
        (ProviderType.GOOGLE, GoogleLLMClient),
        (ProviderType.ANTHROPIC, AnthropicLLMClient),
        (ProviderType.MISTRAL, MistralLLMClient),
    ],
)
def test_llm_client_factory_supports_current_provider_types(provider_type, client_class):
    client = LLMClientFactory.create(provider_type=provider_type, api_key="key-test")

    assert isinstance(client, client_class)
    assert client.config.api_key == "key-test"
    assert client.config.model == DEFAULT_MODEL_BY_PROVIDER[provider_type]


def test_llm_client_factory_supports_custom_provider_type():
    client = LLMClientFactory.create(
        provider_type=ProviderType.CUSTOM,
        api_key="key-test",
        model="custom-model",
        base_url="https://custom.example/v1",
    )

    assert isinstance(client, CustomLLMClient)
    assert client.config.api_key == "key-test"
    assert client.config.model == "custom-model"
    assert client.config.base_url == "https://custom.example/v1"


@pytest.mark.parametrize(
    "kwargs",
    [{"model": None, "base_url": "https://custom.example/v1"}, {"model": "custom-model", "base_url": None}],
)
def test_llm_client_factory_rejects_incomplete_custom_provider(kwargs):
    with pytest.raises(ValueError):
        LLMClientFactory.create(provider_type=ProviderType.CUSTOM, api_key="key-test", **kwargs)


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
        (CustomLLMClient, CHAT_JSON_OK),
        (MistralLLMClient, CHAT_JSON_OK),
    ],
)
async def test_generate_with_response_format_returns_dict(monkeypatch, client_class, provider_payload):
    recorder = _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(_client_config(client_class))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Analyze requirements"}],
        system="You are a BA.",
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
        messages=[{"role": "user", "content": "Analyze requirements"}],
        system="You are a BA.",
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


def test_plain_dict_preserves_openai_sdk_output_text_property():
    class ResponseLike:
        output_text = "pong"

        def model_dump(self, **_kwargs):
            return {"output": []}

    assert _plain_dict(ResponseLike())["output_text"] == "pong"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Thieu tool"},
        {"tool": "finalize", "message": "Sai enum"},
        {"tool": "ask_user", "message": "ok", "extra": "invalid"},
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

    with pytest.raises(ValueError, match="does not match JSON Schema"):
        _parse_generate_text(json.dumps(payload, ensure_ascii=False), response_format)


def test_parse_generate_text_accepts_tool_args_prompt_alias():
    payload = {"tools": [{"name": "ask_user", "args": {"prompt": "Which part do you want to analyze?"}}]}
    schema = {"type": "object", "properties": {"tools": {"type": "array"}}, "required": ["tools"]}

    result = _parse_generate_text(json.dumps(payload, ensure_ascii=False), schema)

    assert result == payload


@pytest.mark.parametrize(
    "wrapped",
    [
        'Here is the JSON:\n{"answer": "ok"}',
        '{"answer": "ok"}\n\nLet me know if you need more.',
        'Sure — ```json\n{"answer": "ok"}\n``` done.',
        '{"answer": "ok with a } brace inside"}',
    ],
)
def test_parse_generate_text_recovers_object_wrapped_in_prose(wrapped):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

    result = _parse_generate_text(wrapped, schema)

    assert result == {"answer": result["answer"]}
    assert result["answer"].startswith("ok")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_class", "provider_payload"),
    [
        (OpenAILLMClient, {"output_text": "not json"}),
        (AnthropicLLMClient, {"content": [{"type": "text", "text": "not json"}]}),
        (GoogleLLMClient, {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}),
        (BedrockLLMClient, {"output": {"message": {"content": [{"text": "not json"}]}}}),
        (CustomLLMClient, CHAT_INVALID_JSON),
        (MistralLLMClient, CHAT_INVALID_JSON),
    ],
)
async def test_generate_with_response_format_rejects_invalid_json(monkeypatch, client_class, provider_payload):
    _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(_client_config(client_class))

    with pytest.raises(ValueError, match="Could not parse JSON"):
        await client.generate(
            messages=[{"role": "user", "content": "Analyze requirements"}],
            system="You are a BA.",
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
        (CustomLLMClient, CHAT_RAW_ANSWER),
        (MistralLLMClient, CHAT_RAW_ANSWER),
    ],
)
async def test_generate_without_response_format_returns_raw_text_and_no_extra_params(monkeypatch, client_class, provider_payload):
    recorder = _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(_client_config(client_class))

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
@pytest.mark.parametrize(
    "client_class",
    [AnthropicLLMClient, BedrockLLMClient, CustomLLMClient, MistralLLMClient],
)
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
    elif client_class in {CustomLLMClient, MistralLLMClient}:
        recorder = _install_httpx_recorder(
            monkeypatch,
            {"choices": [{"message": {"content": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}}]},
        )
    client = client_class(_client_config(client_class))

    await client.generate(
        messages=[{"role": "user", "content": "Analyze"}],
        system="You are a BA.",
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
        (
            CustomLLMClient,
            {"choices": [{"message": {"content": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}}]},
        ),
        (
            MistralLLMClient,
            {"choices": [{"message": {"content": json.dumps(ANALYSIS_RESULT, ensure_ascii=False)}}]},
        ),
    ],
)
async def test_generate_parses_analysis_result_schema_for_each_provider(monkeypatch, client_class, provider_payload):
    _install_httpx_recorder(monkeypatch, provider_payload)
    client = client_class(_client_config(client_class))

    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Analyze"}],
        system="You are a BA.",
        max_tokens=512,
        response_format={"name": "analysis_result", "schema": ANALYSIS_RESULT_SCHEMA},
    )

    assert result == ANALYSIS_RESULT


def _install_httpx_recorder(monkeypatch, payload):
    recorder = _HttpxRecorder(payload)
    monkeypatch.setattr(httpx, "AsyncClient", recorder.client_class)
    monkeypatch.setattr(llm_client_module, "_create_openai_sdk", recorder.openai_client)
    monkeypatch.setattr(llm_client_module, "_create_anthropic_sdk", recorder.anthropic_client)
    monkeypatch.setattr(llm_client_module, "_create_google_sdk", recorder.google_client)
    monkeypatch.setattr(llm_client_module, "_create_mistral_sdk", recorder.mistral_client)
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

def test_strict_tool_schema_for_providers_that_support_it():
    openai_tool = _to_openai_tool(_ASK_TOOL)
    anthropic_tool = _to_anthropic_tool(_ASK_TOOL)
    chat_tool = _to_openai_chat_tool(_ASK_TOOL)

    assert openai_tool["strict"] is True
    assert openai_tool["parameters"]["additionalProperties"] is False
    assert openai_tool["parameters"]["required"] == ["message"]
    assert anthropic_tool["strict"] is True
    assert anthropic_tool["input_schema"]["additionalProperties"] is False
    assert "strict" not in chat_tool["function"]


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
            {
                "content": {
                    "parts": [
                        {"text": "draft text"},
                        {"functionCall": {"id": "g1", "name": "ask_user", "args": {"message": "hi"}}},
                    ]
                }
            }
        ]
    },
    BedrockLLMClient: {
        "output": {"message": {"content": [
            {"text": "draft text"},
            {"toolUse": {"toolUseId": "b1", "name": "ask_user", "input": {"message": "hi"}}},
        ]}}
    },
    CustomLLMClient: {
        "choices": [
            {
                "message": {
                    "content": "draft text",
                    "tool_calls": [
                        {
                            "id": "d1",
                            "type": "function",
                            "function": {"name": "ask_user", "arguments": "{\"message\": \"hi\"}"},
                        }
                    ],
                }
            }
        ]
    },
    MistralLLMClient: {
        "choices": [
            {
                "message": {
                    "content": "draft text",
                    "tool_calls": [
                        {
                            "id": "m1",
                            "type": "function",
                            "function": {"name": "ask_user", "arguments": "{\"message\": \"hi\"}"},
                        }
                    ],
                }
            }
        ]
    },
}


_PROBE_TOOL_RESPONSE_BY_PROVIDER = {
    OpenAILLMClient: {
        "output": [
            {"type": "function_call", "call_id": "p1", "name": "probe", "arguments": "{\"ok\": true}"},
        ]
    },
    AnthropicLLMClient: {
        "content": [{"type": "tool_use", "id": "p1", "name": "probe", "input": {"ok": True}}],
    },
    GoogleLLMClient: {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"id": "p1", "name": "probe", "args": {"ok": True}}}]}}
        ],
    },
    BedrockLLMClient: {
        "output": {
            "message": {"content": [{"toolUse": {"toolUseId": "p1", "name": "probe", "input": {"ok": True}}}]}
        }
    },
    CustomLLMClient: {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "p1",
                            "type": "function",
                            "function": {"name": "probe", "arguments": "{\"ok\": true}"},
                        }
                    ],
                }
            }
        ]
    },
    MistralLLMClient: {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "p1",
                            "type": "function",
                            "function": {"name": "probe", "arguments": "{\"ok\": true}"},
                        }
                    ],
                }
            }
        ]
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_TOOL_RESPONSE_BY_PROVIDER))
async def test_generate_with_tools_returns_ai_message(monkeypatch, client_class):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[client_class])
    # Bedrock api-key path (no secret_key) uses httpx — the recorder covers it.
    client = client_class(_client_config(client_class))

    # Call exactly like analyze_node does: tools set, response_format OMITTED. Guards the regression
    # where the real clients declared response_format without a default and raised TypeError.
    result, _ = await client.generate(
        messages=[{"role": "user", "content": "Analyze"}],
        system="You are a BA.",
        max_tokens=256,
        tools=[_ASK_TOOL],
    )

    assert isinstance(result, AIMessage)
    assert [tc["name"] for tc in result.tool_calls] == ["ask_user"]
    assert result.tool_calls[0]["id"]
    assert result.tool_calls[0]["args"] == {"message": "hi"}
    assert result.content == "draft text"  # client surfaces the text verbatim; analyze_node treats it as reasoning, not a draft
    # The request carried the tool config (so a real provider would force the call).
    assert "ask_user" in str(recorder.requests[0]["json"])


_THREAD_WITH_TOOL_BLOCKS = [
    {"role": "user", "content": "I want to set goals for a study group product."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Toi will ghi nhan du kien truoc."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "explore_note",
                "input": {"content": "Primary users are study group students."},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "name": "explore_note",
                "content": "Da ghi nhan key fact.",
            },
            {"type": "text", "text": "Continue analysis from that fact."},
        ],
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_TOOL_RESPONSE_BY_PROVIDER))
async def test_generate_with_tools_serializes_thread_tool_blocks(monkeypatch, client_class):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[client_class])
    client = client_class(_client_config(client_class))

    await client.generate(
        messages=_THREAD_WITH_TOOL_BLOCKS,
        system="You are a BA.",
        max_tokens=256,
        tools=[_ASK_TOOL],
    )

    body = recorder.requests[0]["json"]
    if client_class is OpenAILLMClient:
        assert {
            "type": "function_call",
            "call_id": "call_1",
            "name": "explore_note",
            "arguments": "{\"content\": \"Primary users are study group students.\"}",
        } in body["input"]
        assert {"type": "function_call_output", "call_id": "call_1", "output": "Da ghi nhan key fact."} in body["input"]
    elif client_class is AnthropicLLMClient:
        assert body["messages"][1]["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "explore_note",
            "input": {"content": "Primary users are study group students."},
        }
        assert body["messages"][2]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "Da ghi nhan key fact.",
        }
    elif client_class is GoogleLLMClient:
        assert body["contents"][1]["parts"][1] == {
            "functionCall": {
                "id": "call_1",
                "name": "explore_note",
                "args": {"content": "Primary users are study group students."},
            }
        }
        assert body["contents"][2]["parts"][0] == {
            "functionResponse": {
                "id": "call_1",
                "name": "explore_note",
                "response": {"content": "Da ghi nhan key fact."},
            }
        }
    elif client_class in {CustomLLMClient, MistralLLMClient}:
        assert body["messages"][2]["tool_calls"][0] == {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "explore_note",
                "arguments": "{\"content\": \"Primary users are study group students.\"}",
            },
        }
        assert body["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Da ghi nhan key fact.",
        }
        assert body["messages"][4] == {"role": "user", "content": "Continue analysis from that fact."}
    else:
        assert body["messages"][1]["content"][1] == {
            "toolUse": {
                "toolUseId": "call_1",
                "name": "explore_note",
                "input": {"content": "Primary users are study group students."},
            }
        }
        assert body["messages"][2]["content"][0] == {
            "toolResult": {"toolUseId": "call_1", "content": [{"text": "Da ghi nhan key fact."}]}
        }


def test_google_function_call_replay_preserves_thought_signature_and_original_id():
    from app.graphs.analysis.prompt_assembly import _client_message_from_state
    from app.graphs.analysis.tool_gating import _model_tool_calls
    from app.graphs.nodes import _dispatch_ai_message

    model_message = _parse_google_tool_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "google-call-1",
                                    "name": "confirm_intent",
                                    "args": {"summary": "Need attendance workflow."},
                                },
                                "thoughtSignature": b"encrypted-signature",
                            }
                        ]
                    }
                }
            ]
        }
    )
    stored_signature = model_message.additional_kwargs["provider_tool_calls"]["google-call-1"]["google"][
        "thoughtSignature"
    ]
    assert stored_signature["encoding"] == "base64"
    assert not isinstance(stored_signature, bytes)
    model_tool_call = _model_tool_calls(model_message)[0]
    dispatch_message = _dispatch_ai_message(
        [
            {
                "id": "runtime-call-1",
                "name": model_tool_call["name"],
                "args": model_tool_call["args"],
                "provider_metadata": model_tool_call["provider_metadata"],
            }
        ]
    )

    tool_names_by_id: dict[str, str] = {}
    tool_provider_metadata_by_id: dict[str, dict] = {}
    assistant_message = _client_message_from_state(
        dispatch_message,
        tool_names_by_id,
        tool_provider_metadata_by_id,
    )
    tool_result_message = _client_message_from_state(
        ToolMessage(content="Confirmed.", tool_call_id="runtime-call-1"),
        tool_names_by_id,
        tool_provider_metadata_by_id,
    )

    contents = _google_contents([assistant_message, tool_result_message])

    function_call_part = contents[0]["parts"][0]
    assert function_call_part["functionCall"]["id"] == "google-call-1"
    assert function_call_part["thoughtSignature"] == b"encrypted-signature"
    assert contents[1]["parts"][0]["functionResponse"]["id"] == "google-call-1"


def test_google_synthetic_tool_call_without_signature_renders_as_text():
    """tool_gating._plain_response_tool fabricates a "respond"/"ask_user" call locally when the
    model answers in plain text; it carries no provider thoughtSignature since it never came from
    Gemini. Replaying it as a functionCall/functionResponse pair on the next turn triggers Gemini's
    'missing a thought signature in functionCall parts' 400. It must render as plain text instead.
    """
    from app.graphs.analysis.prompt_assembly import _client_message_from_state
    from app.graphs.nodes import _dispatch_ai_message

    dispatch_message = _dispatch_ai_message(
        [{"id": "synthetic-call-1", "name": "respond", "args": {"message": "Here is my analysis."}}]
    )

    tool_names_by_id: dict[str, str] = {}
    tool_provider_metadata_by_id: dict[str, dict] = {}
    assistant_message = _client_message_from_state(
        dispatch_message,
        tool_names_by_id,
        tool_provider_metadata_by_id,
    )
    tool_result_message = _client_message_from_state(
        ToolMessage(content="Delivered.", tool_call_id="synthetic-call-1"),
        tool_names_by_id,
        tool_provider_metadata_by_id,
    )

    contents = _google_contents([assistant_message, tool_result_message])

    assistant_parts = contents[0]["parts"]
    result_parts = contents[1]["parts"]
    assert all("functionCall" not in part for part in assistant_parts)
    assert all("functionResponse" not in part for part in result_parts)
    assert {"text": "Here is my analysis."} in assistant_parts


# ---------------------------------------------------------------------------
# Health-check tool probe — verifies provider calls the expected probe tool with valid args.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", list(_PROBE_TOOL_RESPONSE_BY_PROVIDER))
async def test_ping_tool_calling_accepts_valid_probe_call(monkeypatch, client_class):
    recorder = _install_httpx_recorder(monkeypatch, _PROBE_TOOL_RESPONSE_BY_PROVIDER[client_class])
    client = client_class(_client_config(client_class))

    assert await client.ping_tool_calling("auto") is True

    body = recorder.requests[0]["json"]
    if client_class is OpenAILLMClient:
        assert body["tool_choice"] == "auto"
        assert body["tools"][0]["strict"] is True
    elif client_class is AnthropicLLMClient:
        assert body["tool_choice"] == {"type": "auto"}
        assert body["tools"][0]["strict"] is True
    elif client_class is GoogleLLMClient:
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    elif client_class is BedrockLLMClient:
        assert "toolChoice" not in body["toolConfig"]
    elif client_class is MistralLLMClient:
        assert body["tool_choice"] == "auto"
    else:
        assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_payload",
    [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "p1",
                                "type": "function",
                                "function": {"name": "wrong_probe", "arguments": "{\"ok\": true}"},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "p1",
                                "type": "function",
                                "function": {"name": "probe", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "p1",
                                "type": "function",
                                "function": {"name": "probe", "arguments": "{\"ok\": \"yes\"}"},
                            }
                        ],
                    }
                }
            ]
        },
    ],
)
async def test_ping_tool_calling_rejects_invalid_probe_call(monkeypatch, provider_payload):
    _install_httpx_recorder(monkeypatch, provider_payload)
    client = CustomLLMClient(_client_config(CustomLLMClient))

    assert await client.ping_tool_calling() is False


# ---------------------------------------------------------------------------
# tool_choice wire format — verify "auto" (default) and "required" (rollback) reach each provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_tool_choice_auto_sends_auto(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[OpenAILLMClient])
    client = OpenAILLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="auto")
    assert recorder.requests[0]["json"]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openai_tool_choice_required_sends_required(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[OpenAILLMClient])
    client = OpenAILLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="required")
    assert recorder.requests[0]["json"]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_anthropic_tool_choice_auto_sends_type_auto(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[AnthropicLLMClient])
    client = AnthropicLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="auto")
    assert recorder.requests[0]["json"]["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_anthropic_tool_choice_required_sends_type_any(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[AnthropicLLMClient])
    client = AnthropicLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="required")
    assert recorder.requests[0]["json"]["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
async def test_anthropic_generate_with_tools_marks_system_as_ephemeral_cache_block(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[AnthropicLLMClient])
    client = AnthropicLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(
        messages=[{"role": "user", "content": "x"}],
        system="Static analyst policy contract.",
        max_tokens=16,
        tools=[_ASK_TOOL],
    )
    assert recorder.requests[0]["json"]["system"] == [
        {"type": "text", "text": "Static analyst policy contract.", "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.asyncio
async def test_anthropic_generate_without_tools_marks_system_as_ephemeral_cache_block(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, {"content": [{"type": "text", "text": "raw answer"}]})
    client = AnthropicLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(
        messages=[{"role": "user", "content": "x"}],
        system="You are a BA.",
        max_tokens=16,
        response_format=None,
    )
    assert recorder.requests[0]["json"]["system"] == [
        {"type": "text", "text": "You are a BA.", "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.asyncio
async def test_google_tool_choice_auto_sends_mode_auto(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[GoogleLLMClient])
    client = GoogleLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="auto")
    assert recorder.requests[0]["json"]["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"


@pytest.mark.asyncio
async def test_google_tool_choice_required_sends_mode_any(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[GoogleLLMClient])
    client = GoogleLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="required")
    assert recorder.requests[0]["json"]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


@pytest.mark.asyncio
async def test_custom_chat_tool_choice_required_sends_required(monkeypatch):
    client_class = CustomLLMClient
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[client_class])
    client = client_class(_client_config(client_class))
    await client.generate(
        messages=[{"role": "user", "content": "x"}],
        system=None,
        max_tokens=16,
        tools=[_ASK_TOOL],
        tool_choice="required",
    )
    assert recorder.requests[0]["json"]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_custom_chat_completion_posts_to_configured_base_url(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, CHAT_RAW_ANSWER)
    client = CustomLLMClient(_client_config(CustomLLMClient))

    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16)

    request = recorder.requests[0]
    assert request["url"] == "https://custom.example/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer key-test"
    assert request["json"]["model"] == "model-test"


@pytest.mark.asyncio
async def test_custom_chat_completion_disables_streaming(monkeypatch):
    requests = []

    class StreamSensitiveAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, **kwargs):
            requests.append({"url": url, **kwargs})
            if kwargs["json"].get("stream") is False:
                return httpx.Response(200, json=CHAT_RAW_ANSWER, request=httpx.Request("POST", url))
            return httpx.Response(
                200,
                content=b'{"choices":[{"message":{"content":"pong"}}]}\n{"choices":[{"message":{"content":"extra"}}]}',
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", StreamSensitiveAsyncClient)
    client = CustomLLMClient(_client_config(CustomLLMClient))

    assert await client.ping() == "raw answer"
    assert requests[0]["json"]["stream"] is False


@pytest.mark.asyncio
async def test_mistral_tool_choice_required_sends_any(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[MistralLLMClient])
    client = MistralLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(
        messages=[{"role": "user", "content": "x"}],
        system=None,
        max_tokens=16,
        tools=[_ASK_TOOL],
        tool_choice="required",
    )
    assert recorder.requests[0]["json"]["tool_choice"] == "any"


@pytest.mark.asyncio
async def test_bedrock_tool_choice_auto_omits_tool_choice_field(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[BedrockLLMClient])
    client = BedrockLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="auto")
    assert "toolChoice" not in recorder.requests[0]["json"]["toolConfig"]


@pytest.mark.asyncio
async def test_bedrock_tool_choice_required_sends_any(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _TOOL_RESPONSE_BY_PROVIDER[BedrockLLMClient])
    client = BedrockLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))
    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16, tools=[_ASK_TOOL], tool_choice="required")
    assert recorder.requests[0]["json"]["toolConfig"]["toolChoice"] == {"any": {}}


@pytest.mark.asyncio
async def test_bedrock_ping_tool_calling_forces_any_tool_choice(monkeypatch):
    recorder = _install_httpx_recorder(monkeypatch, _PROBE_TOOL_RESPONSE_BY_PROVIDER[BedrockLLMClient])
    client = BedrockLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))

    assert await client.ping_tool_calling() is True
    assert recorder.requests[0]["json"]["toolConfig"]["toolChoice"] == {"any": {}}


@pytest.mark.asyncio
async def test_bedrock_iam_key_generate_reuses_boto3_client_across_calls(monkeypatch):
    call_count = {"n": 0}

    class _FakeBotoClient:
        def converse(self, **kwargs):
            return _TOOL_RESPONSE_BY_PROVIDER[BedrockLLMClient]

    def _fake_boto3_client(*args, **kwargs):
        call_count["n"] += 1
        return _FakeBotoClient()

    monkeypatch.setattr("boto3.client", _fake_boto3_client)
    client = BedrockLLMClient(
        LLMClientConfig(api_key="AKIATEST", model="model-test", secret_key="secret-test")
    )

    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16)
    await client.generate(messages=[{"role": "user", "content": "y"}], system=None, max_tokens=16)

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_bedrock_iam_key_ping_reuses_boto3_client_across_calls(monkeypatch):
    call_count = {"n": 0}

    class _FakeBotoClient:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "pong"}]}}}

    def _fake_boto3_client(*args, **kwargs):
        call_count["n"] += 1
        return _FakeBotoClient()

    monkeypatch.setattr("boto3.client", _fake_boto3_client)
    client = BedrockLLMClient(
        LLMClientConfig(api_key="AKIATEST", model="model-test", secret_key="secret-test")
    )

    await client.ping()
    await client.ping()

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_mistral_generate_reuses_sdk_client_across_calls(monkeypatch):
    create_count = {"n": 0}

    def _counting_create_mistral_sdk(*, api_key, timeout):
        create_count["n"] += 1
        return _RecordingMistralClient(_HttpxRecorder(CHAT_JSON_OK))

    monkeypatch.setattr(llm_client_module, "_create_mistral_sdk", _counting_create_mistral_sdk)
    client = MistralLLMClient(LLMClientConfig(api_key="key-test", model="model-test"))

    await client.generate(messages=[{"role": "user", "content": "x"}], system=None, max_tokens=16)
    await client.generate(messages=[{"role": "user", "content": "y"}], system=None, max_tokens=16)

    assert create_count["n"] == 1


class _HttpxRecorder:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def client_class(self, *args, **kwargs):
        return _RecordingAsyncClient(self)

    def openai_client(self, *args, **kwargs):
        return _RecordingOpenAIClient(self)

    def openai_chat_client(self, *args, **kwargs):
        return _RecordingOpenAIChatClient(self)

    def anthropic_client(self, *args, **kwargs):
        return _RecordingAnthropicClient(self)

    def google_client(self, *args, **kwargs):
        return _RecordingGoogleClient(self)

    def mistral_client(self, *args, **kwargs):
        return _RecordingMistralClient(self)


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


class _RecordingOpenAIClient:
    def __init__(self, recorder):
        self.responses = _RecordingSDKResource(recorder)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingOpenAIChatClient:
    def __init__(self, recorder):
        self.chat = _RecordingChatNamespace(recorder)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingAnthropicClient:
    def __init__(self, recorder):
        self.messages = _RecordingSDKResource(recorder)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingSDKResource:
    def __init__(self, recorder):
        self.recorder = recorder

    async def create(self, **kwargs):
        self.recorder.requests.append({"json": kwargs})
        return self.recorder.payload


class _RecordingChatNamespace:
    def __init__(self, recorder):
        self.completions = _RecordingSDKResource(recorder)


class _RecordingMistralClient:
    def __init__(self, recorder):
        self.chat = _RecordingMistralChat(recorder)


class _RecordingMistralChat:
    def __init__(self, recorder):
        self.recorder = recorder

    def complete(self, **kwargs):
        self.recorder.requests.append({"json": kwargs})
        return self.recorder.payload


class _RecordingGoogleClient:
    def __init__(self, recorder):
        self.aio = _RecordingGoogleAsyncClient(recorder)


class _RecordingGoogleAsyncClient:
    def __init__(self, recorder):
        self.models = _RecordingGoogleModels(recorder)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _RecordingGoogleModels:
    def __init__(self, recorder):
        self.recorder = recorder

    async def generate_content(self, *, model, contents, config=None):
        self.recorder.requests.append({"json": {"model": model, "contents": contents, **(config or {})}})
        return self.recorder.payload
