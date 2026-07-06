import asyncio
import copy
import json
import threading
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx
import jsonschema
from langchain_core.messages import AIMessage

from app.models.llm_provider import ProviderType

DEFAULT_MODEL_BY_PROVIDER = {
    ProviderType.BEDROCK: "amazon.nova-lite-v1:0",
    ProviderType.OPENAI: "gpt-5.4-mini",
    ProviderType.GOOGLE: "gemini-3.5-flash",
    ProviderType.ANTHROPIC: "claude-haiku-4-5-20251001",
    ProviderType.MISTRAL: "mistral-large-latest",
}


@dataclass(frozen=True)
class LLMClientConfig:
    api_key: str
    model: str
    region: str | None = None
    secret_key: str | None = None
    base_url: str | None = None


_TOOL_CALL_PROBE = {
    "name": "probe",
    "description": "Connectivity probe.",
    "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
}


def _to_bedrock_probe_tool() -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": _TOOL_CALL_PROBE["name"],
            "description": _TOOL_CALL_PROBE["description"],
            "inputSchema": {"json": _TOOL_CALL_PROBE["parameters"]},
        }
    }


# Provider-agnostic tool schema (input to generate(tools=...)):
#   {"name": str, "description": str, "parameters": <JSON Schema object>}
# Each converter maps it to the wire format the provider's tool-calling API expects.
_EMPTY_PARAMS = {"type": "object", "properties": {}}


def _to_anthropic_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool_schema["name"],
        "description": tool_schema.get("description", ""),
        "input_schema": tool_schema.get("parameters") or dict(_EMPTY_PARAMS),
    }


def _to_openai_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
    # Responses API uses the flat function format (not nested under "function" like Chat
    # Completions) — confirmed live by ping_tool_calling.
    return {
        "type": "function",
        "name": tool_schema["name"],
        "description": tool_schema.get("description", ""),
        "parameters": tool_schema.get("parameters") or dict(_EMPTY_PARAMS),
    }


def _to_openai_chat_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_schema["name"],
            "description": tool_schema.get("description", ""),
            "parameters": tool_schema.get("parameters") or dict(_EMPTY_PARAMS),
        },
    }


def _to_google_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool_schema["name"],
        "description": tool_schema.get("description", ""),
        "parametersJsonSchema": tool_schema.get("parameters") or dict(_EMPTY_PARAMS),
    }


def _to_bedrock_tool(tool_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool_schema["name"],
            "description": tool_schema.get("description", ""),
            "inputSchema": {"json": tool_schema.get("parameters") or dict(_EMPTY_PARAMS)},
        }
    }


def _parse_anthropic_tool_response(data: dict[str, Any]) -> AIMessage:
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in data.get("content") or []:
        if block.get("type") == "tool_use":
            tool_calls.append(
                {"id": block.get("id") or "", "name": block.get("name") or "", "args": block.get("input") or {}}
            )
        elif block.get("type") == "text":
            text_parts.append(block.get("text") or "")
    return AIMessage(content=" ".join(p for p in text_parts if p).strip(), tool_calls=tool_calls)


def _parse_openai_tool_response(data: dict[str, Any]) -> AIMessage:
    # Responses API: output is a list; function_call items carry .name, .arguments (JSON string),
    # .call_id; message items hold output_text parts.
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for item in data.get("output") or []:
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name") or "",
                    "args": _loads_args(item.get("arguments")),
                }
            )
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text") is not None:
                    text_parts.append(part["text"])
    return AIMessage(content=" ".join(p for p in text_parts if p).strip(), tool_calls=tool_calls)


def _parse_openai_chat_tool_response(data: dict[str, Any]) -> AIMessage:
    choices = data.get("choices") or []
    if not choices:
        return AIMessage(content="", tool_calls=[])

    message = choices[0].get("message") or {}
    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        tool_calls.append(
            {
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "args": _loads_args(function.get("arguments")),
            }
        )
    return AIMessage(content=_chat_content_text(message.get("content")), tool_calls=tool_calls)


def _parse_google_tool_response(data: dict[str, Any]) -> AIMessage:
    candidates = data.get("candidates") or []
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    if candidates:
        for part in candidates[0].get("content", {}).get("parts") or []:
            fn = part.get("functionCall") or part.get("function_call")
            if fn:
                tool_calls.append(
                    {"id": fn.get("id") or "", "name": fn.get("name") or "", "args": fn.get("args") or {}}
                )
            elif part.get("text"):
                text_parts.append(part["text"])
    return AIMessage(content=" ".join(p for p in text_parts if p).strip(), tool_calls=tool_calls)


def _parse_bedrock_tool_response(data: dict[str, Any]) -> AIMessage:
    content = data.get("output", {}).get("message", {}).get("content") or []
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in content:
        use = block.get("toolUse")
        if use:
            tool_calls.append(
                {"id": use.get("toolUseId") or "", "name": use.get("name") or "", "args": use.get("input") or {}}
            )
        elif block.get("text"):
            text_parts.append(block["text"])
    return AIMessage(content=" ".join(p for p in text_parts if p).strip(), tool_calls=tool_calls)


def _loads_args(raw: Any) -> dict[str, Any]:
    """Parse a tool-call arguments payload that may arrive as a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(dumped, dict):
            return {}
        output_text = getattr(value, "output_text", None)
        if output_text is not None and "output_text" not in dumped:
            dumped["output_text"] = output_text
        return dumped
    return {}


def _create_openai_sdk(*, api_key: str, timeout: float):
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, timeout=timeout)


def _create_anthropic_sdk(*, api_key: str, timeout: float):
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=api_key, timeout=timeout)


def _create_mistral_sdk(*, api_key: str, timeout: float):
    _ = timeout
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral

    return Mistral(api_key=api_key)


def _create_google_sdk(*, api_key: str, timeout: float):
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))


async def _openai_create_response(api_key: str, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    async with _create_openai_sdk(api_key=api_key, timeout=timeout) as client:
        response = await client.responses.create(**body)
    return _plain_dict(response)


async def _mistral_create_chat_completion(
    api_key: str, timeout: float, body: dict[str, Any], *, client: Any = None
) -> dict[str, Any]:
    def _complete() -> dict[str, Any]:
        sdk_client = client if client is not None else _create_mistral_sdk(api_key=api_key, timeout=timeout)
        return _plain_dict(sdk_client.chat.complete(**body))

    return await asyncio.to_thread(_complete)


async def _anthropic_create_message(api_key: str, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    async with _create_anthropic_sdk(api_key=api_key, timeout=timeout) as client:
        response = await client.messages.create(**body)
    return _plain_dict(response)


async def _google_generate_content(api_key: str, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    client = _create_google_sdk(api_key=api_key, timeout=timeout)
    async with client.aio as aio_client:
        response = await aio_client.models.generate_content(
            model=body["model"],
            contents=body["contents"],
            config=body.get("config"),
        )
    return _plain_dict(response)


class LLMClient(Protocol):
    async def ping(self) -> str | None:
        pass

    async def ping_tool_calling(self) -> bool:
        pass

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        pass


class OpenAILLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        data = await _openai_create_response(
            self.config.api_key,
            5.0,
            {"model": self.config.model, "input": "ping", "max_output_tokens": 5},
        )
        return data.get("output_text")

    async def ping_tool_calling(self) -> bool:
        body = {
            "model": self.config.model,
            "input": "ok",
            "max_output_tokens": 20,
            "tools": [
                {
                    "type": "function",
                    "name": _TOOL_CALL_PROBE["name"],
                    "description": _TOOL_CALL_PROBE["description"],
                    "parameters": _TOOL_CALL_PROBE["parameters"],
                }
            ],
            "tool_choice": "required",
        }
        data = await _openai_create_response(self.config.api_key, 10.0, body)
        output = data.get("output") or []
        return any(item.get("type") == "function_call" for item in output)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        if tools:
            return await self._generate_with_tools(messages, system, max_tokens, tools, tool_choice=tool_choice)

        input_messages = _openai_messages(messages, system)
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if response_format:
            body["text"] = {"format": _responses_json_schema_format(response_format)}

        data = await _openai_create_response(self.config.api_key, 30.0, body)

        text = _extract_openai_text(data)
        return _parse_generate_text(text, response_format), _extract_openai_usage(data)

    async def _generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> tuple[AIMessage, dict[str, int] | None]:
        body = {
            "model": self.config.model,
            "input": _openai_messages(messages, system),
            "max_output_tokens": max_tokens,
            "tools": [_to_openai_tool(t) for t in tools],
            "tool_choice": tool_choice,
        }
        data = await _openai_create_response(self.config.api_key, 30.0, body)
        return _parse_openai_tool_response(data), _extract_openai_usage(data)


class ChatCompletionsLLMClient:
    required_tool_choice = "required"

    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        data = await self._chat_completion(
            5.0,
            {"model": self.config.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
        )
        return _extract_openai_chat_text(data)

    async def ping_tool_calling(self) -> bool:
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 20,
            "tools": [_to_openai_chat_tool(_TOOL_CALL_PROBE)],
            "tool_choice": self.required_tool_choice,
        }
        data = await self._chat_completion(10.0, body)
        return bool(_parse_openai_chat_tool_response(data).tool_calls)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        if tools:
            return await self._generate_with_tools(messages, system, max_tokens, tools, tool_choice=tool_choice)

        body = {
            "model": self.config.model,
            "messages": _openai_chat_messages(messages, _system_with_schema_instruction(system, response_format)),
            "max_tokens": max_tokens,
        }
        data = await self._chat_completion(30.0, body)
        return _parse_generate_text(_extract_openai_chat_text(data), response_format), _extract_openai_chat_usage(data)

    async def _generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> tuple[AIMessage, dict[str, int] | None]:
        body = {
            "model": self.config.model,
            "messages": _openai_chat_messages(messages, system),
            "max_tokens": max_tokens,
            "tools": [_to_openai_chat_tool(t) for t in tools],
            "tool_choice": self._wire_tool_choice(tool_choice),
        }
        data = await self._chat_completion(30.0, body)
        return _parse_openai_chat_tool_response(data), _extract_openai_chat_usage(data)

    async def _chat_completion(self, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _wire_tool_choice(self, tool_choice: str) -> str:
        if tool_choice == "required":
            return self.required_tool_choice
        if tool_choice == "auto":
            return "auto"
        return tool_choice


class CustomLLMClient(ChatCompletionsLLMClient):
    async def _chat_completion(self, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
        if not self.config.base_url:
            raise ValueError("base_url is required for custom provider")
        request_body = {**body, "stream": False}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=request_body,
            )
            response.raise_for_status()
            return response.json()


class MistralLLMClient(ChatCompletionsLLMClient):
    required_tool_choice = "any"

    def __init__(self, config: LLMClientConfig):
        super().__init__(config)
        self._sdk_client: Any = None

    async def _chat_completion(self, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
        if self._sdk_client is None:
            self._sdk_client = _create_mistral_sdk(api_key=self.config.api_key, timeout=timeout)
        return await _mistral_create_chat_completion(self.config.api_key, timeout, body, client=self._sdk_client)


class GoogleLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        data = await _google_generate_content(
            self.config.api_key,
            5.0,
            {
                "model": self.config.model,
                "contents": [{"parts": [{"text": "ping"}]}],
                "config": {"maxOutputTokens": 5},
            },
        )
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        return parts[0].get("text")

    async def ping_tool_calling(self) -> bool:
        body = {
            "model": self.config.model,
            "contents": [{"role": "user", "parts": [{"text": "ok"}]}],
            "config": {
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": _TOOL_CALL_PROBE["name"],
                                "description": _TOOL_CALL_PROBE["description"],
                                "parametersJsonSchema": _TOOL_CALL_PROBE["parameters"],
                            }
                        ]
                    }
                ],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                "maxOutputTokens": 20,
            },
        }
        data = await _google_generate_content(self.config.api_key, 10.0, body)
        candidates = data.get("candidates") or []
        if not candidates:
            return False
        parts = candidates[0].get("content", {}).get("parts") or []
        return any("functionCall" in p for p in parts)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        if tools:
            return await self._generate_with_tools(messages, system, max_tokens, tools, tool_choice=tool_choice)

        config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if response_format:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = response_format.get("schema", response_format)
        if system:
            config["systemInstruction"] = {"parts": [{"text": system}]}

        body: dict[str, Any] = {
            "model": self.config.model,
            "contents": _google_contents(messages),
            "config": config,
        }

        data = await _google_generate_content(self.config.api_key, 30.0, body)

        text = _extract_google_text(data)
        return _parse_generate_text(text, response_format), _extract_google_usage(data)

    async def _generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> tuple[AIMessage, dict[str, int] | None]:
        _GOOGLE_MODE = {"auto": "AUTO", "required": "ANY"}
        body: dict[str, Any] = {
            "model": self.config.model,
            "contents": _google_contents(messages),
            "config": {
                "tools": [{"functionDeclarations": [_to_google_tool(t) for t in tools]}],
                "toolConfig": {"functionCallingConfig": {"mode": _GOOGLE_MODE.get(tool_choice, "AUTO")}},
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["config"]["systemInstruction"] = {"parts": [{"text": system}]}
        data = await _google_generate_content(self.config.api_key, 30.0, body)
        return _parse_google_tool_response(data), _extract_google_usage(data)


class AnthropicLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        data = await _anthropic_create_message(
            self.config.api_key,
            5.0,
            {"model": self.config.model, "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]},
        )
        content = data.get("content") or []
        if not content:
            return None
        return content[0].get("text")

    async def ping_tool_calling(self) -> bool:
        body = {
            "model": self.config.model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "ok"}],
            "tools": [
                {
                    "name": _TOOL_CALL_PROBE["name"],
                    "description": _TOOL_CALL_PROBE["description"],
                    "input_schema": _TOOL_CALL_PROBE["parameters"],
                }
            ],
            "tool_choice": {"type": "any"},
        }
        data = await _anthropic_create_message(self.config.api_key, 10.0, body)
        return any(b.get("type") == "tool_use" for b in (data.get("content") or []))

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        if tools:
            return await self._generate_with_tools(messages, system, max_tokens, tools, tool_choice=tool_choice)

        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": _anthropic_messages(messages),
        }
        final_system = _system_with_schema_instruction(system, response_format)
        if final_system:
            body["system"] = _anthropic_system_blocks(final_system)

        data = await _anthropic_create_message(self.config.api_key, 30.0, body)

        text = _extract_anthropic_text(data)
        return _parse_generate_text(text, response_format), _extract_anthropic_usage(data)

    async def _generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> tuple[AIMessage, dict[str, int] | None]:
        _ANTHROPIC_CHOICE = {"auto": {"type": "auto"}, "required": {"type": "any"}}
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": _anthropic_messages(messages),
            "tools": [_to_anthropic_tool(t) for t in tools],
            "tool_choice": _ANTHROPIC_CHOICE.get(tool_choice, {"type": "auto"}),
        }
        if system:
            body["system"] = _anthropic_system_blocks(system)
        data = await _anthropic_create_message(self.config.api_key, 30.0, body)
        return _parse_anthropic_tool_response(data), _extract_anthropic_usage(data)


class BedrockLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config
        self._iam_boto3_client: Any = None
        self._iam_boto3_client_lock = threading.Lock()

    def _get_iam_boto3_client(self) -> Any:
        # Runs inside asyncio.to_thread() worker threads — run_critique can dispatch more than
        # one concurrently via LangGraph's gather-based ToolNode, so the lazy-init check-then-set
        # needs a real thread lock, not just an asyncio-level guard.
        if self._iam_boto3_client is None:
            with self._iam_boto3_client_lock:
                if self._iam_boto3_client is None:
                    import boto3

                    self._iam_boto3_client = boto3.client(
                        "bedrock-runtime",
                        region_name=self.config.region or "us-east-1",
                        aws_access_key_id=self.config.api_key,
                        aws_secret_access_key=self.config.secret_key,
                    )
        return self._iam_boto3_client

    async def ping(self) -> str | None:
        if self.config.secret_key:
            return await self._ping_with_iam_keys()
        return await self._ping_with_api_key()

    async def ping_tool_calling(self) -> bool:
        if self.config.secret_key:
            return await self._ping_tool_calling_with_iam_keys()
        return await self._ping_tool_calling_with_api_key()

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str | dict[str, Any] | AIMessage, dict[str, int] | None]:
        if self.config.secret_key:
            data = await self._generate_with_iam_keys(
                messages, system, max_tokens, response_format, tools, tool_choice=tool_choice
            )
        else:
            data = await self._generate_with_api_key(
                messages, system, max_tokens, response_format, tools, tool_choice=tool_choice
            )
        if tools:
            return _parse_bedrock_tool_response(data), _extract_bedrock_usage(data)
        text = _extract_bedrock_text(data)
        return _parse_generate_text(text, response_format), _extract_bedrock_usage(data)

    async def _ping_with_iam_keys(self) -> str | None:
        def _ping() -> str | None:
            client = self._get_iam_boto3_client()
            response = client.converse(
                modelId=self.config.model,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0.0},
            )
            return _extract_bedrock_text(response)

        return await asyncio.to_thread(_ping)

    async def _ping_with_api_key(self) -> str | None:
        import httpx

        region = self.config.region or "us-east-1"
        model_id = quote(self.config.model, safe="")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "messages": [{"role": "user", "content": [{"text": "ping"}]}],
                    "inferenceConfig": {"maxTokens": 5, "temperature": 0.0},
                },
            )
            response.raise_for_status()
            data = response.json()
        return _extract_bedrock_text(data)

    async def _ping_tool_calling_with_iam_keys(self) -> bool:
        def _check() -> bool:
            client = self._get_iam_boto3_client()
            response = client.converse(
                modelId=self.config.model,
                messages=[{"role": "user", "content": [{"text": "ok"}]}],
                inferenceConfig={"maxTokens": 20, "temperature": 0.0},
                toolConfig={"tools": [_to_bedrock_probe_tool()], "toolChoice": {"any": {}}},
            )
            content = response.get("output", {}).get("message", {}).get("content") or []
            return any(b.get("toolUse") for b in content)

        return await asyncio.to_thread(_check)

    async def _ping_tool_calling_with_api_key(self) -> bool:
        import httpx

        region = self.config.region or "us-east-1"
        model_id = quote(self.config.model, safe="")
        body = {
            "messages": [{"role": "user", "content": [{"text": "ok"}]}],
            "inferenceConfig": {"maxTokens": 20, "temperature": 0.0},
            "toolConfig": {"tools": [_to_bedrock_probe_tool()], "toolChoice": {"any": {}}},
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        content = data.get("output", {}).get("message", {}).get("content") or []
        return any(b.get("toolUse") for b in content)

    async def _generate_with_iam_keys(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        def _generate() -> dict[str, Any]:
            client = self._get_iam_boto3_client()
            converse_kwargs: dict[str, Any] = {
                "modelId": self.config.model,
                "messages": _bedrock_messages(messages),
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
            }
            # Tool calls suppress the JSON-schema system instruction: the toolConfig drives the
            # structured output instead, so only the raw caller system prompt is forwarded.
            bedrock_system = [{"text": system}] if (tools and system) else _bedrock_system(system, response_format)
            if bedrock_system:
                converse_kwargs["system"] = bedrock_system
            if tools:
                tool_config: dict[str, Any] = {"tools": [_to_bedrock_tool(t) for t in tools]}
                # Bedrock maps "required" → {"any": {}} (must call at least one).
                # "auto" leaves toolChoice unset (model decides).
                if tool_choice == "required":
                    tool_config["toolChoice"] = {"any": {}}
                converse_kwargs["toolConfig"] = tool_config
            return client.converse(**converse_kwargs)

        return await asyncio.to_thread(_generate)

    async def _generate_with_api_key(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        import httpx

        region = self.config.region or "us-east-1"
        model_id = quote(self.config.model, safe="")
        body: dict[str, Any] = {
            "messages": _bedrock_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
        }
        bedrock_system = [{"text": system}] if (tools and system) else _bedrock_system(system, response_format)
        if bedrock_system:
            body["system"] = bedrock_system
        if tools:
            tool_config: dict[str, Any] = {"tools": [_to_bedrock_tool(t) for t in tools]}
            if tool_choice == "required":
                tool_config["toolChoice"] = {"any": {}}
            body["toolConfig"] = tool_config

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=body,
            )
            response.raise_for_status()
            return response.json()


class LLMClientFactory:
    _client_classes = {
        ProviderType.BEDROCK: BedrockLLMClient,
        ProviderType.OPENAI: OpenAILLMClient,
        ProviderType.GOOGLE: GoogleLLMClient,
        ProviderType.ANTHROPIC: AnthropicLLMClient,
        ProviderType.MISTRAL: MistralLLMClient,
        ProviderType.CUSTOM: CustomLLMClient,
    }

    @classmethod
    def create(
        cls,
        *,
        provider_type: ProviderType,
        api_key: str,
        model: str | None = None,
        region: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMClient:
        client_class = cls._client_classes.get(provider_type)
        if client_class is None:
            raise ValueError(f"Unsupported LLM provider type: {provider_type.value}")
        if provider_type == ProviderType.CUSTOM:
            if not model:
                raise ValueError("model is required for custom provider")
            if not base_url:
                raise ValueError("base_url is required for custom provider")
            resolved_model = model
        else:
            resolved_model = model or DEFAULT_MODEL_BY_PROVIDER[provider_type]
        return client_class(
            LLMClientConfig(
                api_key=api_key,
                secret_key=secret_key,
                region=region,
                model=resolved_model,
                base_url=base_url,
            )
        )


def _extract_bedrock_text(data: dict[str, Any]) -> str | None:
    content = data.get("output", {}).get("message", {}).get("content") or []
    if not content:
        return None
    return content[0].get("text")


def _normalize_usage(input_tokens: Any, output_tokens: Any, total_tokens: Any = None) -> dict[str, int] | None:
    """Normalize token usage into {"input", "output", "total"}; return None if the provider omits it."""
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    total = int(total_tokens) if total_tokens is not None else inp + out
    return {"input": inp, "output": out, "total": total}


def _extract_openai_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage")
    if not usage:
        return None
    return _normalize_usage(usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"))


def _extract_openai_chat_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage")
    if not usage:
        return None
    return _normalize_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))


def _extract_anthropic_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage")
    if not usage:
        return None
    return _normalize_usage(usage.get("input_tokens"), usage.get("output_tokens"))


def _extract_google_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usageMetadata")
    if not usage:
        return None
    return _normalize_usage(
        usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), usage.get("totalTokenCount")
    )


def _extract_bedrock_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage")
    if not usage:
        return None
    return _normalize_usage(usage.get("inputTokens"), usage.get("outputTokens"), usage.get("totalTokens"))


def _parse_generate_text(text: str | None, response_format: dict[str, Any] | None) -> str | dict[str, Any]:
    text = text or ""
    if not response_format:
        return text

    stripped = _strip_json_fence(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Instruction-based structured output (Anthropic/Bedrock) is not a hard
        # constraint, so the model may wrap the object in prose or a partial fence.
        # Recover the first balanced JSON object before giving up.
        candidate = _extract_json_object(stripped)
        if candidate is None:
            raise ValueError("Could not parse JSON from LLM response") from exc
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON from LLM response") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Could not parse JSON object from LLM response")
    schema = _parse_validation_schema(response_format)
    if schema:
        _validate_json_schema(parsed, schema)
    return parsed


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object substring, or None.

    Scans from the first '{' to its matching '}', tracking string literals so
    braces inside string values do not unbalance the count.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped.removeprefix("```json").removesuffix("```").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped.removeprefix("```").removesuffix("```").strip()
    return stripped


def _json_schema_format(response_format: dict[str, Any]) -> dict[str, Any]:
    if "json_schema" in response_format:
        return response_format["json_schema"]
    if "schema" in response_format:
        return {
            "name": response_format.get("name", "structured_output"),
            "schema": response_format["schema"],
            "strict": response_format.get("strict", True),
        }
    return response_format


def _responses_json_schema_format(response_format: dict[str, Any]) -> dict[str, Any]:
    schema_format = _json_schema_format(response_format)
    if schema_format.get("type") == "json_schema":
        normalized = dict(schema_format)
        normalized["schema"] = _normalize_json_schema(
            schema_format.get("schema") or {},
            require_all_properties=True,
        )
        normalized["strict"] = True
        return normalized
    return {
        "type": "json_schema",
        "name": schema_format.get("name", "structured_output"),
        "schema": _normalize_json_schema(
            schema_format.get("schema", schema_format),
            require_all_properties=True,
        ),
        "strict": True,
    }


def _parse_validation_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    schema_format = _json_schema_format(response_format)
    if schema_format.get("type") == "json_schema":
        schema = schema_format.get("schema") or {}
    else:
        schema = schema_format.get("schema", schema_format)
    if (schema.get("type") == "object" or "properties" in schema) and not schema.get("properties"):
        return {}
    return _normalize_json_schema(schema, require_all_properties=False)


def _normalize_json_schema(schema: dict[str, Any], *, require_all_properties: bool) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    if normalized.get("type") == "object" or "properties" in normalized:
        properties = normalized.get("properties") or {}
        original_required = set(normalized.get("required") or [])
        normalized["properties"] = {
            key: _normalize_json_schema(value, require_all_properties=require_all_properties)
            for key, value in properties.items()
        }
        if require_all_properties:
            normalized["required"] = list(properties.keys())
            for key, value in normalized["properties"].items():
                if key not in original_required:
                    normalized["properties"][key] = _nullable_schema(value)
        elif original_required:
            normalized["required"] = list(normalized.get("required") or [])
            for key, value in normalized["properties"].items():
                if key not in original_required:
                    normalized["properties"][key] = _nullable_schema(value)
        elif "required" in normalized:
            normalized["required"] = []
            for key, value in normalized["properties"].items():
                normalized["properties"][key] = _nullable_schema(value)
        normalized["additionalProperties"] = False
    if normalized.get("type") == "array" and isinstance(normalized.get("items"), dict):
        normalized["items"] = _normalize_json_schema(
            normalized["items"],
            require_all_properties=require_all_properties,
        )
    return normalized


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    nullable = copy.deepcopy(schema)
    schema_type = nullable.get("type")
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            nullable["type"] = [*schema_type, "null"]
    elif schema_type:
        nullable["type"] = [schema_type, "null"]
    else:
        nullable["type"] = ["null"]
    if isinstance(nullable.get("enum"), list) and None not in nullable["enum"]:
        nullable["enum"] = [*nullable["enum"], None]
    return nullable


def _validate_json_schema(value: Any, schema: dict[str, Any]) -> None:
    """Validate a parsed LLM payload against `schema` via the `jsonschema` library.

    Raises `ValueError` (not `jsonschema.ValidationError`) so the retry-on-mismatch paths that catch
    `ValueError` are unchanged. Replaces a hand-rolled walker that silently ignored constraints such
    as numeric bounds, nested `oneOf`, and other keywords, degrading judge output without explanation.
    """
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as exc:
        location = "$" + "".join(f"[{part!r}]" if isinstance(part, str) else f"[{part}]" for part in exc.absolute_path)
        raise ValueError(f"LLM response does not match JSON Schema at {location}: {exc.message}") from exc


def _system_with_schema_instruction(system: str | None, response_format: dict[str, Any] | None) -> str | None:
    if not response_format:
        return system
    schema_format = _json_schema_format(response_format)
    instruction = (
        "Respond only with valid JSON matching the following JSON Schema. "
        "Do not add markdown, explanation, or text outside JSON.\n"
        f"JSON Schema:\n{json.dumps(schema_format, ensure_ascii=False)}"
    )
    if system:
        return f"{system}\n\n{instruction}"
    return instruction


def _message_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    text = str(content or "")
    return [{"type": "text", "text": text}] if text else []


def _block_text(block: dict[str, Any]) -> str:
    return str(block.get("text") if block.get("text") is not None else block.get("content") or "")


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content or "")
    return "\n".join(_block_text(block) for block in _message_blocks(content) if block.get("type") == "text")


def _tool_use_id(block: dict[str, Any]) -> str:
    return str(block.get("id") or block.get("tool_use_id") or block.get("tool_call_id") or "")


def _tool_input(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("input") if block.get("input") is not None else block.get("args")
    return dict(value) if isinstance(value, dict) else {}


def _tool_result_name(block: dict[str, Any]) -> str:
    return str(block.get("name") or "tool")


def _openai_messages(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    for item in messages:
        role = item["role"]
        content = item.get("content", "")
        if not isinstance(content, list):
            result.append({"role": role, "content": _content_text(content)})
            continue
        for block in _message_blocks(content):
            block_type = block.get("type")
            if block_type == "text":
                text = _block_text(block)
                if text:
                    result.append({"role": role, "content": text})
            elif block_type == "tool_use":
                result.append(
                    {
                        "type": "function_call",
                        "call_id": _tool_use_id(block),
                        "name": block.get("name") or "",
                        "arguments": json.dumps(_tool_input(block), ensure_ascii=False),
                    }
                )
            elif block_type == "tool_result":
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": _tool_use_id(block),
                        "output": _block_text(block),
                    }
                )
    return result


def _openai_chat_messages(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    for item in messages:
        role = _assistant_to_model_role(item["role"], "assistant")
        content = item.get("content", "")
        if not isinstance(content, list):
            result.append({"role": role, "content": _content_text(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in _message_blocks(content):
            block_type = block.get("type")
            if block_type == "text":
                text = _block_text(block)
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": _tool_use_id(block),
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": json.dumps(_tool_input(block), ensure_ascii=False),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_use_id(block),
                        "content": _block_text(block),
                    }
                )

        if tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                }
            )
            result.extend(tool_results)
        elif tool_results:
            result.extend(tool_results)
            if text_parts:
                result.append({"role": role, "content": "\n".join(text_parts)})
        else:
            result.append({"role": role, "content": "\n".join(text_parts)})
    return result


def _chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("text") is not None:
            text_parts.append(str(part["text"]))
        elif part.get("type") in {"text", "output_text"} and part.get("content") is not None:
            text_parts.append(str(part["content"]))
    return " ".join(p for p in text_parts if p).strip()


def _extract_openai_text(data: dict[str, Any]) -> str | None:
    if "output_text" in data:
        return data.get("output_text")

    output = data.get("output") or []
    for item in output:
        content = item.get("content") or []
        for part in content:
            if part.get("type") in {"output_text", "text"} and part.get("text") is not None:
                return part["text"]
    return None


def _extract_openai_chat_text(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return _chat_content_text(message.get("content"))


def _anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    if not isinstance(content, list):
        return _content_text(content)

    result: list[dict[str, Any]] = []
    for block in _message_blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            text = _block_text(block)
            if text:
                result.append({"type": "text", "text": text})
        elif block_type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": _tool_use_id(block),
                    "name": block.get("name") or "",
                    "input": _tool_input(block),
                }
            )
        elif block_type == "tool_result":
            result.append(
                {
                    "type": "tool_result",
                    "tool_use_id": _tool_use_id(block),
                    "content": _block_text(block),
                }
            )
    return result


def _anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": _assistant_to_model_role(item["role"], "assistant"),
            "content": _anthropic_content(item.get("content")),
        }
        for item in messages
    ]


def _anthropic_system_blocks(system: str) -> list[dict[str, Any]]:
    """Wrap the system prompt as a single cache-eligible block.

    `system` (get_instruction + artifact contract) is stable for the lifetime of a session, so
    marking it with an ephemeral cache breakpoint lets Anthropic serve it from cache on turn 2+
    instead of re-billing it as fresh input tokens every call.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _extract_anthropic_text(data: dict[str, Any]) -> str | None:
    content = data.get("content") or []
    for item in content:
        if item.get("text") is not None:
            return item["text"]
    return None


def _google_parts(content: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in _message_blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            text = _block_text(block)
            if text:
                result.append({"text": text})
        elif block_type == "tool_use":
            function_call = {"name": block.get("name") or "", "args": _tool_input(block)}
            call_id = _tool_use_id(block)
            if call_id:
                function_call["id"] = call_id
            result.append({"functionCall": function_call})
        elif block_type == "tool_result":
            function_response = {
                "name": _tool_result_name(block),
                "response": {"content": _block_text(block)},
            }
            call_id = _tool_use_id(block)
            if call_id:
                function_response["id"] = call_id
            result.append({"functionResponse": function_response})
    return result or [{"text": ""}]


def _google_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": _assistant_to_model_role(item["role"], "model"), "parts": _google_parts(item.get("content"))}
        for item in messages
    ]


def _extract_google_text(data: dict[str, Any]) -> str | None:
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return None
    return parts[0].get("text")


def _bedrock_content(content: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in _message_blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            text = _block_text(block)
            if text:
                result.append({"text": text})
        elif block_type == "tool_use":
            result.append(
                {
                    "toolUse": {
                        "toolUseId": _tool_use_id(block),
                        "name": block.get("name") or "",
                        "input": _tool_input(block),
                    }
                }
            )
        elif block_type == "tool_result":
            result.append(
                {
                    "toolResult": {
                        "toolUseId": _tool_use_id(block),
                        "content": [{"text": _block_text(block)}],
                    }
                }
            )
    return result or [{"text": ""}]


def _bedrock_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": _assistant_to_model_role(item["role"], "assistant"), "content": _bedrock_content(item.get("content"))}
        for item in messages
    ]


def _bedrock_system(system: str | None, response_format: dict[str, Any] | None) -> list[dict[str, str]]:
    final_system = _system_with_schema_instruction(system, response_format)
    if not final_system:
        return []
    return [{"text": final_system}]


def _assistant_to_model_role(role: str, assistant_role: str) -> str:
    if role == "assistant":
        return assistant_role
    return role
