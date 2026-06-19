import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from app.models.llm_provider import ProviderType


DEFAULT_MODEL_BY_PROVIDER = {
    ProviderType.BEDROCK: "amazon.nova-lite-v1:0",
    ProviderType.OPENAI: "gpt-4o-mini",
    ProviderType.GOOGLE: "gemini-1.5-flash",
    ProviderType.ANTHROPIC: "claude-3-5-haiku-20241022",
}


@dataclass(frozen=True)
class LLMClientConfig:
    api_key: str
    model: str
    region: str | None = None
    secret_key: str | None = None


class LLMClient(Protocol):
    async def ping(self) -> str | None:
        pass

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        pass


class OpenAILLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"model": self.config.model, "input": "ping", "max_output_tokens": 5},
            )
            response.raise_for_status()
            data = response.json()
        return data.get("output_text")

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        import httpx

        input_messages = _openai_messages(messages, system)
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": _json_schema_format(response_format),
            }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        text = _extract_openai_text(data)
        return _parse_generate_text(text, response_format)


class GoogleLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent",
                params={"key": self.config.api_key},
                json={"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 5}},
            )
            response.raise_for_status()
            data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        return parts[0].get("text")

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        import httpx

        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if response_format:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_format.get("schema", response_format)

        body: dict[str, Any] = {
            "contents": _google_contents(messages),
            "generationConfig": generation_config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent",
                params={"key": self.config.api_key},
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        text = _extract_google_text(data)
        return _parse_generate_text(text, response_format)


class AnthropicLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
                json={"model": self.config.model, "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]},
            )
            response.raise_for_status()
            data = response.json()
        content = data.get("content") or []
        if not content:
            return None
        return content[0].get("text")

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        import httpx

        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": _anthropic_messages(messages),
        }
        final_system = _system_with_schema_instruction(system, response_format)
        if final_system:
            body["system"] = final_system

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        text = _extract_anthropic_text(data)
        return _parse_generate_text(text, response_format)


class BedrockLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        if self.config.secret_key:
            return await self._ping_with_iam_keys()
        return await self._ping_with_api_key()

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        if self.config.secret_key:
            text = await self._generate_with_iam_keys(messages, system, max_tokens, response_format)
        else:
            text = await self._generate_with_api_key(messages, system, max_tokens, response_format)
        return _parse_generate_text(text, response_format)

    async def _ping_with_iam_keys(self) -> str | None:
        def _ping() -> str | None:
            import boto3

            client_kwargs: dict[str, Any] = {
                "region_name": self.config.region or "us-east-1",
                "aws_access_key_id": self.config.api_key,
                "aws_secret_access_key": self.config.secret_key,
            }
            client = boto3.client("bedrock-runtime", **client_kwargs)
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

    async def _generate_with_iam_keys(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | None:
        def _generate() -> str | None:
            import boto3

            client_kwargs: dict[str, Any] = {
                "region_name": self.config.region or "us-east-1",
                "aws_access_key_id": self.config.api_key,
                "aws_secret_access_key": self.config.secret_key,
            }
            client = boto3.client("bedrock-runtime", **client_kwargs)
            converse_kwargs: dict[str, Any] = {
                "modelId": self.config.model,
                "messages": _bedrock_messages(messages),
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
            }
            bedrock_system = _bedrock_system(system, response_format)
            if bedrock_system:
                converse_kwargs["system"] = bedrock_system
            response = client.converse(**converse_kwargs)
            return _extract_bedrock_text(response)

        return await asyncio.to_thread(_generate)

    async def _generate_with_api_key(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str | None:
        import httpx

        region = self.config.region or "us-east-1"
        model_id = quote(self.config.model, safe="")
        body: dict[str, Any] = {
            "messages": _bedrock_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
        }
        bedrock_system = _bedrock_system(system, response_format)
        if bedrock_system:
            body["system"] = bedrock_system

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        return _extract_bedrock_text(data)


class LLMClientFactory:
    _client_classes = {
        ProviderType.BEDROCK: BedrockLLMClient,
        ProviderType.OPENAI: OpenAILLMClient,
        ProviderType.GOOGLE: GoogleLLMClient,
        ProviderType.ANTHROPIC: AnthropicLLMClient,
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
    ) -> LLMClient:
        client_class = cls._client_classes[provider_type]
        return client_class(
            LLMClientConfig(
                api_key=api_key,
                secret_key=secret_key,
                region=region,
                model=model or DEFAULT_MODEL_BY_PROVIDER[provider_type],
            )
        )


def _extract_bedrock_text(data: dict[str, Any]) -> str | None:
    content = data.get("output", {}).get("message", {}).get("content") or []
    if not content:
        return None
    return content[0].get("text")


def _parse_generate_text(text: str | None, response_format: dict[str, Any] | None) -> str | dict[str, Any]:
    text = text or ""
    if not response_format:
        return text

    try:
        parsed = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Không parse được JSON từ phản hồi LLM") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Không parse được JSON object từ phản hồi LLM")
    return parsed


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


def _system_with_schema_instruction(system: str | None, response_format: dict[str, Any] | None) -> str | None:
    if not response_format:
        return system
    schema_format = _json_schema_format(response_format)
    instruction = (
        "Trả lời duy nhất bằng JSON hợp lệ khớp JSON Schema sau. "
        "Không thêm markdown, giải thích, hoặc văn bản ngoài JSON.\n"
        f"JSON Schema:\n{json.dumps(schema_format, ensure_ascii=False)}"
    )
    if system:
        return f"{system}\n\n{instruction}"
    return instruction


def _openai_messages(messages: list[dict[str, str]], system: str | None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend({"role": item["role"], "content": item["content"]} for item in messages)
    return result


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


def _anthropic_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": _assistant_to_model_role(item["role"], "assistant"), "content": item["content"]} for item in messages]


def _extract_anthropic_text(data: dict[str, Any]) -> str | None:
    content = data.get("content") or []
    for item in content:
        if item.get("text") is not None:
            return item["text"]
    return None


def _google_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"role": _assistant_to_model_role(item["role"], "model"), "parts": [{"text": item["content"]}]}
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


def _bedrock_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"role": _assistant_to_model_role(item["role"], "assistant"), "content": [{"text": item["content"]}]}
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
