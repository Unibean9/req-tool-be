import asyncio
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


class BedrockLLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    async def ping(self) -> str | None:
        if self.config.secret_key:
            return await self._ping_with_iam_keys()
        return await self._ping_with_api_key()

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
