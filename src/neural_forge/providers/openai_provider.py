"""OpenAI-compatible provider implementation."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from neural_forge.models import ChatRequest, ChatResponse, StreamChunk, Usage
from neural_forge.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI and OpenAI-compatible API provider."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, name: str, api_key: str, base_url: str | None = None, **kwargs):
        super().__init__(name, **kwargs)
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def complete(self, request: ChatRequest, model: str | None = None) -> ChatResponse:
        payload = self._build_payload(request, model, stream=False)
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_response(data)

    async def stream(
        self, request: ChatRequest, model: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(request, model, stream=True)
        model_name = model or self.DEFAULT_MODEL

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    return
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                yield StreamChunk(
                    delta=delta.get("content", ""),
                    model=model_name,
                    provider=self.name,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )

    def _build_payload(self, request: ChatRequest, model: str | None, stream: bool) -> dict:
        payload = {
            "model": model or self.DEFAULT_MODEL,
            "messages": [m.to_dict() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        return payload

    def _parse_response(self, data: dict) -> ChatResponse:
        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        return ChatResponse(
            content=choice["message"].get("content", ""),
            model=data["model"],
            provider=self.name,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def close(self):
        await self._client.aclose()
