"""Core Forge class - the main entry point."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from neural_forge.models import ChatRequest, ChatResponse, Message, StreamChunk
from neural_forge.providers.base import BaseProvider
from neural_forge.providers.openai_provider import OpenAIProvider
from neural_forge.routing import Router, Strategy
from neural_forge.hooks import EventEmitter


class Forge:
    """Multi-model AI agent with intelligent routing.

    Example:
        forge = Forge(
            providers={"openai": {"api_key": "sk-..."}},
            routing_strategy=Strategy.COST_OPTIMIZED,
        )
        response = await forge.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        providers: dict[str, dict],
        routing_strategy: Strategy | str = Strategy.COST_OPTIMIZED,
        fallback_chain: list[str] | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self._providers: dict[str, BaseProvider] = {}
        self._router = Router(strategy=routing_strategy, fallback_chain=fallback_chain)
        self._hooks = EventEmitter()
        self._max_retries = max_retries
        self._timeout = timeout

        for name, config in providers.items():
            self._providers[name] = self._create_provider(name, config)

    def _create_provider(self, name: str, config: dict) -> BaseProvider:
        """Factory method for provider instantiation."""
        provider_map = {
            "openai": OpenAIProvider,
        }
        cls = provider_map.get(name)
        if cls is None:
            # Default to OpenAI-compatible for custom providers
            cls = OpenAIProvider
        return cls(name=name, **config)

    async def chat(
        self,
        messages: list[dict | Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request with automatic routing."""
        request = ChatRequest(
            messages=[Message(**m) if isinstance(m, dict) else m for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        provider_name = model.split("/")[0] if model and "/" in model else None
        model_name = model.split("/", 1)[1] if model and "/" in model else model

        last_error = None
        for attempt in range(self._max_retries):
            if provider_name:
                provider = self._providers.get(provider_name)
            else:
                provider = self._router.select(self._providers, request)

            if provider is None:
                raise ValueError("No available provider")

            try:
                start = time.monotonic()
                response = await asyncio.wait_for(
                    provider.complete(request, model=model_name),
                    timeout=self._timeout,
                )
                response.latency_ms = (time.monotonic() - start) * 1000

                await self._hooks.emit("request", {
                    "provider": provider.name,
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "cost": response.cost,
                    "tokens": response.usage.total,
                })
                return response

            except Exception as e:
                last_error = e
                self._router.record_failure(provider.name if provider else "unknown")
                await self._hooks.emit("error", {
                    "provider": provider.name if provider else "unknown",
                    "error": str(e),
                    "attempt": attempt + 1,
                    "is_rate_limit": "429" in str(e) or "rate" in str(e).lower(),
                })

        raise last_error or RuntimeError("All retries exhausted")

    async def stream(
        self,
        messages: list[dict | Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion with backpressure support."""
        request = ChatRequest(
            messages=[Message(**m) if isinstance(m, dict) else m for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        provider_name = model.split("/")[0] if model and "/" in model else None
        model_name = model.split("/", 1)[1] if model and "/" in model else model

        if provider_name:
            provider = self._providers.get(provider_name)
        else:
            provider = self._router.select(self._providers, request)

        if provider is None:
            raise ValueError("No available provider")

        async for chunk in provider.stream(request, model=model_name):
            yield chunk

    def on(self, event: str):
        """Decorator to register event handlers."""
        return self._hooks.on(event)

    @property
    def providers(self) -> dict[str, str]:
        """List available providers and their status."""
        return {name: p.status for name, p in self._providers.items()}
