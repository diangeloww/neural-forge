"""Base provider interface."""

from __future__ import annotations

import abc
from typing import AsyncIterator

from neural_forge.models import ChatRequest, ChatResponse, StreamChunk


class BaseProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, **config):
        self.name = name
        self.config = config
        self._status = "ready"

    @property
    def status(self) -> str:
        return self._status

    @abc.abstractmethod
    async def complete(self, request: ChatRequest, model: str | None = None) -> ChatResponse:
        """Send a non-streaming completion request."""
        ...

    @abc.abstractmethod
    async def stream(
        self, request: ChatRequest, model: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming completion request."""
        ...

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for the given token counts."""
        return 0.0
