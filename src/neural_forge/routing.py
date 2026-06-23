"""Intelligent routing strategies for multi-provider orchestration."""

from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_forge.models import ChatRequest
    from neural_forge.providers.base import BaseProvider


class Strategy(str, Enum):
    COST_OPTIMIZED = "cost_optimized"
    LATENCY = "latency"
    QUALITY = "quality"
    ROUND_ROBIN = "round_robin"
    CUSTOM = "custom"


# Approximate cost per 1M tokens (input, output)
MODEL_COSTS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-3.5": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Quality scores (1-10)
MODEL_QUALITY = {
    "gpt-4o": 9,
    "gpt-4.1": 9,
    "o3-mini": 8,
    "claude-sonnet-4": 9,
    "gemini-2.5-pro": 9,
    "gemini-2.0-flash": 7,
    "gpt-4o-mini": 7,
    "claude-haiku-3.5": 7,
}


class Router:
    """Routes requests to the best provider based on strategy."""

    def __init__(
        self,
        strategy: Strategy | str = Strategy.COST_OPTIMIZED,
        fallback_chain: list[str] | None = None,
    ):
        self.strategy = Strategy(strategy) if isinstance(strategy, str) else strategy
        self.fallback_chain = fallback_chain or []
        self._latencies: dict[str, list[float]] = {}
        self._failures: dict[str, int] = {}
        self._round_robin_idx = 0

    def select(
        self, providers: dict[str, BaseProvider], request: ChatRequest
    ) -> BaseProvider | None:
        """Select the best provider based on routing strategy."""
        available = {
            name: p for name, p in providers.items()
            if self._failures.get(name, 0) < 3
        }
        if not available:
            # Reset failures and try all
            self._failures.clear()
            available = providers

        if not available:
            return None

        match self.strategy:
            case Strategy.ROUND_ROBIN:
                names = list(available.keys())
                provider = available[names[self._round_robin_idx % len(names)]]
                self._round_robin_idx += 1
                return provider

            case Strategy.LATENCY:
                return min(
                    available.values(),
                    key=lambda p: self._avg_latency(p.name),
                )

            case Strategy.QUALITY:
                return max(
                    available.values(),
                    key=lambda p: self._quality_score(p.name),
                )

            case Strategy.COST_OPTIMIZED:
                return min(
                    available.values(),
                    key=lambda p: self._cost_score(p.name, request.max_tokens),
                )

            case _:
                return random.choice(list(available.values()))

    def record_failure(self, provider_name: str) -> None:
        self._failures[provider_name] = self._failures.get(provider_name, 0) + 1

    def record_latency(self, provider_name: str, latency_ms: float) -> None:
        if provider_name not in self._latencies:
            self._latencies[provider_name] = []
        self._latencies[provider_name].append(latency_ms)
        # Keep rolling window
        self._latencies[provider_name] = self._latencies[provider_name][-100:]

    def _avg_latency(self, provider_name: str) -> float:
        latencies = self._latencies.get(provider_name, [])
        return sum(latencies) / len(latencies) if latencies else 5000.0

    def _quality_score(self, provider_name: str) -> float:
        return MODEL_QUALITY.get(provider_name, 5)

    def _cost_score(self, provider_name: str, max_tokens: int) -> float:
        costs = MODEL_COSTS.get(provider_name, (1.0, 5.0))
        return costs[0] * (max_tokens / 1_000_000) + costs[1] * (max_tokens / 1_000_000)
