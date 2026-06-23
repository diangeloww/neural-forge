"""Event system for observability hooks."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable


class EventEmitter:
    """Async event emitter for request lifecycle hooks."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str):
        """Decorator to register an event handler.

        Usage:
            @forge.on("request")
            async def log(event):
                print(event)
        """
        def decorator(func: Callable[..., Awaitable]):
            self._handlers[event].append(func)
            return func
        return decorator

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._handlers.get(event, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Don't let handler errors break the pipeline
