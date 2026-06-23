"""Data models for Neural Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: str
    content: str | list[dict] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatRequest:
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False
    response_format: dict | None = None
    tools: list[dict] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    content: str
    model: str
    provider: str
    usage: Usage
    cost: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    delta: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: Usage | None = None
    cost: float = 0.0
