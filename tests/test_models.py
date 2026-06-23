"""Tests for data models."""

from neural_forge.models import Message, Usage, ChatResponse


def test_message_to_dict():
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_message_with_name():
    msg = Message(role="user", content="hi", name="test_user")
    d = msg.to_dict()
    assert d["name"] == "test_user"


def test_usage_total():
    usage = Usage(prompt_tokens=100, completion_tokens=50)
    assert usage.total == 150


def test_chat_response():
    resp = ChatResponse(
        content="Hello!",
        model="gpt-4o",
        provider="openai",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    assert resp.content == "Hello!"
    assert resp.usage.total == 15
