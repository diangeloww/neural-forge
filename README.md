<div align="center">

# Neural Forge

### Production-Grade Multi-Model AI Agent Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/diangeloww/neural-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/diangeloww/neural-forge/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff)

Intelligent routing across multiple LLM providers with structured output, streaming, evaluation pipelines, and built-in observability.

[Quick Start](#quick-start) · [Architecture](#architecture) · [Providers](#supported-providers) · [Examples](#examples) · [API Reference](#api-reference)

</div>

---

## Why Neural Forge?

| Feature | Neural Forge | Raw SDKs | LangChain |
|---|---|---|---|
| Multi-provider routing | Built-in | Manual | Plugin-based |
| Structured output (JSON Schema) | Native | Partial | Chain-based |
| Streaming with backpressure | Async-native | Basic | Callback |
| Token cost tracking | Per-request | No | Optional |
| Automatic retry + fallback | Configurable | No | Agent-level |
| Evaluation pipeline | Built-in | No | Separate lib |
| Observability hooks | Event-driven | No | Callback |

## Quick Start

```bash
pip install neural-forge
```

```python
from neural_forge import Forge

# Initialize with multiple providers
forge = Forge(
    providers={
        "openai": {"api_key": "sk-..."},
        "anthropic": {"api_key": "sk-ant-..."},
        "google": {"api_key": "..."},
    },
    routing_strategy="cost_optimized",  # or "latency", "quality", "round_robin"
)

# Single call with automatic provider selection
response = await forge.chat(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=1000,
)

print(response.content)
print(f"Provider: {response.provider} | Model: {response.model}")
print(f"Tokens: {response.usage.total} | Cost: ${response.cost:.4f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Neural Forge                      │
├─────────────┬──────────────┬───────────────────────┤
│   Router    │   Executor   │    Observability      │
│             │              │                       │
│  Strategy   │  Streaming   │  Token Tracking       │
│  Selection  │  Backpress.  │  Latency Metrics      │
│  Fallback   │  Parallel    │  Cost Calculation     │
│  Scoring    │  Batch       │  Event Hooks          │
├─────────────┴──────────────┴───────────────────────┤
│              Provider Abstraction                   │
├──────────┬──────────┬──────────┬───────────────────┤
│  OpenAI  │ Anthropic│  Google  │  Custom (OAI-comp) │
│  GPT-4o  │ Claude   │  Gemini  │  Any endpoint      │
│  o3-mini │ Haiku    │  Flash   │                    │
└──────────┴──────────┴──────────┴───────────────────┘
```

## Supported Providers

| Provider | Models | Streaming | Structured Output | Vision |
|---|---|---|---|---|
| OpenAI | GPT-4o, o3-mini, GPT-4.1 | Yes | JSON Schema | Yes |
| Anthropic | Claude Sonnet 4, Haiku 3.5 | Yes | Tool Use | Yes |
| Google | Gemini 2.5 Pro, Flash | Yes | Response Schema | Yes |
| Custom | Any OpenAI-compatible | Yes | Partial | Varies |

## Routing Strategies

```python
from neural_forge.routing import Strategy

forge = Forge(
    providers=providers,
    routing_strategy=Strategy.COST_OPTIMIZED,
    # Options:
    #   COST_OPTIMIZED  - cheapest model that meets quality threshold
    #   LATENCY         - fastest provider based on rolling average
    #   QUALITY         - highest capability model first
    #   ROUND_ROBIN     - distribute load evenly
    #   CUSTOM          - your own scoring function
    fallback_chain=["openai", "anthropic", "google"],
)
```

## Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: float
    topics: list[str]
    summary: str

result = await forge.structured(
    messages=[{"role": "user", "content": "Analyze this review: ..."}],
    schema=Analysis,
)

print(result.parsed.sentiment)  # 0.85
print(result.parsed.topics)     # ["customer_service", "product_quality"]
```

## Streaming

```python
async for chunk in forge.stream(
    messages=[{"role": "user", "content": "Write a haiku"}],
):
    print(chunk.delta, end="", flush=True)

print(f"\nCost: ${chunk.cost:.4f}")
```

## Evaluation Pipeline

```python
from neural_forge.eval import EvalSuite, TestCase

suite = EvalSuite("my-benchmark", [
    TestCase(
        input="What is 2+2?",
        expected="4",
        rubric="Must give exact numeric answer",
    ),
    TestCase(
        input="Explain recursion",
        rubric="Clear, uses analogies, includes base case",
    ),
])

report = await suite.run(forge, models=["openai/gpt-4o", "anthropic/claude-sonnet-4"])
report.print_table()
# ┌─────────────────┬────────┬──────────┬──────────┬────────┐
# │ Model           │ Pass   │ Avg Cost │ Avg Time │ Score  │
# ├─────────────────┼────────┼──────────┼──────────┼────────┤
# │ gpt-4o          │ 2/2    │ $0.003   │ 1.2s     │ 9.5/10 │
# │ claude-sonnet-4 │ 2/2    │ $0.002   │ 0.9s     │ 9.2/10 │
# └─────────────────┴────────┴──────────┴──────────┴────────┘
```

## Observability

```python
from neural_forge.hooks import on_request, on_error

@on_request
async def log_request(event):
    print(f"[{event.provider}] {event.model} - {event.latency_ms}ms - ${event.cost}")

@on_error
async def handle_error(event):
    if event.is_rate_limit:
        await event.retry_with_backoff()
```

## Examples

| Example | Description |
|---|---|
| [basic_chat.py](examples/basic_chat.py) | Simple chat completion |
| [multi_provider.py](examples/multi_provider.py) | Route across providers |
| [structured_output.py](examples/structured_output.py) | Pydantic schema extraction |
| [streaming.py](examples/streaming.py) | Async streaming with backpressure |
| [evaluation.py](examples/evaluation.py) | Run eval suite and compare models |
| [cost_tracking.py](examples/cost_tracking.py) | Track and limit spending |

## API Reference

Full documentation: [docs/](docs/)

```
GET  /health          - Service health check
POST /chat            - Chat completion (OpenAI-compatible)
POST /chat/stream     - Streaming chat completion
POST /evaluate        - Run evaluation suite
GET  /metrics         - Prometheus metrics
GET  /providers       - List available providers and status
```

## Development

```bash
git clone https://github.com/diangeloww/neural-forge.git
cd neural-forge
pip install -e ".[dev]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with care by [diangeloww](https://github.com/diangeloww)**

</div>
