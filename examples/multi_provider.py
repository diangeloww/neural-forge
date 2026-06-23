"""Multi-provider routing example."""

import asyncio
from neural_forge import Forge
from neural_forge.routing import Strategy


async def main():
    forge = Forge(
        providers={
            "openai": {"api_key": "sk-..."},
            "anthropic": {"api_key": "sk-ant-..."},
            "google": {"api_key": "..."},
        },
        routing_strategy=Strategy.COST_OPTIMIZED,
        fallback_chain=["openai", "anthropic", "google"],
    )

    # Automatic routing
    response = await forge.chat(
        messages=[{"role": "user", "content": "Explain transformers in ML"}],
    )
    print(f"Routed to: {response.provider}/{response.model}")

    # Force a specific provider
    response = await forge.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="anthropic/claude-sonnet-4",
    )
    print(f"Forced to: {response.provider}/{response.model}")


asyncio.run(main())
