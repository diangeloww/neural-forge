"""Basic chat completion example."""

import asyncio
from neural_forge import Forge


async def main():
    forge = Forge(
        providers={
            "openai": {"api_key": "sk-your-key-here"},
        },
    )

    response = await forge.chat(
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        max_tokens=500,
    )

    print(response.content)
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total}")
    print(f"Cost: ${response.cost:.4f}")


asyncio.run(main())
