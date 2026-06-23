"""Command-line interface for Neural Forge."""

import asyncio
import sys

from neural_forge import Forge


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: neural-forge <prompt>")
        print("       neural-forge --providers")
        sys.exit(1)

    if sys.argv[1] == "--providers":
        print("Available providers: openai, anthropic, google, custom")
        return

    prompt = " ".join(sys.argv[1:])
    forge = Forge(providers={})
    print(f"Prompt: {prompt}")
    print("(Configure providers in neural_forge.yaml to enable completions)")


if __name__ == "__main__":
    main()
