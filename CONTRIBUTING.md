# Contributing to Neural Forge

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/diangeloww/neural-forge.git
cd neural-forge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=neural_forge  # with coverage
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Guidelines

- Write tests for new features
- Follow existing code patterns
- Update documentation as needed
- Keep PRs focused and small
