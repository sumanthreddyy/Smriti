# Contributing to Smriti

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/sumanthreddyy/Smriti.git
cd Smriti
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check smriti/ tests/
ruff format --check smriti/ tests/
```

## Making Changes

1. Fork the repo and create a branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Keep commits focused — one logical change per commit.
5. Open a PR with a clear description of what changed and why.

## Code Style

- Python 3.10+ — use `X | Y` union syntax, not `Union[X, Y]`.
- Type hints on public API methods.
- No unnecessary abstractions — keep it simple.

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
