# Contributing to AgentForge

Thank you for your interest in contributing to AgentForge!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/agentforge.git`
3. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
4. Install in dev mode: `pip install -e ".[dev]"` or `pip install -e . && pip install pytest`
5. Run tests: `python -m pytest tests/ -v`

## Making Changes

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality
4. Run tests: `python -m pytest tests/ -v`
5. Commit with a clear message
6. Push and open a pull request

## Adding a New Domain

The easiest way to contribute is by adding a new simulation domain:

1. Create a YAML config in `configs/` (see `customer_support.yaml` for the format)
2. Define tools with mock responses
3. Define scenarios with varying difficulty (easy/medium/hard)
4. Add expected tool calls and success criteria
5. Create a demo script (e.g., `demo_your_domain.py`)
6. Add tests

## Code Style

- Use type hints
- Keep functions focused and small
- Add docstrings to public classes and functions

## Reporting Issues

Open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, model used)
