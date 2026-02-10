# AgentForge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Co-evolutionary framework for training robust AI agents via self-play. AgentForge automatically discovers agent weaknesses, generates harder scenarios, and tracks improvement — all running 100% locally on your Mac.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/agentforge.git
cd agentforge

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Run the demo
python demo.py --model Qwen/Qwen2.5-0.5B-Instruct
```

The model (~1GB) will be downloaded from Hugging Face on first run.

<!-- ![AgentForge Demo](docs/demo.gif) -->

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  YOUR MAC (16GB)                                     │
│                                                       │
│  ┌──────────────┐         ┌──────────────────┐       │
│  │ HuggingFace  │◄───────►│  AgentForge      │       │
│  │ Qwen2.5-0.5B │ local   │                  │       │
│  │ (~1GB RAM)   │ inference│  demo.py         │       │
│  └──────────────┘         │  ├─ Agent        │       │
│                            │  ├─ Environment  │       │
│                            │  ├─ Analyzer     │       │
│                            │  └─ Generator    │       │
│                            └──────────────────┘       │
└─────────────────────────────────────────────────────┘
```

Everything runs locally. No API keys. No cloud. No cost.

## Features

- **Local LLM Agent**: Runs Qwen2.5-0.5B-Instruct via HuggingFace Transformers
- **Simulation Environments**: YAML-configurable scenarios with mock tools
- **Failure Analysis**: Automatic diagnosis of agent weaknesses
- **Scenario Generation**: LLM-powered generation of harder test cases
- **Co-Evolutionary Loop**: Iterative evaluate → analyze → generate → re-evaluate
- **Verifiable Rewards**: Structured reward signals for agent evaluation
- **Two Domains**: Customer support and code review out of the box

## Usage

### Run the customer support demo

```bash
python demo.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Run the code review demo

```bash
python demo_code_review.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Run the full co-evolutionary loop

```bash
python run_forge_local.py --model Qwen/Qwen2.5-0.5B-Instruct --rounds 3
```

### Run tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
agentforge/
├── agentforge/
│   ├── __init__.py
│   ├── core.py           # Main co-evolutionary loop
│   ├── analyzer.py       # LLM failure analysis
│   ├── generator.py      # Scenario generation
│   ├── curriculum.py     # Training curriculum strategies
│   ├── environment.py    # Simulation environment
│   ├── rewards.py        # Verifiable reward functions
│   ├── local_agent.py    # HuggingFace-powered agent
│   └── cli.py            # CLI
├── configs/
│   ├── customer_support.yaml
│   └── code_review.yaml
├── tests/
│   └── test_local_agent.py
├── demo.py
├── demo_code_review.py
├── run_forge_local.py
├── pyproject.toml
└── README.md
```

## Adding Your Own Domain

Create a YAML config in `configs/` with tools and scenarios:

```yaml
tools:
  - name: my_tool
    description: "What the tool does"
    parameters:
      param1: "string - description"
    mock_responses:
      default:
        result: "mock data"

scenarios:
  - id: "scenario_1"
    description: "What this tests"
    user_message: "What the user says"
    difficulty: "easy"
    expected_tool_calls: ["my_tool"]
```

Then run:

```bash
python demo.py --config configs/your_domain.yaml
```

## License

Apache 2.0
