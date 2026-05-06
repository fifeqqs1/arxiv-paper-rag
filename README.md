# MortyClaw Core

MortyClaw is a transparent, controllable, memory-enabled local Agent runtime built with LangGraph and LangChain. This repository currently keeps the core source code only, focused on task routing, planning, approval, execution, memory, runtime persistence, and terminal observability.

The public repository intentionally excludes local runtime artifacts, documentation assets, private configuration, and other non-core files.

It now contains two paired modules:

- `mortyclaw/`: the local Agent runtime and execution core
- `arxiv_rag/`: the companion paper RAG project used for retrieval, question answering, and paper-oriented workflows

## What This Repo Includes

- `mortyclaw/`: core Agent runtime, workflow nodes, tools, memory, storage, observability
- `arxiv_rag/`: restored companion RAG project snapshot, kept as a sibling top-level directory
- `entry/`: CLI entrypoints for `config`, `run`, `monitor`, `heartbeat`, and related commands
- `requirements.txt` and `setup.py`: minimal packaging and dependency definition
- `.env.example`: safe configuration template without secrets

## Core Capabilities

- LangGraph-based runtime graph with `router`, `planner`, `approval_gate`, `reviewer`, and final execution flow
- Fast / slow task routing for lightweight requests and multi-step high-risk work
- Approval-gated execution for file writes, shell commands, and other sensitive operations
- Layered memory with working memory, session memory, and long-term memory
- SQLite-backed runtime state for sessions, tasks, task runs, and inbox events
- Structured handoff summaries for long-context compression and interrupted-task recovery
- Project and office tool boundaries for safer local automation
- JSONL audit logs and terminal monitor for end-to-end observability

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Copy the example file and fill in your local values:

```bash
cp .env.example .env
```

The real `.env` file must stay local and should never be committed.

## CLI Usage

Configure model settings:

```bash
mortyclaw config
```

Start the interactive runtime:

```bash
mortyclaw run
mortyclaw run --new
mortyclaw run --thread-id local_geek_master
```

Monitor a session:

```bash
mortyclaw monitor --latest
mortyclaw monitor --thread-id local_geek_master
```

Run scheduled task delivery:

```bash
mortyclaw heartbeat
mortyclaw heartbeat --interval 5
```

## Repository Layout

```text
.
├── entry/
│   ├── cli.py
│   ├── main.py
│   └── monitor.py
├── arxiv_rag/
│   ├── src/
│   ├── scripts/
│   ├── airflow/
│   ├── eval/
│   └── pyproject.toml
├── mortyclaw/
│   └── core/
│       ├── agent/
│       ├── approval/
│       ├── context/
│       ├── memory/
│       ├── observability/
│       ├── planning/
│       ├── routing/
│       ├── runtime/
│       ├── storage/
│       └── tools/
├── .env.example
├── requirements.txt
└── setup.py
```

## Intentionally Excluded From This Repo

To keep the repository focused on reusable core code, the following content is intentionally not uploaded:

- `docs/`
- `.env` and any other secret-bearing local environment files
- `workspace/`
- `logs/` and archived runtime logs
- local SQLite runtime databases
- local office files, generated artifacts, and experimental data

## Notes

- This repository is meant to publish the core implementation only.
- If you want to run the system locally, use `.env.example` as the starting point and create your own local `.env`.
- Runtime data such as sessions, tasks, logs, and memory stores should remain outside version control.
