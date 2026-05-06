# arXiv Paper RAG

An end-to-end RAG system for collecting, indexing, searching, and explaining arXiv papers. The project combines FastAPI, PostgreSQL, OpenSearch, Redis, Airflow, LangGraph, Langfuse, Ollama/Qwen-compatible generation, Gradio, Telegram, and Feishu integrations.

## Features

- Automated arXiv paper ingestion with PDF download and parsing
- PostgreSQL storage for paper metadata and parsed content
- OpenSearch BM25 and hybrid retrieval
- Chunking and indexing pipeline for long academic papers
- RAG answer generation with source citations
- Agentic RAG workflow with query rewriting, guardrails, grading, and answer generation
- Redis exact-match caching
- Langfuse tracing for observability
- Gradio web UI
- Telegram and Feishu bot integrations

## Project Layout

```text
.
├── airflow/                 # Airflow DAGs and runtime image
├── scripts/                 # Local ingestion helpers
├── src/                     # API, services, schemas, and app code
├── tests/                   # Unit, API, and integration tests
├── compose.yml              # Local service stack
├── Dockerfile               # API image
├── Makefile                 # Common development commands
└── pyproject.toml           # Python dependencies and tooling
```

`notebooks/`, local documentation assets, `.env`, local data, OpenSearch volumes, and Ollama model data are intentionally ignored and are not uploaded to Git.

## Requirements

- Python 3.12
- Docker and Docker Compose
- uv package manager
- Ollama or a Qwen-compatible API endpoint
- Optional API keys for Jina, Langfuse, Telegram, and Feishu

## Setup

Copy the example environment file and fill in local values:

```bash
cp .env.example .env
```

Install dependencies:

```bash
uv sync
```

Start the local stack:

```bash
docker compose up -d
```

Run database and service checks:

```bash
make test
```

## Ingest Papers

Fetch recent papers and index them:

```bash
uv run python scripts/ingest_recent_papers.py --max-results 5
```

You can also search by a custom arXiv query:

```bash
uv run python scripts/ingest_recent_papers.py --search-query "all:UAV OR all:drone" --max-results 5
```

## API

Start the API locally:

```bash
uv run fastapi dev src/main.py
```

Useful endpoints:

```text
GET  /health
POST /api/v1/search
POST /api/v1/hybrid-search/
POST /api/v1/ask
POST /api/v1/ask-agentic
POST /api/v1/stream
```

Example RAG request:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain the interception method in arXiv:2603.16279v1","top_k":5,"use_hybrid":true}'
```

## Web UI

Run the Gradio interface:

```bash
uv run python gradio_launcher.py
```

## Airflow

Airflow is included for scheduled ingestion. After the Docker stack starts, open:

```text
http://localhost:8080
```

The main DAG is `arxiv_paper_ingestion`.

## Safety Notes

- `.env` is ignored and should hold real secrets only locally.
- `ollama_data/` is ignored so local model weights are not committed.
- `opensearch_data/` and `data/` are ignored to avoid uploading generated indexes, PDFs, and parsed data.
- `notebooks/` is ignored and kept local only.

## Development

Run tests:

```bash
uv run pytest
```

Run lint checks:

```bash
uv run ruff check src tests
```
