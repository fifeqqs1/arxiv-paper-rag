# Airflow

This directory contains the scheduled ingestion workflow for the arXiv Paper RAG project.

## DAGs

- `hello_world_dag.py`: lightweight Airflow health check
- `arxiv_paper_ingestion.py`: scheduled arXiv ingestion pipeline
- `dags/arxiv_ingestion/`: reusable ingestion task modules

## Pipeline

The ingestion DAG:

1. Checks service connectivity
2. Fetches recent arXiv papers
3. Downloads and parses PDFs
4. Stores paper metadata and parsed content in PostgreSQL
5. Indexes chunks into OpenSearch
6. Cleans up generated runtime files
7. Emits a processing report

## Local Usage

Start the full stack from the repository root:

```bash
docker compose up -d
```

Open Airflow:

```text
http://localhost:8080
```

The container reads the project source through Docker Compose mounts and shares the same PostgreSQL/OpenSearch services as the API.

## Runtime Data

Airflow logs and generated runtime files are ignored by Git. They stay local and should not be committed.
