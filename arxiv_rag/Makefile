.PHONY: help start stop restart status logs health setup format lint test test-cov start-feishu-bot run-feishu-bot ingest-sample-papers clean

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Service management
start: ## Start all services
	docker compose up --build -d

stop: ## Stop all services
	docker compose down

restart: ## Restart all services
	docker compose restart

status: ## Show service status
	docker compose ps

logs: ## Show service logs
	docker compose logs -f

# Health checks
health: ## Check all services health
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health | jq . || echo "API not responding"
	@curl -s http://localhost:9200/_cluster/health | jq . || echo "OpenSearch not responding"
	@curl -s http://localhost:8080/api/v2/monitor/health || echo "Airflow not responding"
	@curl -s http://localhost:11435/api/version | jq . || echo "Ollama not responding"

# Development
setup: ## Install Python dependencies
	uv sync

format: ## Format code
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

start-feishu-bot: ## Start the Feishu bot container profile
	docker-compose --profile feishu up -d --build feishu-bot

run-feishu-bot: ## Run the standalone Feishu bot locally
	uv run python -m src.services.feishu.runner

ingest-sample-papers: ## One-off arXiv ingestion (set DATE=YYYYMMDD or FROM_DATE=... TO_DATE=... MAX_RESULTS=5)
	DEBUG=true uv run python scripts/ingest_recent_papers.py $(if $(SEARCH_QUERY),--search-query '$(SEARCH_QUERY)',) $(if $(DATE),--target-date $(DATE),) $(if $(FROM_DATE),--from-date $(FROM_DATE),) $(if $(TO_DATE),--to-date $(TO_DATE),) $(if $(MAX_RESULTS),--max-results $(MAX_RESULTS),)

# Cleanup
clean: ## Clean up everything
	docker compose down -v
	docker system prune -f
