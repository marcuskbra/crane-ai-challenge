# Makefile for crane-challenge
# Modern Python project with Clean Architecture

.PHONY: help install dev-install test lint lint-fix format format-fix type-check coverage validate clean run \
	test-all test-unit test-integration test-fast api-dev api-prod api-test api-docs api-health \
	ui-install ui-dev ui-build ui-clean ui-lint ui-test \
	llm-docker-up llm-docker-down llm-docker-logs llm-docker-test llm-docker-clean \
	llm-local-setup llm-local-pull llm-local-pull-fast llm-local-start llm-local-stop llm-local-test \
	llm-check llm-status llm-models

# ============================================================================
# Help & Documentation
# ============================================================================

help: ## Show this help message
	@echo "crane-challenge - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@grep -E '^(install|dev-install):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^(run|test|lint|format|type-check|validate):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "API Development:"
	@grep -E '^(api-.*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Frontend UI (Visualization):"
	@grep -E '^(ui-.*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Testing:"
	@grep -E '^(test[^-]|test-all|test-integration|test-fast|coverage):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Local LLM Testing:"
	@grep -E '^(llm-.*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Cleanup:"
	@grep -E '^(clean.*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Setup & Installation
# ============================================================================

install: ## Install production dependencies
	uv sync --no-dev

dev-install: ## Install all dependencies including dev and test extras
	uv sync --all-extras
	@echo "Installing tox globally for convenience..."
	uv tool install tox --with tox-uv
	@echo "Installing pre-commit hooks..."
	pre-commit install

# ============================================================================
# Development & Running
# ============================================================================

run: ## Run the application
	uv run python -m challenge

# ============================================================================
# API Development
# ============================================================================

api-dev: ## Run the API server in development mode (auto-reload)
	uv run uvicorn challenge.presentation.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Run the API server in production mode
	uv run uvicorn challenge.presentation.main:app --host 0.0.0.0 --port 8000 --workers 4

api-test: ## Run API tests only
	$(PYTEST_ENV) uv run pytest tests/unit/presentation/api/ -xvs

api-health: ## Check API health endpoint
	@echo "Checking API health..."
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool || echo "API is not running. Start with 'make api-dev'"

api-docs: ## Open API documentation in browser
	@echo "Opening API docs at http://localhost:8000/api/docs"
	@python -m webbrowser http://localhost:8000/api/docs || open http://localhost:8000/api/docs || xdg-open http://localhost:8000/api/docs

# ============================================================================
# Frontend UI (Visualization Tool)
# ============================================================================

ui-install: ## Install frontend dependencies
	@echo "📦 Installing frontend dependencies..."
	cd ui-react && npm install
	@echo "✅ Frontend dependencies installed!"

ui-dev: ## Run frontend development server
	@echo "🚀 Starting frontend development server..."
	@echo "📝 Note: This is a visualization tool, not production-ready"
	@echo "🌐 Frontend will be available at http://localhost:3000"
	cd ui-react && npm run dev

ui-build: ## Build frontend for production (visualization only)
	@echo "🏗️  Building frontend..."
	cd ui-react && npm run build
	@echo "✅ Frontend build complete (dist/)"

ui-clean: ## Clean frontend build artifacts and dependencies
	@echo "🧹 Cleaning frontend files..."
	cd ui-react && rm -rf node_modules dist .vite
	@echo "✅ Frontend cleanup complete!"

ui-lint: ## Run frontend linting
	@echo "🔍 Linting frontend code..."
	cd ui-react && npm run lint || echo "⚠️  Linting issues found"

ui-test: ## Run frontend tests (if available)
	@echo "🧪 Running frontend tests..."
	@echo "⚠️  Frontend tests not implemented (visualization tool only)"

# ============================================================================
# Testing
# ============================================================================

test: ## Run unit tests (default)
	uv run pytest tests/unit/ -xvs --tb=short

test-all: ## Run all tests (unit + integration)
	uv run pytest tests/ -xvs --tb=short

test-unit: ## Run unit tests explicitly
	uv run pytest tests/unit/ -xvs

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -xvs --tb=short

test-fast: ## Run tests quickly (less verbose)
	uv run pytest tests/unit/ -x -q

coverage: ## Run tests with coverage report
	uv run pytest tests/ \
		--cov=src/challenge \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml
	@echo "Coverage report generated in htmlcov/index.html"

# ============================================================================
# Local LLM Testing
# ============================================================================

# Docker-based Local LLM Testing
# --------------------------------

llm-docker-up: ## Start local LLM services via Docker Compose
	@echo "🚀 Starting Ollama + LiteLLM services..."
	docker compose -f docker-compose.litellm.yml up -d
	@echo "⏳ Waiting for services to be ready (30s)..."
	@sleep 30
	@echo "✅ Services started! LiteLLM proxy available at http://localhost:4000"
	@echo "📊 View logs: make llm-docker-logs"

llm-docker-down: ## Stop local LLM Docker services
	@echo "🛑 Stopping Ollama + LiteLLM services..."
	docker compose -f docker-compose.litellm.yml down
	@echo "✅ Services stopped!"

llm-docker-logs: ## View logs from local LLM Docker services
	docker compose -f docker-compose.litellm.yml logs -f

llm-docker-test: ## Run tests using Docker-based local LLM
	@echo "🧪 Running tests with Docker local LLM..."
	@echo "📝 Ensure services are running: make llm-docker-up"
	OPENAI_BASE_URL=http://localhost:4000 \
	OPENAI_MODEL=qwen2.5:3b \
	uv run pytest tests/ -xvs --tb=short -m "not openai"
	@echo "✅ Tests completed with local LLM!"

llm-docker-clean: ## Clean local LLM Docker volumes and images
	@echo "🧹 Cleaning Docker volumes and images..."
	docker compose -f docker-compose.litellm.yml down -v
	@echo "✅ Cleanup complete!"

# Local Development LLM Testing
# ------------------------------

llm-local-setup: ## Install Ollama and LiteLLM for local development
	@echo "📦 Installing local LLM dependencies..."
	@echo ""
	@echo "1️⃣  Checking Ollama installation..."
	@if command -v ollama >/dev/null 2>&1; then \
		echo "   ✅ Ollama already installed ($$(ollama --version))"; \
	else \
		echo "   📥 Installing Ollama..."; \
		if [[ "$$(uname)" == "Darwin" ]]; then \
			if command -v brew >/dev/null 2>&1; then \
				brew install ollama; \
			else \
				echo "   ⚠️  Homebrew not found. Install from https://ollama.ai"; \
				exit 1; \
			fi; \
		else \
			curl -fsSL https://ollama.ai/install.sh | sh; \
		fi; \
	fi
	@echo ""
	@echo "2️⃣  Installing LiteLLM..."
	@pip install litellm 2>/dev/null || uv pip install litellm
	@echo "   ✅ LiteLLM installed"
	@echo ""
	@echo "3️⃣  Starting Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services start ollama 2>/dev/null || ollama serve & \
	else \
		systemctl start ollama 2>/dev/null || ollama serve & \
	fi
	@sleep 5
	@echo "   ✅ Ollama service started"
	@echo ""
	@echo "✅ Local LLM setup complete!"
	@echo "📝 Next: make llm-local-pull (to download models)"

llm-local-pull: ## Pull recommended LLM models
	@echo "📥 Pulling recommended models..."
	@echo ""
	@echo "1️⃣  Pulling Qwen2.5-3B (best quality, 2.3GB)..."
	ollama pull qwen2.5:3b
	@echo ""
	@echo "2️⃣  Pulling Qwen2.5-1.5B (faster, 1GB)..."
	ollama pull qwen2.5:1.5b
	@echo ""
	@echo "3️⃣  Pulling Phi-3-mini (alternative, 2.2GB)..."
	ollama pull phi3:mini
	@echo ""
	@echo "✅ Models downloaded!"
	@echo "📝 Next: make llm-local-start (to start proxy)"

llm-local-pull-fast: ## Pull only fast/small models (1.5B)
	@echo "📥 Pulling fast models for straightforward prompts..."
	@echo ""
	@echo "1️⃣  Pulling Qwen2.5-1.5B (fast, 1GB)..."
	ollama pull qwen2.5:1.5b
	@echo ""
	@echo "✅ Fast model downloaded!"
	@echo "📝 Use: export OPENAI_MODEL=qwen2.5:1.5b"
	@echo "📝 Then: make llm-local-start"

llm-local-start: ## Start LiteLLM proxy for local development
	@echo "🚀 Starting LiteLLM proxy..."
	@echo "📝 Config: config/litellm_config.yaml"
	@echo "🌐 Proxy will be available at http://localhost:4000"
	@echo ""
	@echo "Press Ctrl+C to stop the proxy"
	@echo "─────────────────────────────────────"
	litellm --config config/litellm_config.yaml

llm-local-stop: ## Stop LiteLLM proxy and Ollama service
	@echo "🛑 Stopping LiteLLM and Ollama..."
	@pkill -f "litellm" 2>/dev/null || true
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services stop ollama 2>/dev/null || pkill -f "ollama serve"; \
	else \
		systemctl stop ollama 2>/dev/null || pkill -f "ollama serve"; \
	fi
	@echo "✅ Services stopped!"

llm-local-test: ## Run tests using local LLM (proxy must be running)
	@echo "🧪 Running tests with local LLM..."
	@echo "📝 Ensure LiteLLM proxy is running: make llm-local-start"
	@echo ""
	OPENAI_BASE_URL=http://localhost:4000 \
	OPENAI_MODEL=qwen2.5:3b \
	uv run pytest tests/ -xvs --tb=short -m "not openai"
	@echo ""
	@echo "✅ Tests completed with local LLM!"

# LLM Utilities
# --------------

llm-check: ## Check local LLM installation status
	@echo "🔍 Checking local LLM installation..."
	@echo ""
	@echo "Ollama:"
	@if command -v ollama >/dev/null 2>&1; then \
		echo "  ✅ Installed: $$(ollama --version)"; \
		if pgrep -f "ollama serve" >/dev/null 2>&1; then \
			echo "  ✅ Service: Running"; \
		else \
			echo "  ⚠️  Service: Not running"; \
		fi; \
	else \
		echo "  ❌ Not installed"; \
	fi
	@echo ""
	@echo "LiteLLM:"
	@if command -v litellm >/dev/null 2>&1; then \
		echo "  ✅ Installed: $$(litellm --version 2>/dev/null || echo 'version unknown')"; \
		if pgrep -f "litellm" >/dev/null 2>&1; then \
			echo "  ✅ Proxy: Running at http://localhost:4000"; \
		else \
			echo "  ⚠️  Proxy: Not running"; \
		fi; \
	else \
		echo "  ❌ Not installed"; \
	fi
	@echo ""
	@echo "Models:"
	@if command -v ollama >/dev/null 2>&1; then \
		ollama list 2>/dev/null || echo "  ⚠️  Unable to list models (is Ollama running?)"; \
	else \
		echo "  ⚠️  Ollama not installed"; \
	fi

llm-status: ## Show status of local LLM services and configuration
	@echo "📊 Local LLM Status"
	@echo "═══════════════════════════════════════"
	@echo ""
	@make llm-check
	@echo ""
	@echo "Configuration:"
	@echo "  Base URL: $${OPENAI_BASE_URL:-http://localhost:4000}"
	@echo "  Model: $${OPENAI_MODEL:-qwen2.5:3b}"
	@echo "  Config: config/litellm_config.yaml"
	@echo ""
	@echo "Quick Commands:"
	@echo "  Start:  make llm-local-start  (or make llm-docker-up)"
	@echo "  Test:   make llm-local-test   (or make llm-docker-test)"
	@echo "  Stop:   make llm-local-stop   (or make llm-docker-down)"

llm-models: ## List available and downloaded LLM models
	@echo "📦 Available LLM Models"
	@echo "═══════════════════════════════════════"
	@echo ""
	@echo "Downloaded models:"
	@if command -v ollama >/dev/null 2>&1; then \
		ollama list 2>/dev/null || echo "⚠️  Unable to list models (is Ollama running?)"; \
	else \
		echo "❌ Ollama not installed"; \
	fi
	@echo ""
	@echo "Recommended models:"
	@echo "  ⚡ qwen2.5:1.5b - FASTEST (1GB, 2-3x faster, 91% accuracy)"
	@echo "  🎯 qwen2.5:3b  - Best quality (2.3GB, 97% accuracy)"
	@echo "  🔄 phi3:mini   - Alternative (2.2GB, 96% accuracy)"
	@echo ""
	@echo "For straightforward prompts: make llm-local-pull-fast"
	@echo "For all models: make llm-local-pull"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linter (ruff)
	uv run ruff check src/ tests/

lint-fix: ## Run linter and auto-fix issues
	uv run ruff check src/ tests/ --fix

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check if code is properly formatted
	uv run ruff format src/ tests/ --check

type-check: ## Run type checking with ty
	uv run ty check src/ tests/

validate: ## Run all validation steps (tests, lint, format, type-check)
	@echo "🧪 Running tests..."
	@make test-fast
	@echo ""
	@echo "🔍 Running linter..."
	@make lint
	@echo ""
	@echo "📝 Checking format..."
	@make format-check
	@echo ""
	@echo "🔎 Running type checker..."
	@make type-check
	@echo ""
	@echo "✅ All validation checks passed!"

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean Python cache and build files
	@echo "🧹 Cleaning Python files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@find . -type f -name "*.py.bak" -delete 2>/dev/null || true
	@rm -rf .tox 2>/dev/null || true
	@echo "✅ Python cleanup complete!"

# ============================================================================
# Tox Commands (for CI/CD and multi-environment testing)
# ============================================================================

tox-unit: ## Run unit tests via tox
	tox -e unit

tox-integration: ## Run integration tests via tox
	tox -e integration

tox-coverage: ## Run coverage via tox
	tox -e coverage

tox-validate: ## Run all validation via tox
	tox -e validate

tox-py312: ## Run tests on Python 3.12 specifically
	tox -e py312

# ============================================================================
# Utility Commands
# ============================================================================

deps-tree: ## Show dependency tree
	uv tree

deps-outdated: ## Show outdated dependencies
	uv tree --outdated

deps-upgrade: ## Upgrade all dependencies and update lock file
	uv lock --upgrade
	uv sync --all-extras

version: ## Show project version
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# ============================================================================
# Development Shortcuts
# ============================================================================

fix: lint-fix format ## Auto-fix all code issues (lint + format)

check: lint format-check type-check ## Run all checks without tests

quick: test-fast lint format-check ## Quick validation (fast tests + quality checks)
