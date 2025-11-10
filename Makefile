# Makefile for crane-challenge
# Modern Python project with Clean Architecture

.PHONY: help install dev-install setup first-run test lint lint-fix format format-fix type-check coverage validate clean run stop-all \
	test-all test-unit test-integration test-fast api-dev api-prod api-test api-docs api-health \
	backend-dev backend-stop backend-logs \
	ui-install ui-dev ui-build ui-clean ui-lint ui-test \
	llm-local-setup llm-local-pull llm-local-pull-fast llm-local-start llm-local-stop llm-local-test \
	llm-check llm-status llm-models llm-config-check

# ============================================================================
# Help & Documentation
# ============================================================================

help: ## Show this help message
	@echo "crane-challenge - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@grep -E '^(setup|first-run|install|dev-install):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^(run|test|lint|format|type-check|validate):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "API Development:"
	@grep -E '^(api-.*|backend-.*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
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
	@grep -E '^(clean.*|stop-all):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

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
	@echo ""
	@echo "✅ Dependencies installed!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "  1. Copy .env.example to .env and configure LLM credentials"
	@echo "  2. Run: make llm-config-check (to verify configuration)"
	@echo "  3. Run: make test (to verify everything works)"

setup: first-run ## Alias for first-run (complete automated setup)

first-run: ## 🚀 Complete first-time setup (deps + native LLM + backend + optional frontend)
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║          🚀 Crane Challenge - First-Time Setup                 ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "This will:"
	@echo "  1. Check prerequisites (Docker, uv)"
	@echo "  2. Install all dependencies"
	@echo "  3. Setup .env configuration"
	@echo "  4. Start native LLM services (Ollama + LiteLLM)"
	@echo "  5. Start backend API in background"
	@echo "  6. Optionally start frontend UI"
	@echo ""
	@read -p "Continue? [y/N] " response; \
	if [ "$$response" != "y" ] && [ "$$response" != "Y" ]; then \
		echo "Setup cancelled."; \
		exit 1; \
	fi
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Step 1/5: Checking Prerequisites"
	@echo "═══════════════════════════════════════════════════════════════"
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "❌ Docker not found. Please install Docker Desktop:"; \
		echo "   https://www.docker.com/products/docker-desktop"; \
		exit 1; \
	fi
	@echo "✅ Docker installed: $$(docker --version)"
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "❌ uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "✅ uv installed: $$(uv --version)"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Step 2/5: Installing Dependencies"
	@echo "═══════════════════════════════════════════════════════════════"
	@make dev-install
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Step 3/5: Setting up .env Configuration"
	@echo "═══════════════════════════════════════════════════════════════"
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "✅ .env file created"; \
		echo ""; \
		echo "⚠️  Using default configuration (local LLM via Docker)"; \
		echo "   To use cloud providers (OpenAI/Anthropic), edit .env"; \
	else \
		echo "✅ .env file already exists (keeping your configuration)"; \
	fi
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Step 4/5: Setting Up Native Local LLM Services"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "🚀 Installing and configuring Ollama + LiteLLM..."
	@make llm-local-setup
	@echo ""
	@echo "📥 Pulling qwen2.5:3b model (~1.9GB)..."
	@ollama pull qwen2.5:3b
	@echo "✅ Model downloaded!"
	@echo ""
	@echo "🚀 Starting LiteLLM proxy..."
	@make llm-local-start
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Step 5/5: Verifying Configuration"
	@echo "═══════════════════════════════════════════════════════════════"
	@make llm-config-check
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║              ✅ Setup Complete - Ready to Use!                 ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Starting Backend API"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@make backend-dev
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Optional: Start Frontend UI"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@read -p "🎨 Start frontend UI? [y/N] " start_frontend; \
	echo ""; \
	if [ "$$start_frontend" = "y" ] || [ "$$start_frontend" = "Y" ]; then \
		echo "🎨 Starting Frontend UI in background..."; \
		echo "   → Running on: http://localhost:3000"; \
		echo "   → Will open in browser automatically"; \
		echo ""; \
		nohup make ui-dev > /tmp/crane-ui.log 2>&1 & \
		sleep 3; \
		echo "✅ Frontend started in background (PID: $$!)"; \
		echo ""; \
		echo "📝 Frontend Management:"; \
		echo "  • View logs:     tail -f /tmp/crane-ui.log"; \
		echo "  • Stop frontend: make stop-all"; \
		echo "  • Or kill process on port 3000"; \
		echo ""; \
	fi; \
	echo ""; \
	echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║              🎉 All Services Running!                          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🔗 Service URLs:"; \
	echo "  • Backend API:         http://localhost:8000"; \
	echo "  • API Docs:            http://localhost:8000/api/docs"; \
	echo "  • Frontend UI:         http://localhost:3000"; \
	echo ""; \
	echo "🎯 Quick Commands:"; \
	echo "  • Test everything:     make test-all"; \
	echo "  • Backend logs:        make backend-logs"; \
	echo "  • Frontend logs:       tail -f /tmp/crane-ui.log"; \
	echo "  • Stop all services:   make stop-all"; \
	echo "  • Check LLM status:    make llm-status"; \
	echo ""; \
	echo "🤖 Native LLM Services:"; \
	echo "  • View logs:           tail -f /tmp/litellm.log"; \
	echo "  • Stop services:       make llm-local-stop"; \
	echo "  • Restart services:    make llm-local-stop && make llm-local-start"; \
	echo "  • Pull more models:    ollama pull qwen2.5:3b"; \
	echo ""; \
	echo "📚 Documentation:"; \
	echo "  • README.md"; \
	echo "  • docs/architecture.md"; \
	echo "  • docs/multi_provider_llm.md"; \
	echo ""

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
	uv run pytest tests/unit/presentation/api/ -xvs

api-health: ## Check API health endpoint
	@echo "Checking API health..."
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool || echo "API is not running. Start with 'make api-dev'"

api-docs: ## Open API documentation in browser
	@echo "Opening API docs at http://localhost:8000/api/docs"
	@python -m webbrowser http://localhost:8000/api/docs || open http://localhost:8000/api/docs || xdg-open http://localhost:8000/api/docs

backend-dev: ## Run backend API in background (similar to ui-dev)
	@echo "🚀 Starting backend API server in background..."
	@if lsof -ti:8000 >/dev/null 2>&1; then \
		echo "⚠️  Backend already running on port 8000"; \
		echo "   Stop it with: make backend-stop"; \
	else \
		nohup uv run python -m challenge > /tmp/crane-backend.log 2>&1 & \
		echo $$! > .backend.pid; \
		sleep 2; \
		if lsof -ti:8000 >/dev/null 2>&1; then \
			echo "✅ Backend started successfully (PID: $$(cat .backend.pid))"; \
			echo "   API: http://localhost:8000"; \
			echo "   Docs: http://localhost:8000/api/docs"; \
			echo "   Logs: tail -f /tmp/crane-backend.log"; \
			echo "   Stop: make backend-stop"; \
		else \
			echo "❌ Backend failed to start"; \
			echo "   Check logs: cat /tmp/crane-backend.log"; \
			rm -f .backend.pid; \
		fi; \
	fi

backend-stop: ## Stop backend API server
	@echo "🛑 Stopping backend API server..."
	@if [ -f .backend.pid ]; then \
		PID=$$(cat .backend.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			kill $$PID 2>/dev/null && echo "   ✅ Backend stopped (PID: $$PID)" || echo "   ⚠️  Could not stop backend"; \
		else \
			echo "   ℹ️  Backend process not found (cleaning up stale PID file)"; \
		fi; \
		rm -f .backend.pid; \
	elif lsof -ti:8000 >/dev/null 2>&1; then \
		kill -9 $$(lsof -ti:8000) 2>/dev/null && echo "   ✅ Backend stopped (port 8000)" || echo "   ⚠️  Could not stop backend"; \
	else \
		echo "   ℹ️  No backend running on port 8000"; \
	fi

backend-logs: ## View backend API logs
	@echo "📋 Backend API logs (Ctrl+C to exit):"
	@tail -f /tmp/crane-backend.log

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

stop-all: ## 🛑 Stop all services (native LLM + backend + frontend)
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║              🛑 Stopping All Services...                       ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1️⃣  Stopping native LLM services (LiteLLM + Ollama)..."
	@make llm-local-stop
	@echo ""
	@echo "2️⃣  Stopping backend API..."
	@make backend-stop
	@echo ""
	@echo "3️⃣  Stopping frontend UI (port 3000)..."
	@if lsof -ti:3000 >/dev/null 2>&1; then \
		kill -9 $$(lsof -ti:3000) 2>/dev/null && echo "   ✅ Frontend stopped (port 3000)" || echo "   ⚠️  Could not stop frontend"; \
	else \
		echo "   ℹ️  No frontend running on port 3000"; \
	fi
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║              ✅ All Services Stopped!                          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📝 Note: Redis/Postgres Docker containers kept running."
	@echo "   Stop them with: docker compose down"
	@echo ""

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
	@if command -v litellm >/dev/null 2>&1; then \
		echo "   ✅ LiteLLM already installed ($$(litellm --version 2>/dev/null || echo 'version unknown'))"; \
	else \
		uv tool install 'litellm[proxy]'; \
		echo "   ✅ LiteLLM installed with proxy extras"; \
	fi
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
	@./scripts/start-litellm.sh

llm-local-stop: ## Stop LiteLLM proxy and Ollama service
	@./scripts/stop-litellm.sh
	@echo "🛑 Stopping Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services stop ollama 2>/dev/null || pkill -f "ollama serve" 2>/dev/null || true; \
	else \
		systemctl stop ollama 2>/dev/null || pkill -f "ollama serve" 2>/dev/null || true; \
	fi
	@echo "✅ All LLM services stopped!"

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

llm-config-check: ## Verify LLM configuration (API keys, base URL, etc.)
	@echo "🔍 Verifying LLM configuration..."
	@uv run python scripts/verify_llm_config.py

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

llm-status: ## Show status of native LLM services and configuration
	@echo "📊 Native LLM Status"
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
	@echo "  Start:  make llm-local-start"
	@echo "  Test:   make llm-local-test"
	@echo "  Stop:   make llm-local-stop"
	@echo "  Logs:   tail -f /tmp/litellm.log"

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
