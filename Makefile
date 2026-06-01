# =============================================================================
#  AlgoTrade Engine — Makefile
# =============================================================================
#  Targets:
#    setup      — Install project dependencies
#    test       — Run tests with verbose output
#    lint       — Run ruff static analysis
#    docker     — Build the Docker image
#    run-paper  — Run in paper trading mode
#    clean      — Remove all caches and build artifacts
#    coverage   — Run tests with coverage report
# =============================================================================

.PHONY: setup test lint docker run-paper clean coverage

# ── Setup ─────────────────────────────────────────────────────────────────
setup:
	uv sync

# ── Testing ───────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

coverage:
	uv run pytest tests/ -v --cov --cov-report=term --cov-report=html

# ── Linting ───────────────────────────────────────────────────────────────
lint:
	uv run ruff check .

# ── Docker ────────────────────────────────────────────────────────────────
docker:
	docker compose -f docker/docker-compose.yml build

# ── Running ───────────────────────────────────────────────────────────────
run-paper:
	PAPER_TRADING=true uv run python -m strategies.engine

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage coverage.xml htmlcov/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	@echo "Cleaned all caches and build artifacts."
