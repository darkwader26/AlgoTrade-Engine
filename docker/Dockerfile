# =============================================================================
#  AlgoTrade Engine — Production Docker Image
# =============================================================================
# Multi-stage build: builder (dependencies) → runtime (slim production image)
# =============================================================================

# ---------------------------------------------------------------------------
#  Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL stage=builder

# Install uv — fast Python package installer and resolver
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata first for layer caching
COPY pyproject.toml ./

# Copy source packages
COPY core/ core/
COPY features/ features/
COPY strategies/ strategies/
COPY tests/ tests/

# Install production dependencies (no dev extras)
RUN uv sync --no-dev --frozen

# ---------------------------------------------------------------------------
#  Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AlgoTrade Engine" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.description="Data pipeline, feature engineering, and ML signal generation for algorithmic trading"

# Create non-root user
RUN groupadd --system --gid 1000 trading && \
    useradd --system --uid 1000 --gid trading trading

# Copy installed packages and source from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Set PYTHONPATH so packages can be resolved without explicit installation
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Healthcheck — verify all core packages can be imported
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import core; import features; import strategies" || exit 1

# Drop privileges
USER trading

# Default entry point
ENTRYPOINT ["python", "-m", "strategies.engine"]
