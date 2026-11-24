FROM python:3.12-slim

# uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. dependencies first: this layer is cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2. then the code (changes often, so it comes last)
COPY app ./app

# 3. run as a non-root user; data lives in /app/data (mounted as a volume)
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    DATABASE_URL=sqlite:////app/data/modelflow.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]