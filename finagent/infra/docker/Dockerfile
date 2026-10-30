FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock /app/
COPY finagent /app/finagent
RUN uv sync --frozen --no-dev --extra production
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8080
CMD ["uv", "run", "--frozen", "uvicorn", "finagent.apps.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
