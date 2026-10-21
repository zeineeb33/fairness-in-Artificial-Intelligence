# Multi-stage image for Render's free tier (512 MB RAM, 0.1 CPU).
#   builder  - runtime virtualenv: core + API dependencies only (no scikit-learn)
#   trainer  - adds the training stack and runs the case study during the build
#   runtime  - slim image with the virtualenv and the generated case-study artifacts
ARG PYTHON_IMAGE=python:3.12-slim-trixie

FROM ghcr.io/astral-sh/uv:0.9.22 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, so this layer is reused when only the source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra api --no-editable

FROM builder AS trainer
RUN uv sync --frozen --no-dev --extra api --extra study --no-editable
COPY configs ./configs
COPY data/JEMLI_ZEINEB.csv ./data/
RUN /opt/venv/bin/fairaudit train --config configs/chestxray.yaml --output-dir /artifacts/chestxray

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
COPY --from=trainer --chown=app:app /artifacts /app/artifacts
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FAIRAUDIT_ARTIFACTS_DIR=/app/artifacts \
    PORT=8000
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\", timeout=4)"]
# `fairaudit serve` binds 0.0.0.0:$PORT (Render injects PORT).
CMD ["fairaudit", "serve"]
