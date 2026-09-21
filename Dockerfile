FROM node:24.15.0-alpine AS frontend-builder

WORKDIR /web-ui

COPY web-ui/package.json web-ui/package-lock.json ./
RUN npm ci

COPY web-ui ./
RUN npm run build


FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt


FROM builder AS test

WORKDIR /app

COPY requirements-dev.txt ./
RUN /opt/venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt

COPY alembic.ini ./
COPY evaluate.py ./
COPY alembic ./alembic
COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY example_submission ./example_submission
COPY tests ./tests

CMD ["sh", "-c", "/opt/venv/bin/alembic upgrade head && /opt/venv/bin/python -m pytest -q"]


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /data \
    && chown appuser:appuser /app/artifacts /data

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser evaluate.py ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
COPY --from=frontend-builder --chown=appuser:appuser /web-ui/dist ./frontend/dist
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/vehicle-reid-entrypoint

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/vehicle-reid-entrypoint"]
CMD ["python", "-m", "backend", "--host", "0.0.0.0", "--port", "8000"]
