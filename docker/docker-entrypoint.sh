#!/bin/sh
set -e

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Running database migrations..."
  python -m alembic upgrade head
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers "${WEB_CONCURRENCY:-4}"
