#!/usr/bin/env bash
# Launch the FastAPI server via the repo venv.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "INFO: .venv not found at .venv; using current python on PATH."
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
exec python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
