#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .venv/bin/activate ]; then
  echo ".venv not found. Run ./scripts/setup_macos.sh first." >&2
  exit 1
fi

set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

source .venv/bin/activate
python -m uvicorn voice_mvp.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"

