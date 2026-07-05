#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  python3.12 -m voice_mvp.assistant --help
  exit 0
fi

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
python -m voice_mvp.assistant "$@"
