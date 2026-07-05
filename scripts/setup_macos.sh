#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'MSG'
Usage: ./scripts/setup_macos.sh

Creates .venv with python3.12 and installs requirements.txt.

After setup:
  source .venv/bin/activate
  huggingface-cli login
  ollama pull llama3.2:3b
  ./scripts/doctor.py
MSG
  exit 0
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 is required. Install it with pyenv or Homebrew, then rerun this script." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

cat <<'MSG'

Setup complete.

Next:
  source .venv/bin/activate
  huggingface-cli login
  ollama pull llama3.2:3b
  ./scripts/doctor.py

MSG
