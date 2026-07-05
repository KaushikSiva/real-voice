#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"

APP_NAME="${APP_NAME:-real-voice}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
CSM_MODEL_ID="${CSM_MODEL_ID:-sesame/csm-1b}"
PORT="${PORT:-8000}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
PULL_OLLAMA_MODEL="${PULL_OLLAMA_MODEL:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CHECK_CSM_PROCESSOR="${CHECK_CSM_PROCESSOR:-1}"
PRELOAD_CSM_MODEL="${PRELOAD_CSM_MODEL:-1}"
RUN_DOCTOR="${RUN_DOCTOR:-1}"
OVERWRITE_ENV="${OVERWRITE_ENV:-0}"
INSTALL_CADDY="${INSTALL_CADDY:-0}"
INSTALL_APP_SERVICE="${INSTALL_APP_SERVICE:-0}"

if [[ -n "${HOST+x}" ]]; then
  SERVER_HOST="$HOST"
elif [[ "$INSTALL_CADDY" == "1" && -n "${PUBLIC_DOMAIN:-}" ]]; then
  SERVER_HOST="127.0.0.1"
else
  SERVER_HOST="0.0.0.0"
fi

usage() {
  cat <<'MSG'
Usage: ./scripts/setup_vm_cuda.sh

Sets up a Linux CUDA VM for the browser voice console:
  - apt packages: ffmpeg, git, curl, Python venv tooling
  - Python .venv and requirements.txt
  - Ollama install/start and model pull
  - .env with CUDA defaults
  - Hugging Face login/access check for sesame/csm-1b
  - optional CSM model preload

Useful environment variables:
  HF_TOKEN=...                    Non-interactive Hugging Face login token.
  OLLAMA_MODEL=llama3.2:3b         Ollama model to pull.
  PRELOAD_CSM_MODEL=0              Skip downloading/loading CSM during setup.
  OVERWRITE_ENV=1                  Rewrite an existing .env.
  INSTALL_CADDY=1 PUBLIC_DOMAIN=voice.example.com
                                  Configure HTTPS reverse proxy with Caddy.
  INSTALL_APP_SERVICE=1            Install and start a systemd service.

After setup:
  ./scripts/run_server.sh
MSG
}

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf '\n[warn] %s\n' "$*" >&2
}

die() {
  printf '\n[error] %s\n' "$*" >&2
  exit 1
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root."
    sudo "$@"
  fi
}

sudo_env() {
  if [[ "$(id -u)" -eq 0 ]]; then
    env "$@"
  else
    command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root."
    sudo env "$@"
  fi
}

install_system_packages() {
  command -v apt-get >/dev/null 2>&1 || die "This setup script expects a Debian/Ubuntu VM with apt-get."

  log "Installing system packages"
  sudo_env DEBIAN_FRONTEND=noninteractive apt-get update
  sudo_env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    python3-dev \
    python3-pip \
    python3-venv

  if [[ "$INSTALL_CADDY" == "1" ]]; then
    sudo_env DEBIAN_FRONTEND=noninteractive apt-get install -y caddy
  fi
}

select_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN not found: $PYTHON_BIN"
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
    return
  fi

  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      return
    fi
  done

  die "No usable python3 executable found."
}

check_python_version() {
  "$PYTHON_BIN" - <<'PY'
import sys

version = ".".join(map(str, sys.version_info[:3]))
print(f"Using Python {version}")
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
if sys.version_info[:2] != (3, 12):
    print("[warn] Python 3.12 is recommended; continuing with this VM Python.")
PY
}

install_python_deps() {
  log "Creating Python virtual environment"
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements.txt
}

write_env_file() {
  if [[ -f .env && "$OVERWRITE_ENV" != "1" ]]; then
    warn ".env already exists; leaving it unchanged. Set OVERWRITE_ENV=1 to rewrite it."
    return
  fi

  log "Writing .env"
  cat > .env <<EOF
OLLAMA_BASE_URL=$OLLAMA_BASE_URL
OLLAMA_MODEL=$OLLAMA_MODEL
CSM_MODEL_ID=$CSM_MODEL_ID

HOST=$SERVER_HOST
PORT=$PORT
TTS_DEVICE=cuda
TTS_DTYPE=float16
WHISPER_DEVICE=cuda
WHISPER_MODEL=base

AUTO_WARMUP=1
MAX_TTS_CHUNKS=1
CSM_MAX_NEW_TOKENS=80
MAX_SPOKEN_WORDS=10
REFERENCE_SECONDS=3

CANNED_FILLERS_ENABLED=1
CANNED_FILLERS_AUTO_BUILD=1
CANNED_FILLERS="sure=Sure.|okay=Okay.|hmm=Hmm.|sorry=Sorry.|cough=Cough.|sneeze=Achoo."
CANNED_AUTO_FILLERS=sure,okay,hmm
CANNED_MAX_NEW_TOKENS=48

NO_TORCH_COMPILE=1
PYTORCH_ENABLE_MPS_FALLBACK=1
EOF
}

wait_for_ollama() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_ollama() {
  log "Starting Ollama"
  if command -v systemctl >/dev/null 2>&1; then
    sudo_cmd systemctl enable --now ollama || warn "Could not start Ollama with systemd."
  fi

  if wait_for_ollama; then
    return
  fi

  warn "Ollama system service did not answer; starting ollama serve in the background."
  mkdir -p logs
  nohup ollama serve > logs/ollama.log 2>&1 &
  wait_for_ollama || die "Ollama did not start. Check logs/ollama.log."
}

install_ollama() {
  if [[ "$INSTALL_OLLAMA" != "1" ]]; then
    warn "Skipping Ollama install/start because INSTALL_OLLAMA=$INSTALL_OLLAMA."
    return
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh -o /tmp/ollama-install.sh
    sh /tmp/ollama-install.sh
  fi

  start_ollama

  if [[ "$PULL_OLLAMA_MODEL" == "1" ]]; then
    log "Pulling Ollama model: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL"
  fi
}

login_hugging_face() {
  # shellcheck disable=SC1091
  source .venv/bin/activate

  local token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -n "$token" ]]; then
    log "Logging in to Hugging Face with HF_TOKEN"
    huggingface-cli login --token "$token"
    return
  fi

  if huggingface-cli whoami >/dev/null 2>&1; then
    log "Hugging Face login already present"
    return
  fi

  if [[ -t 0 ]]; then
    log "Log in to Hugging Face. Make sure your account accepted access to $CSM_MODEL_ID."
    huggingface-cli login
  else
    warn "No Hugging Face login found. Set HF_TOKEN=... or run: source .venv/bin/activate && huggingface-cli login"
  fi
}

check_cuda() {
  # shellcheck disable=SC1091
  source .venv/bin/activate

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Detected NVIDIA GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    warn "nvidia-smi not found on PATH."
  fi

  if python - <<'PY'; then
import torch

print(f"torch={torch.__version__}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.version.cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(2)
PY
    return
  fi

  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    die "Torch cannot see CUDA. Use a GPU image/VM with NVIDIA drivers, or set REQUIRE_CUDA=0 to continue anyway."
  fi
  warn "Torch cannot see CUDA; continuing because REQUIRE_CUDA=$REQUIRE_CUDA."
}

check_csm_processor() {
  if [[ "$CHECK_CSM_PROCESSOR" != "1" ]]; then
    return
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Checking CSM processor access: $CSM_MODEL_ID"
  CSM_MODEL_ID="$CSM_MODEL_ID" python - <<'PY'
import os
from transformers import AutoProcessor

model_id = os.environ["CSM_MODEL_ID"]
processor = AutoProcessor.from_pretrained(model_id)
print(f"CSM processor loaded: {type(processor).__name__}")
PY
}

preload_csm_model() {
  if [[ "$PRELOAD_CSM_MODEL" != "1" ]]; then
    warn "Skipping CSM model preload because PRELOAD_CSM_MODEL=$PRELOAD_CSM_MODEL."
    return
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"

  log "Preloading CSM model on CUDA"
  CSM_MODEL_ID="$CSM_MODEL_ID" python - <<'PY'
import os
from voice_mvp.tts_csm import CsmTts

tts = CsmTts(
    model_id=os.environ["CSM_MODEL_ID"],
    device="cuda",
    dtype="float16",
    allow_cpu=False,
    max_new_tokens=8,
)
print(f"CSM model loaded on {tts.device}")
PY
}

run_doctor() {
  if [[ "$RUN_DOCTOR" != "1" ]]; then
    return
  fi

  log "Running doctor"
  ./scripts/doctor.py --check-csm-download
}

configure_caddy() {
  if [[ "$INSTALL_CADDY" != "1" ]]; then
    return
  fi
  [[ -n "${PUBLIC_DOMAIN:-}" ]] || die "Set PUBLIC_DOMAIN=voice.example.com when INSTALL_CADDY=1."
  command -v caddy >/dev/null 2>&1 || die "caddy was not installed."

  log "Configuring Caddy for https://$PUBLIC_DOMAIN"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
$PUBLIC_DOMAIN {
    reverse_proxy 127.0.0.1:$PORT
}
EOF
  sudo_cmd install -m 0644 "$tmp" /etc/caddy/Caddyfile
  rm -f "$tmp"
  sudo_cmd caddy fmt --overwrite /etc/caddy/Caddyfile || true

  if command -v systemctl >/dev/null 2>&1; then
    sudo_cmd systemctl enable --now caddy
    sudo_cmd systemctl reload caddy
  else
    warn "systemctl not available; start Caddy manually."
  fi
}

install_app_service() {
  if [[ "$INSTALL_APP_SERVICE" != "1" ]]; then
    return
  fi
  command -v systemctl >/dev/null 2>&1 || die "systemd is required for INSTALL_APP_SERVICE=1."

  local app_user service_file tmp
  app_user="${APP_USER:-$(id -un)}"
  service_file="/etc/systemd/system/$APP_NAME.service"
  tmp="$(mktemp)"

  log "Installing systemd service: $APP_NAME"
  cat > "$tmp" <<EOF
[Unit]
Description=Real Voice FastAPI server
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=$app_user
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/.env
ExecStart=$ROOT/.venv/bin/python -m uvicorn voice_mvp.server:app --host $SERVER_HOST --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo_cmd install -m 0644 "$tmp" "$service_file"
  rm -f "$tmp"
  sudo_cmd systemctl daemon-reload
  sudo_cmd systemctl enable --now "$APP_NAME"
}

finish_message() {
  cat <<MSG

Setup complete.

Start the app:
  cd $ROOT
  ./scripts/run_server.sh

Open:
MSG
  if [[ "$INSTALL_CADDY" == "1" && -n "${PUBLIC_DOMAIN:-}" ]]; then
    cat <<MSG
  https://$PUBLIC_DOMAIN

Make sure DNS points to this VM and cloud firewall ports 80/443 are open.
MSG
  else
    cat <<MSG
  http://<VM_PUBLIC_IP>:$PORT

For microphone without HTTPS, use SSH forwarding from your laptop:
  ssh -i ~/.ssh/real_voice_h200 -L $PORT:127.0.0.1:$PORT <user>@<VM_PUBLIC_IP>
  open http://127.0.0.1:$PORT
MSG
  fi

  if [[ "$INSTALL_APP_SERVICE" == "1" ]]; then
    cat <<MSG

Service commands:
  sudo systemctl status $APP_NAME
  sudo journalctl -u $APP_NAME -f
MSG
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

install_system_packages
select_python
check_python_version
install_python_deps
write_env_file
install_ollama
login_hugging_face
check_cuda
check_csm_processor
preload_csm_model
run_doctor
configure_caddy
install_app_service
finish_message
