#!/usr/bin/env bash
# Provisions a fresh Ubuntu EC2 instance to run the voice-enabled RAG service
# as a systemd unit (plan Task 10). Run this ON the instance, as the deploy
# user, from the repo root after cloning the code and creating .env (see
# .env.example / docs/ENVIRONMENT_VARIABLES.md). Provisioning the instance
# itself (security group, Elastic IP, key pair) is deploy/AWS_SETUP.md.
#
# This also installs Caddy for TLS. TLS is mandatory, not cosmetic:
# getUserMedia/MediaRecorder are secure-context-only, so over plain HTTP
# navigator.mediaDevices is undefined and the demo silently has no microphone.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="voice-rag"
SERVICE_USER="$(whoami)"
PUBLIC_HOST="ragingoa.duckdns.org"
# Model cache lives inside the app dir, not $HOME. systemd hardening
# (ProtectHome) makes /home unreadable to the service, and a service that
# cannot read its cached model fails startup with no obvious cause.
HF_CACHE_DIR="${APP_DIR}/.cache/huggingface"
PREFLIGHT_DIR="/opt/hhgoa-rag-preflight"

echo "==> Installing system dependencies"
sudo apt-get update -y
sudo apt-get install -y python3 curl
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Installing Caddy (TLS termination)"
if ! command -v caddy >/dev/null 2>&1; then
  sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list > /dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi

echo "==> Model cache directory"
mkdir -p "$HF_CACHE_DIR"
# Adopt any cache already downloaded under $HOME so a re-run does not refetch ~4GB.
if [ -d "$HOME/.cache/huggingface" ] && [ ! -e "$HF_CACHE_DIR/hub" ]; then
  mv "$HOME/.cache/huggingface/"* "$HF_CACHE_DIR/" 2>/dev/null || true
fi

echo "==> Syncing Python dependencies via uv"
cd "$APP_DIR"
uv sync

if [ ! -f "$APP_DIR/.env" ]; then
  echo "ERROR: $APP_DIR/.env not found. Copy .env.example to .env and fill in" \
       "ELEVENLABS_API_KEY / ANTHROPIC_API_KEY before running this script." >&2
  exit 1
fi

if [ ! -f "$APP_DIR/data/index_whole_passage.faiss" ] || [ ! -f "$APP_DIR/data/passages.pkl" ]; then
  echo "==> Building corpus and indices (first deploy only; skip if data/ already populated)"
  uv run python -m scripts.ingest_dataset
  # One command builds every registered strategy: it derives the slate from the
  # registry, so a strategy added later needs no change here. Chunking is no
  # longer a separate step -- chunkers stream straight into the embedder rather
  # than materialising intermediate files.
  uv run python -m scripts.build_all
  # BM25 for the hybrid strategy, and the per-index off-topic thresholds. The
  # thresholds are not optional: app/guardrails.py raises MissingCalibration
  # rather than borrow another index's number, so an uncalibrated index cannot
  # be served at all.
  uv run python -m scripts.build_lexical
  uv run python -m scripts.calibrate_thresholds
fi

echo "==> Writing systemd unit"
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=Voice-Enabled RAG Pipeline
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
Environment=HF_HOME=${HF_CACHE_DIR}
# Loopback only -- Caddy terminates TLS on 443 and proxies here. Binding
# 0.0.0.0 would expose the app unencrypted on 8000 and break the microphone.
#
# Exec the venv's uvicorn directly rather than via `uv run`. uv sync above
# already installed the dependencies; having systemd shell out to uv at start
# time makes the service depend on uv being on root's PATH and lets a resolver
# step run on every restart -- including on a box whose venv uv would rather
# rebuild. The binary is the same either way.
ExecStart=${APP_DIR}/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
# sentence-transformers import + model load measures ~17.5s; allow for that
# plus FAISS index loading before systemd calls the start a failure.
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

echo "==> Enabling and starting the service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "==> Configuring Caddy + preflight page"
sudo install -d "${PREFLIGHT_DIR}"
sudo install -m 0644 "${APP_DIR}/deploy/preflight.html" "${PREFLIGHT_DIR}/index.html"
sudo install -m 0644 "${APP_DIR}/deploy/Caddyfile" /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile > /dev/null
sudo systemctl enable caddy
sudo systemctl restart caddy

echo "==> Done."
echo "    Service   : sudo systemctl status ${SERVICE_NAME}"
echo "    Logs      : journalctl -u ${SERVICE_NAME} -f"
echo "    Health    : curl https://${PUBLIC_HOST}/health"
echo "    Preflight : https://${PUBLIC_HOST}/preflight   <- open in a browser,"
echo "                confirm the microphone prompt appears before demo day"
