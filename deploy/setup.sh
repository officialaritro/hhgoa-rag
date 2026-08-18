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

echo "==> Syncing Python dependencies via uv"
cd "$APP_DIR"
uv sync

if [ ! -f "$APP_DIR/.env" ]; then
  echo "ERROR: $APP_DIR/.env not found. Copy .env.example to .env and fill in" \
       "ELEVENLABS_API_KEY / ANTHROPIC_API_KEY before running this script." >&2
  exit 1
fi

if [ ! -f "$APP_DIR/data/index_fixed_size.faiss" ] || [ ! -f "$APP_DIR/data/index_semantic.faiss" ]; then
  echo "==> Building corpus and indices (first deploy only; skip if data/ already populated)"
  uv run python -m scripts.ingest_dataset
  uv run python -m scripts.chunk_fixed_size
  uv run python -m scripts.chunk_semantic
  uv run python -m scripts.build_index
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
# Loopback only -- Caddy terminates TLS on 443 and proxies here. Binding
# 0.0.0.0 would expose the app unencrypted on 8000 and break the microphone.
ExecStart=$(command -v uv) run --project ${APP_DIR} uvicorn app.main:app --host 127.0.0.1 --port 8000
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
