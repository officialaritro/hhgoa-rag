#!/usr/bin/env bash
# Provisions a fresh Ubuntu EC2 instance to run the voice-enabled RAG service
# as a systemd unit (plan Task 10). Run this ON the instance, as the deploy
# user, from the repo root after cloning the code and creating .env (see
# .env.example / docs/ENVIRONMENT_VARIABLES.md). Provisioning the instance
# itself (security group, Elastic IP, key pair) is deploy/AWS_SETUP.md.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="voice-rag"
SERVICE_USER="$(whoami)"

echo "==> Installing system dependencies"
sudo apt-get update -y
sudo apt-get install -y python3 curl
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
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
ExecStart=$(command -v uv) run --project ${APP_DIR} uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "==> Enabling and starting the service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "==> Done. Check status with: sudo systemctl status ${SERVICE_NAME}"
echo "==> Health check: curl http://localhost:8000/health"
