#!/usr/bin/env bash
# Deploy a ref to the running instance, with a health gate and automatic
# rollback. Run ON the instance:
#
#   sudo -u ubuntu bash deploy/deploy.sh [git-ref]
#
# Defaults to the current branch's upstream. On failure the previous commit is
# restored and the service restarted, so a bad deploy cannot leave the demo
# down -- which matters more than usual when the live link is the submission.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="voice-rag"
HEALTH_URL="http://127.0.0.1:8000/health"
HEALTH_TIMEOUT=180          # model load alone is ~17s, plus FAISS indices
HEALTH_INTERVAL=5

cd "$APP_DIR"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31m!! %s\033[0m\n' "$*" >&2; }

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  TARGET="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo origin/main)"
fi

PREVIOUS="$(git rev-parse HEAD)"
log "Deploying $TARGET (rollback point: ${PREVIOUS:0:8})"

git fetch --quiet origin
git reset --quiet --hard "$TARGET"
NEW="$(git rev-parse HEAD)"
echo "    ${PREVIOUS:0:8} -> ${NEW:0:8}"

if [ "$PREVIOUS" = "$NEW" ]; then
  echo "    already at target; restarting anyway to pick up any config change"
fi

# Only resync when the dependency set actually moved -- uv sync is the slowest
# step here and most deploys are code-only.
if ! git diff --quiet "$PREVIOUS" "$NEW" -- pyproject.toml uv.lock; then
  log "Dependencies changed -- syncing"
  export PATH="$HOME/.local/bin:$PATH"
  uv sync --locked
else
  echo "    dependencies unchanged; skipping uv sync"
fi

log "Restarting $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

log "Waiting for health (up to ${HEALTH_TIMEOUT}s)"
elapsed=0
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_URL" 2>/dev/null || echo 000)"
  if [ "$code" = "200" ]; then
    log "Healthy after ${elapsed}s -- deploy complete (${NEW:0:8})"
    exit 0
  fi
  printf '    +%ds  health=%s\n' "$elapsed" "$code"
  sleep "$HEALTH_INTERVAL"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

fail "Did not become healthy within ${HEALTH_TIMEOUT}s -- rolling back to ${PREVIOUS:0:8}"
journalctl -u "$SERVICE_NAME" --no-pager -n 30 | grep -v 'GET /health' | tail -15 >&2

git reset --quiet --hard "$PREVIOUS"
sudo systemctl restart "$SERVICE_NAME"

elapsed=0
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_URL" 2>/dev/null || echo 000)"
  if [ "$code" = "200" ]; then
    fail "Rolled back to ${PREVIOUS:0:8}; service healthy again. Deploy FAILED."
    exit 1
  fi
  sleep "$HEALTH_INTERVAL"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

fail "ROLLBACK ALSO UNHEALTHY -- manual intervention required."
fail "  journalctl -u ${SERVICE_NAME} -n 50"
exit 2
