#!/usr/bin/env bash
# Start local AMN Turkish chat on Debian/Linux; model inference never calls a cloud API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -f results/open_turkish_retrain_v2/best.pt ||
      ! -f results/open_turkish_retrain_v2/tokenizer.model ]]; then
  echo "Model bulunamadı. AMN-4W-Edit deposunun güncel main sürümünü indir." >&2
  exit 1
fi
VENV="$ROOT/.venv-amn-chat"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
if [[ ! -f "$VENV/.deps-ready-v1" ]]; then
  "$VENV/bin/python" -m pip install -r requirements.txt
  touch "$VENV/.deps-ready-v1"
fi
exec "$VENV/bin/python" experiments/amn_chat_server.py "$@"
