#!/usr/bin/env bash
# Launch the live conv-autoencoder visualisation on a local web server.
#
#   ./run_cnn_viz.sh          # serve on http://localhost:8000 and open a browser
#   ./run_cnn_viz.sh 9000     # pick a different port
#
# No dependencies beyond Python 3 (standard library http.server only).

set -euo pipefail

PORT="${1:-8000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cnn_viz"
URL="http://localhost:${PORT}/"

if [ ! -f "${DIR}/index.html" ]; then
  echo "error: ${DIR}/index.html not found" >&2
  exit 1
fi

# locate a Python interpreter
PY=""
for cand in python3 python py; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "error: no Python interpreter found (need python3)" >&2
  exit 1
fi

echo "serving ${DIR}"
echo "  -> ${URL}"
echo "press Ctrl+C to stop"

cd "$DIR"
"$PY" -m http.server "$PORT" --bind 127.0.0.1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT INT TERM

# give the server a moment, then open the default browser (best effort)
sleep 1
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open    >/dev/null 2>&1; then open "$URL"     >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$URL" >/dev/null 2>&1 || true
elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" >/dev/null 2>&1 || true
else "$PY" -c "import webbrowser; webbrowser.open('$URL')" >/dev/null 2>&1 || true
fi

wait "$SRV"
