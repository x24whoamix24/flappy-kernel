#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Starts the arcade on a GPU node: the Mirage megakernel process (Qwen3-8B,
# the A2C trainer and the arcade's games in one launch that serves every
# question) and the web server players connect to.
#
# Settings (environment):
#   MODEL=Qwen/Qwen3-8B   LLM to serve (a Qwen3 checkpoint)
#   MAX_SEQ=1024          longest prompt + answer, in tokens
#   LLM_WORKERS=96        SMs running the LLM's decode tasks
#   COTENANT_BLOCKS=32    SMs for the game (1) and the trainer (the rest)
#   PORT=8080             web server port
#   FK=<this repository>  SERVE_DIR=/var/lib/flappy-kernel  MIRAGE=/root/mirage
# Extra arguments go to server/app.py (e.g. --allow-origin, --trust-proxy).
set -euo pipefail
FK=${FK:-$(cd "$(dirname "$0")/.." && pwd)}
SERVE_DIR=${SERVE_DIR:-/var/lib/flappy-kernel}
MIRAGE=${MIRAGE:-/root/mirage}
MODEL=${MODEL:-Qwen/Qwen3-8B}

# Start from a clean request queue and a fresh shared-memory block.
rm -rf "$SERVE_DIR/queue" "$SERVE_DIR/done" "$SERVE_DIR/current.json" "$SERVE_DIR/stop" /dev/shm/flappy
mkdir -p "$SERVE_DIR"
# The server reads the shared block at the offsets of the code Mirage compiles,
# so rebuild the tools from the same sources before exporting the layout.
make -s -C "$FK/kernel"
"$FK/kernel/standalone" layout > "$FK/server/layout.json"

(cd "$MIRAGE/demo/qwen3" && exec env COTENANT_MODE=4 COTENANT_SHM=/flappy COTENANT_QUIET=1 \
    MPK_COTENANT_BLOCKS="${COTENANT_BLOCKS:-32}" MPK_COTENANT_FUSED=1 \
    MPK_NUM_WORKERS="${LLM_WORKERS:-96}" \
    MPK_SERVE_DIR="$SERVE_DIR" MPK_COTENANT_DIR="$FK/kernel" \
    python demo.py --use-mirage --model "$MODEL" --max-seq-length "${MAX_SEQ:-1024}") &
GPU_PID=$!
python "$FK/server/app.py" --shm /flappy --layout "$FK/server/layout.json" \
    --serve-dir "$SERVE_DIR" --model "$MODEL" --port "${PORT:-8080}" "$@" &
WEB_PID=$!

# Whichever process ends first ends the arcade, so a supervisor (Docker,
# Kubernetes) restarts both together.
stop() {
  touch "$SERVE_DIR/stop"
  kill "$WEB_PID" 2>/dev/null || true
  sleep 5
  kill "$GPU_PID" 2>/dev/null || true
}
trap stop TERM INT
set +e
wait -n "$GPU_PID" "$WEB_PID"
status=$?
stop
exit $(( status == 0 ? 1 : status ))
