#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
pid=$(cat bpu/results/web_ui_acceptance/server.pid)
if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Web lab is already stopped.'
    exit 0
fi
if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'scripts/web_embedding_server.py'; then
    echo 'PID belongs to a different process; refusing to stop it.' >&2
    exit 1
fi
kill "$pid"
echo "Stopped web lab PID $pid."
