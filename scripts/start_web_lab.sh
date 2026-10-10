#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
result=bpu/results/web_ui_acceptance
mkdir -p "$result"
if [ -f "$result/server.pid" ]; then
    pid=$(cat "$result/server.pid")
    if kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'scripts/web_embedding_server.py'; then
        echo "Web lab already running, PID $pid: http://<BOARD_IP>:8088"
        exit 0
    fi
fi
.venv/bin/python -m py_compile scripts/web_embedding_engine.py scripts/web_embedding_server.py
nohup .venv/bin/python -u scripts/web_embedding_server.py --host 0.0.0.0 --port 8088 >> "$result/server.log" 2>&1 < /dev/null &
echo $! > "$result/server.pid"
echo "Started PID $(cat "$result/server.pid"); startup status: http://<BOARD_IP>:8088/api/status"
