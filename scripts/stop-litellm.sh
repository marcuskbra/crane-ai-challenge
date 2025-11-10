#!/bin/bash
#
# Stop LiteLLM proxy
#
# This script stops the LiteLLM proxy server by killing the process
# saved in the PID file.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.litellm.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  LiteLLM is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "ℹ️  LiteLLM process not found (cleaning up stale PID file)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "🛑 Stopping LiteLLM (PID: $PID)..."
kill "$PID"

# Wait for process to terminate
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ LiteLLM stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Force killing LiteLLM..."
    kill -9 "$PID"
    sleep 1
fi

rm -f "$PID_FILE"
echo "✅ LiteLLM stopped"
