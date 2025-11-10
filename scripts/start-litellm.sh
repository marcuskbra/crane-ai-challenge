#!/bin/bash
#
# Start LiteLLM proxy in background
#
# This script starts the LiteLLM proxy server as a background process
# and saves the PID for later management.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.litellm.pid"
LOG_FILE="/tmp/litellm.log"
CONFIG_FILE="$PROJECT_ROOT/config/litellm_config.yaml"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  LiteLLM is already running (PID: $PID)"
        echo "   View logs: tail -f $LOG_FILE"
        exit 0
    else
        echo "🧹 Cleaning up stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo "   Start it with: brew services start ollama"
    exit 1
fi

echo "🚀 Starting LiteLLM proxy..."
echo "   Config: $CONFIG_FILE"
echo "   Logs:   $LOG_FILE"

# Start LiteLLM in background
nohup litellm --config "$CONFIG_FILE" --port 4000 > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo "$PID" > "$PID_FILE"

# Wait a moment and check if it's still running
sleep 2
if ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ LiteLLM started successfully (PID: $PID)"
    echo "   API: http://localhost:4000"
    echo "   View logs: tail -f $LOG_FILE"
    echo "   Stop: make llm-stop"
else
    echo "❌ LiteLLM failed to start"
    echo "   Check logs: cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
