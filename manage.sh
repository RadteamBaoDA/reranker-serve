#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="reranker.pid"
LOG_DIR="logs"
ENV_FILE=".env"

# Export environment variables from .env file
load_env() {
    if [ -f "$ENV_FILE" ]; then
        echo "Loading environment from $ENV_FILE..."
        set -a  # automatically export all variables
        source "$ENV_FILE"
        set +a
    else
        echo "Warning: $ENV_FILE not found. Using default settings."
    fi
}

# Determine Python command
if [ -f "venv/Scripts/python" ]; then
    PYTHON_CMD="venv/Scripts/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python"
fi

# Check if python is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: python command not found. Please activate your virtual environment."
    exit 1
fi

# Clean up old log files based on retention days
cleanup_old_logs() {
    RETENTION_DAYS=${RERANKER_LOG_RETENTION_DAYS:-7}
    LOG_DIR=${RERANKER_LOG_DIR:-./logs}

    if [ -d "$LOG_DIR" ]; then
        echo "Cleaning up log files older than $RETENTION_DAYS days in $LOG_DIR..."
        find "$LOG_DIR" -name "log_*.log*" -type f -mtime +$RETENTION_DAYS -exec rm -f {} \; 2>/dev/null
    fi
}start() {
    # Load environment variables first
    load_env
    
    # Clean up old logs
    cleanup_old_logs
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        # Check if process is running
        if ps -p $PID > /dev/null 2>&1; then
            echo "Service is already running with PID $PID."
            return
        else
            echo "PID file exists but process is not running. Removing PID file."
            rm "$PID_FILE"
        fi
    fi

    echo "Starting Reranker Service..."
    # Run in background, redirect stdout/stderr to /dev/null since we use internal file logging
    # Use nohup to prevent SIGHUP when terminal closes
    nohup $PYTHON_CMD -m src.main > /dev/null 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Wait a moment to check if it stays running
    sleep 2
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Service failed to start immediately. Check reranker.log for details."
        rm "$PID_FILE"
        exit 1
    fi
    
    echo "Service started with PID $PID."
    LOG_DIR=${RERANKER_LOG_DIR:-./logs}
    echo "Logs are being written to $LOG_DIR/log_yyyymmddhhmmss.log"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Service is not running (PID file not found)."
        return
    fi

    PID=$(cat "$PID_FILE")
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping service with PID $PID..."
        kill $PID
        
        # Wait for process to exit (timeout after 10 seconds)
        count=0
        while ps -p $PID > /dev/null 2>&1; do
            sleep 1
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "Process did not exit gracefully. Forcing kill..."
                kill -9 $PID
                break
            fi
        done
        echo "Service stopped."
    else
        echo "Process $PID not found."
    fi
    rm "$PID_FILE"
}

restart() {
    stop
    sleep 2
    start
}

status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Service is not running (PID file not found)."
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Service is running with PID $PID."
        return 0
    else
        echo "Service is not running (process $PID not found)."
        return 1
    fi
}

logs() {
    LOG_DIR=${RERANKER_LOG_DIR:-./logs}
    # Find the most recent log file in the log directory
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(ls -t "$LOG_DIR"/log_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            echo "Tailing $LATEST_LOG..."
            tail -f "$LATEST_LOG"
        else
            echo "No log files found in $LOG_DIR matching log_*.log pattern."
        fi
    else
        echo "Log directory $LOG_DIR not found."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        load_env
        status
        ;;
    logs)
        load_env
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
