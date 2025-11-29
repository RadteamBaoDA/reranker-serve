#!/bin/bash

# =============================================================================
# Reranker Service - Daemon Script for Linux/macOS
# =============================================================================
# This script runs the reranker service as a background daemon using nohup.
# It reads configuration from .env file and manages the server process.
# Usage: ./daemon.sh [start|stop|restart|status]
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PID_FILE="$SCRIPT_DIR/.reranker.pid"
LOG_FILE="$SCRIPT_DIR/logs/reranker.log"
ENV_FILE="$SCRIPT_DIR/.env"

# Default settings (can be overridden by .env)
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
LOG_LEVEL="info"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables from .env file
load_env() {
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from $ENV_FILE"
        
        # Export variables from .env file
        set -a
        # Read .env file, skip comments and empty lines
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            if [[ ! "$key" =~ ^# && -n "$key" ]]; then
                # Remove leading/trailing whitespace and quotes
                key=$(echo "$key" | xargs)
                value=$(echo "$value" | xargs | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
                
                if [ -n "$key" ] && [ -n "$value" ]; then
                    export "$key=$value"
                fi
            fi
        done < "$ENV_FILE"
        set +a
        
        # Update local variables from environment
        HOST="${RERANKER_HOST:-$HOST}"
        PORT="${RERANKER_PORT:-$PORT}"
        WORKERS="${RERANKER_WORKERS:-$WORKERS}"
        LOG_LEVEL="${RERANKER_LOG_LEVEL:-$LOG_LEVEL}"
        
        log_success "Environment loaded"
    else
        log_warning ".env file not found at $ENV_FILE"
        log_info "Using default configuration"
    fi
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_info "Please run ./setup.sh first"
        exit 1
    fi
}

# Check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Get the PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# =============================================================================
# Daemon Control Functions
# =============================================================================

start_daemon() {
    log_info "Starting Reranker Service daemon..."
    
    # Check if already running
    if is_running; then
        log_warning "Service is already running (PID: $(get_pid))"
        return 1
    fi
    
    # Check virtual environment
    check_venv
    
    # Load environment
    load_env
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Detect device
    log_info "Detecting device..."
    DEVICE=$("$VENV_DIR/bin/python" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu")
    
    export RERANKER_DEVICE="${RERANKER_DEVICE:-$DEVICE}"
    
    # Display configuration
    echo ""
    log_info "Configuration:"
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  Workers: $WORKERS"
    echo "  Device: $RERANKER_DEVICE"
    echo "  Model: ${RERANKER_MODEL_NAME:-BAAI/bge-reranker-v2-m3}"
    echo "  Log File: $LOG_FILE"
    echo ""
    
    # Start the server with nohup
    log_info "Starting server with nohup..."
    
    cd "$SCRIPT_DIR"
    
    nohup "$VENV_DIR/bin/python" -m uvicorn src.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        >> "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    
    # Wait a moment and check if started
    sleep 2
    
    if is_running; then
        log_success "Service started successfully (PID: $(get_pid))"
        log_info "Logs: tail -f $LOG_FILE"
        log_info "API Docs: http://$HOST:$PORT/docs"
    else
        log_error "Failed to start service"
        log_info "Check logs: cat $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_daemon() {
    log_info "Stopping Reranker Service daemon..."
    
    if ! is_running; then
        log_warning "Service is not running"
        rm -f "$PID_FILE"
        return 0
    fi
    
    PID=$(get_pid)
    log_info "Sending SIGTERM to PID $PID..."
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$PID" 2>/dev/null
    
    # Wait for process to stop
    TIMEOUT=30
    COUNTER=0
    while is_running && [ $COUNTER -lt $TIMEOUT ]; do
        sleep 1
        COUNTER=$((COUNTER + 1))
        echo -n "."
    done
    echo ""
    
    # Force kill if still running
    if is_running; then
        log_warning "Process did not stop gracefully, sending SIGKILL..."
        kill -9 "$PID" 2>/dev/null
        sleep 1
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    if ! is_running; then
        log_success "Service stopped successfully"
    else
        log_error "Failed to stop service"
        return 1
    fi
}

restart_daemon() {
    log_info "Restarting Reranker Service daemon..."
    stop_daemon
    sleep 2
    start_daemon
}

show_status() {
    echo ""
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}  Reranker Service Status${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""
    
    if is_running; then
        PID=$(get_pid)
        echo -e "Status: ${GREEN}Running${NC}"
        echo "PID: $PID"
        
        # Show process info
        if command -v ps &> /dev/null; then
            echo ""
            echo "Process Info:"
            ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,command 2>/dev/null | head -2
        fi
        
        # Show port binding
        if command -v lsof &> /dev/null; then
            echo ""
            echo "Listening on:"
            lsof -i -P -n | grep "$PID" | grep LISTEN 2>/dev/null || echo "  (unable to detect)"
        fi
    else
        echo -e "Status: ${RED}Stopped${NC}"
    fi
    
    # Show log file info
    echo ""
    if [ -f "$LOG_FILE" ]; then
        echo "Log File: $LOG_FILE"
        echo "Log Size: $(du -h "$LOG_FILE" 2>/dev/null | cut -f1)"
        echo ""
        echo "Last 5 log lines:"
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
    else
        echo "Log File: (not created yet)"
    fi
    
    echo ""
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log_error "Log file not found: $LOG_FILE"
        exit 1
    fi
}

show_help() {
    echo ""
    echo "Reranker Service Daemon Manager"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start the service as a background daemon"
    echo "  stop      Stop the running daemon"
    echo "  restart   Restart the daemon"
    echo "  status    Show daemon status"
    echo "  logs      Follow the log file (tail -f)"
    echo "  help      Show this help message"
    echo ""
    echo "Configuration:"
    echo "  The script reads configuration from .env file."
    echo "  Copy .env.example to .env and modify as needed."
    echo ""
    echo "Files:"
    echo "  PID File: $PID_FILE"
    echo "  Log File: $LOG_FILE"
    echo "  Env File: $ENV_FILE"
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================

case "${1:-}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        restart_daemon
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -n "${1:-}" ]; then
            log_error "Unknown command: $1"
        fi
        show_help
        exit 1
        ;;
esac
