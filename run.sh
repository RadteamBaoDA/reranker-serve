#!/bin/bash

# =============================================================================
# Reranker Service - Run Script for Linux/macOS
# =============================================================================
# This script runs the reranker service with optimal settings.
# Usage: ./run.sh [--dev] [--workers N]
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
ENV_FILE=".env"

# Load environment variables from .env file if it exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Loading environment variables from $ENV_FILE...${NC}"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
fi

# Default settings
HOST="${RERANKER_HOST:-0.0.0.0}"
PORT="${RERANKER_PORT:-8000}"
WORKERS="${RERANKER_WORKERS:-1}"
LOG_LEVEL="${RERANKER_LOG_LEVEL:-info}"
DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run.sh [options]"
            echo ""
            echo "Options:"
            echo "  --dev              Enable development mode with auto-reload"
            echo "  --workers N        Number of workers (default: 1)"
            echo "  --port PORT        Server port (default: 8000)"
            echo "  --host HOST        Server host (default: 0.0.0.0)"
            echo ""
            echo "Environment Variables:"
            echo "  Variables are loaded from .env file if present."
            echo "  See .env.example for all available options."
            echo ""
            echo "  Key variables:"
            echo "  RERANKER_HOST           Server host"
            echo "  RERANKER_PORT           Server port"
            echo "  RERANKER_WORKERS        Number of workers"
            echo "  RERANKER_MODEL_NAME     Model to use"
            echo "  RERANKER_MODEL_PATH     Local model path"
            echo "  RERANKER_USE_OFFLINE_MODE  Use offline mode"
            echo "  RERANKER_DEVICE         Device (cuda, mps, cpu)"
            echo "  RERANKER_API_KEY        API key for authentication"
            echo "  RERANKER_LOG_LEVEL      Log level (DEBUG, INFO, WARNING, ERROR)"
            echo "  RERANKER_ENABLE_ASYNC_ENGINE  Enable async engine"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Reranker Service${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check if dependencies are installed
if ! python -c "import fastapi" &> /dev/null; then
    echo -e "${RED}Error: Dependencies not installed!${NC}"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Detect device
echo -e "${YELLOW}Detecting device...${NC}"
DEVICE=$(python -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu")

echo -e "${GREEN}Using device: $DEVICE${NC}"

# Set environment variable for device if not already set
export RERANKER_DEVICE="${RERANKER_DEVICE:-$DEVICE}"

# Set config file path if config.yml exists
if [ -f "config.yml" ]; then
    export RERANKER_CONFIG_PATH="config.yml"
fi

# Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Device: $RERANKER_DEVICE"
echo "  Model: ${RERANKER_MODEL_NAME:-BAAI/bge-reranker-v2-m3}"
if [ -n "$RERANKER_MODEL_PATH" ]; then
    echo "  Model Path: $RERANKER_MODEL_PATH"
fi
echo ""

# Run the server
if [ "$DEV_MODE" = true ]; then
    echo -e "${YELLOW}Starting server in development mode...${NC}"
    echo ""
    uvicorn src.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo -e "${YELLOW}Starting server in production mode...${NC}"
    echo ""
    
    if [ "$WORKERS" -gt 1 ]; then
        # Use multiple workers
        uvicorn src.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level "$LOG_LEVEL"
    else
        # Single worker
        uvicorn src.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --log-level "$LOG_LEVEL"
    fi
fi
