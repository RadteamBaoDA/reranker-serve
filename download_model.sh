#!/bin/bash

# =============================================================================
# Reranker Service - Download Model Script for Linux/macOS
# =============================================================================
# This script downloads a model for offline use.
# Usage: ./download_model.sh [model_name] [output_dir]
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
DEFAULT_MODEL="BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT_DIR="./models"

# Parse arguments
MODEL_NAME="${1:-$DEFAULT_MODEL}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Reranker Service - Model Download${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo "Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download the model
echo -e "${YELLOW}Downloading model...${NC}"
echo ""

python << EOF
import os
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download

model_name = "$MODEL_NAME"
output_dir = "$OUTPUT_DIR"

# Create model-specific directory
model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading {model_name} to {model_dir}...")

try:
    # Download using huggingface_hub
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"\nModel downloaded successfully to: {model_dir}")
    print(f"\nTo use this model offline, set:")
    print(f"  export RERANKER_MODEL_PATH={model_dir}")
    print(f"  export RERANKER_USE_OFFLINE_MODE=true")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
EOF

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Download Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
