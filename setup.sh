#!/bin/bash

# =============================================================================
# Reranker Service - Setup Script for Linux/macOS
# =============================================================================
# This script sets up the virtual environment and installs dependencies.
# Usage: ./setup.sh [--dev] [--cuda]
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
PYTHON_VERSION="3.12"

# Parse arguments
INSTALL_DEV=false
INSTALL_CUDA=false

for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --cuda)
            INSTALL_CUDA=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [--dev] [--cuda]"
            echo ""
            echo "Options:"
            echo "  --dev   Install development dependencies"
            echo "  --cuda  Install CUDA-enabled PyTorch"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Reranker Service - Setup Script${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check for Python
echo -e "${YELLOW}Checking Python installation...${NC}"
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python is not installed!${NC}"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}Found Python $PYTHON_VER${NC}"

# Check Python version
MAJOR_VER=$(echo $PYTHON_VER | cut -d'.' -f1)
MINOR_VER=$(echo $PYTHON_VER | cut -d'.' -f2)

if [ "$MAJOR_VER" -lt 3 ] || ([ "$MAJOR_VER" -eq 3 ] && [ "$MINOR_VER" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required!${NC}"
    exit 1
fi

# Check virtual environment exists
echo ""
echo -e "${YELLOW}Checking virtual environment in ${VENV_DIR}...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_DIR}${NC}"
    echo ""
    echo "Please create a virtual environment first:"
    echo "  $PYTHON_CMD -m venv $VENV_DIR"
    echo ""
    exit 1
fi

echo -e "${GREEN}Virtual environment found!${NC}"

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}Virtual environment activated!${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch (with or without CUDA)
echo ""
if [ "$INSTALL_CUDA" = true ]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${YELLOW}Installing PyTorch (CPU/MPS)...${NC}"
    # This will install the appropriate version for the platform (including MPS for macOS)
    pip install torch torchvision torchaudio
fi

# Install main dependencies
echo ""
echo -e "${YELLOW}Installing main dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    echo ""
    echo -e "${YELLOW}Installing development dependencies...${NC}"
    pip install -r requirements-dev.txt
fi

# Detect platform and provide MPS info for macOS
echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}  macOS Detected - MPS Optimization${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""
    echo -e "${GREEN}Apple Silicon (MPS) support is enabled!${NC}"
    echo "The service will automatically use MPS for GPU acceleration."
    echo ""
    echo "To verify MPS support, run:"
    echo "  python -c \"import torch; print('MPS available:', torch.backends.mps.is_available())\""
fi

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo "To activate the virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start the server:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m src.main"
echo ""
