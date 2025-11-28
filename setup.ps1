# =============================================================================
# Reranker Service - Setup Script for Windows
# =============================================================================
# This script sets up the virtual environment and installs dependencies.
# Usage: .\setup.ps1 [-Dev] [-Cuda]
# =============================================================================

param(
    [switch]$Dev,
    [switch]$Cuda,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$VENV_DIR = "venv"

function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

if ($Help) {
    Write-Output "Usage: .\setup.ps1 [-Dev] [-Cuda]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Dev    Install development dependencies"
    Write-Output "  -Cuda   Install CUDA-enabled PyTorch"
    Write-Output "  -Help   Show this help message"
    exit 0
}

Write-Output ""
Write-ColorOutput Cyan "======================================="
Write-ColorOutput Cyan "  Reranker Service - Setup Script"
Write-ColorOutput Cyan "======================================="
Write-Output ""

# Check for Python
Write-ColorOutput Yellow "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-ColorOutput Green "Found $pythonVersion"
} catch {
    Write-ColorOutput Red "Error: Python is not installed!"
    Write-Output "Please install Python 3.10 or higher."
    exit 1
}

# Parse version
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $majorVer = [int]$Matches[1]
    $minorVer = [int]$Matches[2]
    
    if ($majorVer -lt 3 -or ($majorVer -eq 3 -and $minorVer -lt 10)) {
        Write-ColorOutput Red "Error: Python 3.10 or higher is required!"
        exit 1
    }
}

# Check virtual environment exists
Write-Output ""
Write-ColorOutput Yellow "Checking virtual environment in $VENV_DIR..."

if (-not (Test-Path $VENV_DIR)) {
    Write-ColorOutput Red "Error: Virtual environment not found at $VENV_DIR"
    Write-Output ""
    Write-Output "Please create a virtual environment first:"
    Write-Output "  python -m venv $VENV_DIR"
    Write-Output ""
    exit 1
}

Write-ColorOutput Green "Virtual environment found!"

# Activate virtual environment
Write-Output ""
Write-ColorOutput Yellow "Activating virtual environment..."
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-ColorOutput Green "Virtual environment activated!"
} else {
    Write-ColorOutput Red "Error: Cannot find activation script!"
    exit 1
}

# Upgrade pip
Write-Output ""
Write-ColorOutput Yellow "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with or without CUDA)
Write-Output ""
if ($Cuda) {
    Write-ColorOutput Yellow "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-ColorOutput Yellow "Installing PyTorch (CPU)..."
    pip install torch torchvision torchaudio
}

# Install main dependencies
Write-Output ""
Write-ColorOutput Yellow "Installing main dependencies..."
pip install -r requirements.txt

# Install development dependencies if requested
if ($Dev) {
    Write-Output ""
    Write-ColorOutput Yellow "Installing development dependencies..."
    pip install -r requirements-dev.txt
}

Write-Output ""
Write-ColorOutput Green "======================================="
Write-ColorOutput Green "  Setup Complete!"
Write-ColorOutput Green "======================================="
Write-Output ""
Write-Output "To activate the virtual environment:"
Write-Output "  .\$VENV_DIR\Scripts\Activate.ps1"
Write-Output ""
Write-Output "To start the server:"
Write-Output "  .\run.ps1"
Write-Output ""
Write-Output "Or manually:"
Write-Output "  .\$VENV_DIR\Scripts\Activate.ps1"
Write-Output "  python -m src.main"
Write-Output ""
