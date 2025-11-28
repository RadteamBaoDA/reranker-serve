# =============================================================================
# Reranker Service - Download Model Script for Windows
# =============================================================================
# This script downloads a model for offline use.
# Usage: .\download_model.ps1 [ModelName] [OutputDir]
# =============================================================================

param(
    [Parameter(Position=0)]
    [string]$ModelName = "BAAI/bge-reranker-v2-m3",
    
    [Parameter(Position=1)]
    [string]$OutputDir = ".\models",
    
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
    Write-Output ""
    Write-Output "Reranker Service - Model Download Script"
    Write-Output ""
    Write-Output "Usage: .\download_model.ps1 [ModelName] [OutputDir]"
    Write-Output ""
    Write-Output "Parameters:"
    Write-Output "  ModelName   Model name from HuggingFace (default: BAAI/bge-reranker-v2-m3)"
    Write-Output "  OutputDir   Output directory for models (default: .\models)"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output "  .\download_model.ps1"
    Write-Output "  .\download_model.ps1 BAAI/bge-reranker-large"
    Write-Output "  .\download_model.ps1 BAAI/bge-reranker-v2-m3 C:\models"
    Write-Output ""
    exit 0
}

Write-Output ""
Write-ColorOutput Cyan "======================================="
Write-ColorOutput Cyan "  Reranker Service - Model Download"
Write-ColorOutput Cyan "======================================="
Write-Output ""
Write-Output "Model: $ModelName"
Write-Output "Output Directory: $OutputDir"
Write-Output ""

# Check if virtual environment exists
if (-not (Test-Path $VENV_DIR)) {
    Write-ColorOutput Red "Error: Virtual environment not found!"
    Write-Output "Please run .\setup.ps1 first."
    exit 1
}

# Activate virtual environment
Write-ColorOutput Yellow "Activating virtual environment..."
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-ColorOutput Red "Error: Cannot find activation script!"
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Download the model
Write-ColorOutput Yellow "Downloading model..."
Write-Output ""

$pythonScript = @"
import os
from huggingface_hub import snapshot_download

model_name = "$ModelName"
output_dir = r"$OutputDir"

# Create model-specific directory
model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading {model_name} to {model_dir}...")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"\nModel downloaded successfully to: {model_dir}")
    print(f"\nTo use this model offline, set environment variables:")
    print(f"  `$env:RERANKER_MODEL_PATH='{model_dir}'")
    print(f"  `$env:RERANKER_USE_OFFLINE_MODE='true'")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
"@

python -c $pythonScript

Write-Output ""
Write-ColorOutput Green "======================================="
Write-ColorOutput Green "  Download Complete!"
Write-ColorOutput Green "======================================="
Write-Output ""
