# =============================================================================
# Reranker Service - Run Script for Windows
# =============================================================================
# This script runs the reranker service with optimal settings.
# Usage: .\run.ps1 [-Dev] [-Workers N] [-Port PORT] [-Host HOST]
# =============================================================================

param(
    [switch]$Dev,
    [int]$Workers = 0,
    [int]$Port = 0,
    [string]$Host = "",
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$VENV_DIR = "venv"
$ENV_FILE = ".env"

# Load environment variables from .env file if it exists
if (Test-Path $ENV_FILE) {
    Write-ColorOutput Yellow "Loading environment variables from $ENV_FILE..."
    Get-Content $ENV_FILE | ForEach-Object {
        $line = $_.Trim()
        # Skip comments and empty lines
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line -split "=", 2
            if ($parts.Count -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()
                # Remove quotes if present
                $value = $value -replace '^["'']|["'']$', ''
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
            }
        }
    }
}

# Default settings from environment or defaults
$DEFAULT_HOST = if ($env:RERANKER_HOST) { $env:RERANKER_HOST } else { "0.0.0.0" }
$DEFAULT_PORT = if ($env:RERANKER_PORT) { [int]$env:RERANKER_PORT } else { 8000 }
$DEFAULT_WORKERS = if ($env:RERANKER_WORKERS) { [int]$env:RERANKER_WORKERS } else { 1 }
$LOG_LEVEL = if ($env:RERANKER_LOG_LEVEL) { $env:RERANKER_LOG_LEVEL } else { "info" }

# Apply parameters or defaults
if ($Port -eq 0) { $Port = $DEFAULT_PORT }
if ($Workers -eq 0) { $Workers = $DEFAULT_WORKERS }
if ($Host -eq "") { $Host = $DEFAULT_HOST }

function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

if ($Help) {
    Write-Output ""
    Write-Output "Usage: .\run.ps1 [options]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Dev              Enable development mode with auto-reload"
    Write-Output "  -Workers N        Number of workers (default: 1)"
    Write-Output "  -Port PORT        Server port (default: 8000)"
    Write-Output "  -Host HOST        Server host (default: 0.0.0.0)"
    Write-Output "  -Help             Show this help message"
    Write-Output ""
    Write-Output "Environment Variables:"
    Write-Output "  Variables are loaded from .env file if present."
    Write-Output "  See .env.example for all available options."
    Write-Output ""
    Write-Output "  Key variables:"
    Write-Output "  RERANKER_HOST           Server host"
    Write-Output "  RERANKER_PORT           Server port"
    Write-Output "  RERANKER_WORKERS        Number of workers"
    Write-Output "  RERANKER_MODEL_NAME     Model to use"
    Write-Output "  RERANKER_MODEL_PATH     Local model path"
    Write-Output "  RERANKER_USE_OFFLINE_MODE  Use offline mode"
    Write-Output "  RERANKER_DEVICE         Device (cuda, cpu)"
    Write-Output "  RERANKER_API_KEY        API key for authentication"
    Write-Output "  RERANKER_LOG_LEVEL      Log level (DEBUG, INFO, WARNING, ERROR)"
    Write-Output "  RERANKER_ENABLE_ASYNC_ENGINE  Enable async engine"
    exit 0
}

Write-Output ""
Write-ColorOutput Cyan "======================================="
Write-ColorOutput Cyan "  Reranker Service"
Write-ColorOutput Cyan "======================================="
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

# Check if dependencies are installed
try {
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-ColorOutput Red "Error: Dependencies not installed!"
    Write-Output "Please run .\setup.ps1 first."
    exit 1
}

# Detect device
Write-ColorOutput Yellow "Detecting device..."
$DEVICE = python -c @"
import torch
if torch.cuda.is_available():
    print('cuda')
else:
    print('cpu')
"@ 2>$null

if (-not $DEVICE) { $DEVICE = "cpu" }

Write-ColorOutput Green "Using device: $DEVICE"

# Set environment variable for device if not already set
if (-not $env:RERANKER_DEVICE) {
    $env:RERANKER_DEVICE = $DEVICE
}

# Display configuration
Write-Output ""
Write-ColorOutput Cyan "Configuration:"
Write-Output "  Host: $Host"
Write-Output "  Port: $Port"
Write-Output "  Workers: $Workers"
Write-Output "  Device: $env:RERANKER_DEVICE"
$modelName = if ($env:RERANKER_MODEL_NAME) { $env:RERANKER_MODEL_NAME } else { "BAAI/bge-reranker-v2-m3" }
Write-Output "  Model: $modelName"
if ($env:RERANKER_MODEL_PATH) {
    Write-Output "  Model Path: $env:RERANKER_MODEL_PATH"
}
Write-Output ""

# Run the server
if ($Dev) {
    Write-ColorOutput Yellow "Starting server in development mode..."
    Write-Output ""
    uvicorn src.main:app --host $Host --port $Port --reload --log-level $LOG_LEVEL
} else {
    Write-ColorOutput Yellow "Starting server in production mode..."
    Write-Output ""
    
    if ($Workers -gt 1) {
        uvicorn src.main:app --host $Host --port $Port --workers $Workers --log-level $LOG_LEVEL
    } else {
        uvicorn src.main:app --host $Host --port $Port --log-level $LOG_LEVEL
    }
}
