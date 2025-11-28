# =============================================================================
# Reranker Service - Daemon Script for Windows
# =============================================================================
# This script runs the reranker service as a background process.
# It reads configuration from .env file and manages the server process.
# Usage: .\daemon.ps1 [start|stop|restart|status]
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "help")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $SCRIPT_DIR "venv"
$PID_FILE = Join-Path $SCRIPT_DIR ".reranker.pid"
$LOG_DIR = Join-Path $SCRIPT_DIR "logs"
$LOG_FILE = Join-Path $LOG_DIR "reranker.log"
$ENV_FILE = Join-Path $SCRIPT_DIR ".env"

# Default settings
$DEFAULT_HOST = "0.0.0.0"
$DEFAULT_PORT = "8000"
$DEFAULT_WORKERS = "1"
$DEFAULT_LOG_LEVEL = "info"

function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info($Message) {
    Write-ColorOutput Cyan "[INFO] $Message"
}

function Write-Success($Message) {
    Write-ColorOutput Green "[SUCCESS] $Message"
}

function Write-Warning($Message) {
    Write-ColorOutput Yellow "[WARNING] $Message"
}

function Write-Error($Message) {
    Write-ColorOutput Red "[ERROR] $Message"
}

# Load environment variables from .env file
function Load-EnvFile {
    if (Test-Path $ENV_FILE) {
        Write-Info "Loading environment from $ENV_FILE"
        
        Get-Content $ENV_FILE | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#")) {
                $parts = $line -split "=", 2
                if ($parts.Count -eq 2) {
                    $key = $parts[0].Trim()
                    $value = $parts[1].Trim().Trim('"').Trim("'")
                    [Environment]::SetEnvironmentVariable($key, $value, "Process")
                }
            }
        }
        
        Write-Success "Environment loaded"
    } else {
        Write-Warning ".env file not found at $ENV_FILE"
        Write-Info "Using default configuration"
    }
    
    # Set defaults from environment or use defaults
    $script:HOST = if ($env:RERANKER_HOST) { $env:RERANKER_HOST } else { $DEFAULT_HOST }
    $script:PORT = if ($env:RERANKER_PORT) { $env:RERANKER_PORT } else { $DEFAULT_PORT }
    $script:WORKERS = if ($env:RERANKER_WORKERS) { $env:RERANKER_WORKERS } else { $DEFAULT_WORKERS }
    $script:LOG_LEVEL = if ($env:RERANKER_LOG_LEVEL) { $env:RERANKER_LOG_LEVEL } else { $DEFAULT_LOG_LEVEL }
}

function Check-Venv {
    if (-not (Test-Path $VENV_DIR)) {
        Write-Error "Virtual environment not found at $VENV_DIR"
        Write-Info "Please run .\setup.ps1 first"
        exit 1
    }
}

function Get-ProcessId {
    if (Test-Path $PID_FILE) {
        return Get-Content $PID_FILE
    }
    return $null
}

function Test-Running {
    $pid = Get-ProcessId
    if ($pid) {
        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            return $process -ne $null
        } catch {
            return $false
        }
    }
    return $false
}

function Start-Daemon {
    Write-Info "Starting Reranker Service daemon..."
    
    if (Test-Running) {
        Write-Warning "Service is already running (PID: $(Get-ProcessId))"
        return
    }
    
    Check-Venv
    Load-EnvFile
    
    # Create logs directory
    if (-not (Test-Path $LOG_DIR)) {
        New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
    }
    
    # Detect device
    Write-Info "Detecting device..."
    $pythonExe = Join-Path $VENV_DIR "Scripts\python.exe"
    
    $DEVICE = & $pythonExe -c @"
import torch
if torch.cuda.is_available():
    print('cuda')
else:
    print('cpu')
"@ 2>$null
    
    if (-not $DEVICE) { $DEVICE = "cpu" }
    
    if (-not $env:RERANKER_DEVICE) {
        $env:RERANKER_DEVICE = $DEVICE
    }
    
    # Display configuration
    Write-Output ""
    Write-Info "Configuration:"
    Write-Output "  Host: $HOST"
    Write-Output "  Port: $PORT"
    Write-Output "  Workers: $WORKERS"
    Write-Output "  Device: $env:RERANKER_DEVICE"
    $modelName = if ($env:RERANKER_MODEL_NAME) { $env:RERANKER_MODEL_NAME } else { "BAAI/bge-reranker-v2-m3" }
    Write-Output "  Model: $modelName"
    Write-Output "  Log File: $LOG_FILE"
    Write-Output ""
    
    # Start the server as background process
    Write-Info "Starting server as background process..."
    
    $uvicornPath = Join-Path $VENV_DIR "Scripts\uvicorn.exe"
    
    $processArgs = @{
        FilePath = $pythonExe
        ArgumentList = "-m", "uvicorn", "src.main:app", "--host", $HOST, "--port", $PORT, "--workers", $WORKERS, "--log-level", $LOG_LEVEL
        WorkingDirectory = $SCRIPT_DIR
        RedirectStandardOutput = $LOG_FILE
        RedirectStandardError = $LOG_FILE
        NoNewWindow = $true
        PassThru = $true
    }
    
    $process = Start-Process @processArgs
    
    # Save PID
    $process.Id | Out-File -FilePath $PID_FILE -NoNewline
    
    # Wait and check
    Start-Sleep -Seconds 3
    
    if (Test-Running) {
        Write-Success "Service started successfully (PID: $(Get-ProcessId))"
        Write-Info "Logs: Get-Content $LOG_FILE -Wait"
        Write-Info "API Docs: http://${HOST}:${PORT}/docs"
    } else {
        Write-Error "Failed to start service"
        Write-Info "Check logs: Get-Content $LOG_FILE"
        Remove-Item -Path $PID_FILE -ErrorAction SilentlyContinue
    }
}

function Stop-Daemon {
    Write-Info "Stopping Reranker Service daemon..."
    
    if (-not (Test-Running)) {
        Write-Warning "Service is not running"
        Remove-Item -Path $PID_FILE -ErrorAction SilentlyContinue
        return
    }
    
    $pid = Get-ProcessId
    Write-Info "Stopping process $pid..."
    
    try {
        Stop-Process -Id $pid -Force
        Start-Sleep -Seconds 2
    } catch {
        Write-Warning "Could not stop process gracefully"
    }
    
    # Clean up PID file
    Remove-Item -Path $PID_FILE -ErrorAction SilentlyContinue
    
    if (-not (Test-Running)) {
        Write-Success "Service stopped successfully"
    } else {
        Write-Error "Failed to stop service"
    }
}

function Restart-Daemon {
    Write-Info "Restarting Reranker Service daemon..."
    Stop-Daemon
    Start-Sleep -Seconds 2
    Start-Daemon
}

function Show-Status {
    Write-Output ""
    Write-ColorOutput Cyan "======================================="
    Write-ColorOutput Cyan "  Reranker Service Status"
    Write-ColorOutput Cyan "======================================="
    Write-Output ""
    
    if (Test-Running) {
        $pid = Get-ProcessId
        Write-ColorOutput Green "Status: Running"
        Write-Output "PID: $pid"
        
        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                Write-Output ""
                Write-Output "Process Info:"
                Write-Output "  CPU Time: $($process.TotalProcessorTime)"
                Write-Output "  Memory: $([math]::Round($process.WorkingSet64 / 1MB, 2)) MB"
                Write-Output "  Start Time: $($process.StartTime)"
            }
        } catch {}
    } else {
        Write-ColorOutput Red "Status: Stopped"
    }
    
    Write-Output ""
    if (Test-Path $LOG_FILE) {
        $logSize = (Get-Item $LOG_FILE).Length / 1KB
        Write-Output "Log File: $LOG_FILE"
        Write-Output "Log Size: $([math]::Round($logSize, 2)) KB"
        Write-Output ""
        Write-Output "Last 5 log lines:"
        Get-Content $LOG_FILE -Tail 5 | ForEach-Object { Write-Output "  $_" }
    } else {
        Write-Output "Log File: (not created yet)"
    }
    Write-Output ""
}

function Show-Logs {
    if (Test-Path $LOG_FILE) {
        Get-Content $LOG_FILE -Wait
    } else {
        Write-Error "Log file not found: $LOG_FILE"
        exit 1
    }
}

function Show-Help {
    Write-Output ""
    Write-Output "Reranker Service Daemon Manager"
    Write-Output ""
    Write-Output "Usage: .\daemon.ps1 [command]"
    Write-Output ""
    Write-Output "Commands:"
    Write-Output "  start     Start the service as a background process"
    Write-Output "  stop      Stop the running daemon"
    Write-Output "  restart   Restart the daemon"
    Write-Output "  status    Show daemon status"
    Write-Output "  logs      Follow the log file"
    Write-Output "  help      Show this help message"
    Write-Output ""
    Write-Output "Configuration:"
    Write-Output "  The script reads configuration from .env file."
    Write-Output "  Copy .env.example to .env and modify as needed."
    Write-Output ""
    Write-Output "Files:"
    Write-Output "  PID File: $PID_FILE"
    Write-Output "  Log File: $LOG_FILE"
    Write-Output "  Env File: $ENV_FILE"
    Write-Output ""
}

# Main entry point
switch ($Command) {
    "start" { Start-Daemon }
    "stop" { Stop-Daemon }
    "restart" { Restart-Daemon }
    "status" { Show-Status }
    "logs" { Show-Logs }
    "help" { Show-Help }
    default { Show-Help }
}
