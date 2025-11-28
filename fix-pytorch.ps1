# =============================================================================
# Reranker Service - Fix PyTorch DLL Issue on Windows
# =============================================================================
# This script helps fix the common PyTorch DLL loading issue on Windows.
# Run this if you see: "OSError: [WinError 1114] A dynamic link library (DLL) 
# initialization routine failed"
# =============================================================================

param(
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

if ($Help) {
    Write-Output ""
    Write-Output "Fix PyTorch DLL Issue on Windows"
    Write-Output ""
    Write-Output "This script helps resolve the common 'WinError 1114' DLL loading error."
    Write-Output ""
    Write-Output "The error typically occurs because:"
    Write-Output "  1. Missing Microsoft Visual C++ Redistributable"
    Write-Output "  2. Corrupted PyTorch installation"
    Write-Output ""
    exit 0
}

Write-Output ""
Write-ColorOutput Cyan "======================================="
Write-ColorOutput Cyan "  Fix PyTorch DLL Issue"
Write-ColorOutput Cyan "======================================="
Write-Output ""

# Step 1: Check if Visual C++ Redistributable is installed
Write-ColorOutput Yellow "Step 1: Checking Visual C++ Redistributable..."

$vcRedistKeys = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)

$vcInstalled = $false
foreach ($key in $vcRedistKeys) {
    if (Test-Path $key) {
        $vcInstalled = $true
        break
    }
}

if (-not $vcInstalled) {
    Write-ColorOutput Red "Visual C++ Redistributable NOT found!"
    Write-Output ""
    Write-Output "Please download and install Microsoft Visual C++ Redistributable:"
    Write-Output "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Write-Output ""
    Write-Output "After installing, restart your terminal and try again."
    Write-Output ""
    
    $response = Read-Host "Would you like to open the download page now? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Start-Process "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    }
    exit 1
} else {
    Write-ColorOutput Green "Visual C++ Redistributable is installed."
}

# Step 2: Reinstall PyTorch with explicit CPU build
Write-Output ""
Write-ColorOutput Yellow "Step 2: Reinstalling PyTorch..."

$VENV_DIR = "venv"

if (-not (Test-Path $VENV_DIR)) {
    Write-ColorOutput Red "Virtual environment not found!"
    Write-Output "Please create venv first: python -m venv venv"
    exit 1
}

# Activate venv
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
& $activateScript

Write-Output "Uninstalling existing PyTorch..."
pip uninstall torch torchvision torchaudio -y 2>$null

Write-Output ""
Write-Output "Installing PyTorch CPU build for Windows..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 3: Test PyTorch
Write-Output ""
Write-ColorOutput Yellow "Step 3: Testing PyTorch..."

try {
    $result = python -c "import torch; print('PyTorch', torch.__version__, 'loaded successfully!')"
    Write-ColorOutput Green $result
    Write-Output ""
    Write-ColorOutput Green "======================================="
    Write-ColorOutput Green "  PyTorch is working!"
    Write-ColorOutput Green "======================================="
    Write-Output ""
    Write-Output "You can now run the server with:"
    Write-Output "  .\run.ps1"
} catch {
    Write-ColorOutput Red "PyTorch still not working!"
    Write-Output ""
    Write-Output "Additional steps to try:"
    Write-Output "  1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Write-Output "  2. Restart your computer after installation"
    Write-Output "  3. Try using Python 3.10 or 3.11 instead of 3.12"
    Write-Output ""
}
