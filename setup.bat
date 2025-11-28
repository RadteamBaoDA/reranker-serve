@echo off
REM =============================================================================
REM Reranker Service - Setup Script for Windows (CMD)
REM =============================================================================
REM This script sets up dependencies in existing virtual environment.
REM Usage: setup.bat [--dev] [--cuda]
REM =============================================================================

setlocal enabledelayedexpansion

set "VENV_DIR=venv"
set "INSTALL_DEV=false"
set "INSTALL_CUDA=false"

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--dev" (
    set "INSTALL_DEV=true"
    shift
    goto :parse_args
)
if "%~1"=="--cuda" (
    set "INSTALL_CUDA=true"
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: setup.bat [--dev] [--cuda]
echo.
echo Options:
echo   --dev    Install development dependencies
echo   --cuda   Install CUDA-enabled PyTorch
echo   --help   Show this help message
exit /b 0

:done_parsing

echo.
echo =======================================
echo   Reranker Service - Setup Script
echo =======================================
echo.

REM Check for Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed!
    echo Please install Python 3.10 or higher.
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VER=%%i"
echo Found Python %PYTHON_VER%

REM Check virtual environment exists
echo.
echo Checking virtual environment in %VENV_DIR%...

if not exist "%VENV_DIR%" (
    echo Error: Virtual environment not found at %VENV_DIR%
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv %VENV_DIR%
    echo.
    exit /b 1
)

echo Virtual environment found!

REM Activate virtual environment
echo.
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo Virtual environment activated!

REM Upgrade pip
echo.
echo Upgrading pip...
pip install --upgrade pip

REM Install PyTorch
echo.
if "%INSTALL_CUDA%"=="true" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch (CPU)...
    pip install torch torchvision torchaudio
)

REM Install main dependencies
echo.
echo Installing main dependencies...
pip install -r requirements.txt

REM Install development dependencies if requested
if "%INSTALL_DEV%"=="true" (
    echo.
    echo Installing development dependencies...
    pip install -r requirements-dev.txt
)

echo.
echo =======================================
echo   Setup Complete!
echo =======================================
echo.
echo To activate the virtual environment:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo To start the server:
echo   run.bat
echo.
echo Or manually:
echo   %VENV_DIR%\Scripts\activate.bat
echo   python -m src.main
echo.

endlocal
