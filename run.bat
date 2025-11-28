@echo off
REM =============================================================================
REM Reranker Service - Run Script for Windows (CMD)
REM =============================================================================
REM This script runs the reranker service with optimal settings.
REM Usage: run.bat [--dev] [--port PORT] [--host HOST]
REM =============================================================================

setlocal enabledelayedexpansion

set "VENV_DIR=venv"
set "DEV_MODE=false"

REM Default settings
if not defined RERANKER_HOST set "RERANKER_HOST=0.0.0.0"
if not defined RERANKER_PORT set "RERANKER_PORT=8000"
if not defined RERANKER_WORKERS set "RERANKER_WORKERS=1"
if not defined RERANKER_LOG_LEVEL set "RERANKER_LOG_LEVEL=info"

set "HOST=%RERANKER_HOST%"
set "PORT=%RERANKER_PORT%"
set "WORKERS=%RERANKER_WORKERS%"
set "LOG_LEVEL=%RERANKER_LOG_LEVEL%"

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--dev" (
    set "DEV_MODE=true"
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--workers" (
    set "WORKERS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo.
echo Usage: run.bat [options]
echo.
echo Options:
echo   --dev              Enable development mode with auto-reload
echo   --workers N        Number of workers (default: 1)
echo   --port PORT        Server port (default: 8000)
echo   --host HOST        Server host (default: 0.0.0.0)
echo   --help, -h         Show this help message
echo.
exit /b 0

:done_parsing

echo.
echo =======================================
echo   Reranker Service
echo =======================================
echo.

REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Error: Dependencies not installed!
    echo Please run setup.bat first.
    exit /b 1
)

REM Detect device
echo Detecting device...
for /f "tokens=*" %%i in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2^>nul') do set "DEVICE=%%i"
if not defined DEVICE set "DEVICE=cpu"
echo Using device: %DEVICE%

if not defined RERANKER_DEVICE set "RERANKER_DEVICE=%DEVICE%"

REM Display configuration
echo.
echo Configuration:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Workers: %WORKERS%
echo   Device: %RERANKER_DEVICE%
if defined RERANKER_MODEL_NAME (
    echo   Model: %RERANKER_MODEL_NAME%
) else (
    echo   Model: BAAI/bge-reranker-v2-m3
)
echo.

REM Run the server
if "%DEV_MODE%"=="true" (
    echo Starting server in development mode...
    echo.
    uvicorn src.main:app --host %HOST% --port %PORT% --reload --log-level %LOG_LEVEL%
) else (
    echo Starting server in production mode...
    echo.
    if %WORKERS% GTR 1 (
        uvicorn src.main:app --host %HOST% --port %PORT% --workers %WORKERS% --log-level %LOG_LEVEL%
    ) else (
        uvicorn src.main:app --host %HOST% --port %PORT% --log-level %LOG_LEVEL%
    )
)

endlocal
