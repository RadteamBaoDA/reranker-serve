@echo off
REM =============================================================================
REM Reranker Service - Download Model Script for Windows (CMD)
REM =============================================================================
REM This script downloads a model for offline use.
REM Usage: download_model.bat [model_name] [output_dir]
REM =============================================================================

setlocal enabledelayedexpansion

set "VENV_DIR=venv"
set "MODEL_NAME=%~1"
set "OUTPUT_DIR=%~2"

if "%MODEL_NAME%"=="" set "MODEL_NAME=BAAI/bge-reranker-v2-m3"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=.\models"

if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
goto :main

:show_help
echo.
echo Reranker Service - Model Download Script
echo.
echo Usage: download_model.bat [model_name] [output_dir]
echo.
echo Parameters:
echo   model_name   Model name from HuggingFace (default: BAAI/bge-reranker-v2-m3)
echo   output_dir   Output directory for models (default: .\models)
echo.
echo Examples:
echo   download_model.bat
echo   download_model.bat BAAI/bge-reranker-large
echo   download_model.bat BAAI/bge-reranker-v2-m3 C:\models
echo.
exit /b 0

:main

echo.
echo =======================================
echo   Reranker Service - Model Download
echo =======================================
echo.
echo Model: %MODEL_NAME%
echo Output Directory: %OUTPUT_DIR%
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

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Download the model
echo Downloading model...
echo.

python -c "import os; from huggingface_hub import snapshot_download; model_name='%MODEL_NAME%'; output_dir=r'%OUTPUT_DIR%'; model_dir=os.path.join(output_dir, model_name.replace('/', '_')); os.makedirs(model_dir, exist_ok=True); print(f'Downloading {model_name} to {model_dir}...'); snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False); print(f'Model downloaded successfully to: {model_dir}')"

echo.
echo =======================================
echo   Download Complete!
echo =======================================
echo.
echo To use this model offline, set:
echo   set RERANKER_MODEL_PATH=%OUTPUT_DIR%\%MODEL_NAME:/=_%
echo   set RERANKER_USE_OFFLINE_MODE=true
echo.

endlocal
