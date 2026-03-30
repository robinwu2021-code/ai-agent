@echo off
:: ============================================================
:: start.bat -- AI Agent launcher
:: Python: conda ai (D:\ProgramData\coda_envs\ai)
:: ============================================================

setlocal

:: -- Project root dir ----------------------------------------
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: -- Conda ai Python path (explicit, no PATH dependency) ------
set PYTHON=D:\ProgramData\coda_envs\ai\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please ensure conda ai environment exists.
    pause
    exit /b 1
)

:: -- Model & cache paths (D drive) ----------------------------
set HF_HOME=D:\work\ai\models\huggingface
set SENTENCE_TRANSFORMERS_HOME=D:\work\ai\models\sentence_transformers
set TORCH_HOME=D:\work\ai\models\torch
set PIP_CACHE_DIR=D:\work\ai\cache\pip
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1

:: -- Ensure data dir exists -----------------------------------
if not exist "data" mkdir data

:: -- Print startup info ---------------------------------------
echo.
echo  =========================================================
echo   AI Agent  starting...
echo  =========================================================
echo   Python  : %PYTHON%
echo   Data    : %PROJECT_DIR%data
echo   Models  : D:\work\ai\models
echo  =========================================================
echo.

:: -- Start server ---------------------------------------------
"%PYTHON%" main.py %*

endlocal
