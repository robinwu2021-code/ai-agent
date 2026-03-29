@echo off
:: ============================================================
:: start.bat — 启动 AI Agent 服务
:: 运行环境：conda ai 环境（D:\ProgramData\coda_envs\ai）
:: ============================================================

setlocal

:: ── 项目根目录 ──────────────────────────────────────────────
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: ── 激活 conda ai 环境 ──────────────────────────────────────
call conda activate ai 2>nul || (
    echo [ERROR] 无法激活 conda ai 环境，请先运行 scripts\setup_d_drive.bat
    pause
    exit /b 1
)

:: ── 模型 & 缓存路径（全部 D 盘）────────────────────────────
set HF_HOME=D:\work\ai\models\huggingface
set SENTENCE_TRANSFORMERS_HOME=D:\work\ai\models\sentence_transformers
set TORCH_HOME=D:\work\ai\models\torch
set PIP_CACHE_DIR=D:\work\ai\cache\pip

:: ── 项目数据目录（确保存在）────────────────────────────────
if not exist "data" mkdir data

:: ── 显示启动信息 ─────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║            AI Agent  启动中...               ║
echo  ╚══════════════════════════════════════════════╝
echo  Python  : %CONDA_PREFIX%\python.exe
echo  数据目录 : %PROJECT_DIR%data
echo  模型缓存 : D:\work\ai\models
echo.

:: ── 启动服务 ────────────────────────────────────────────
python main.py %*

endlocal
