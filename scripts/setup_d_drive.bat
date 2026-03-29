@echo off
:: ============================================================
:: scripts\setup_d_drive.bat — 一次性初始化：
::   将模型缓存、pip 缓存、conda 包全部迁移到 D 盘
:: 需要以管理员权限运行（setx /M 写系统环境变量）
:: ============================================================

echo ============================================================
echo  AI Agent — D 盘存储初始化
echo ============================================================
echo.

:: ── 1. 创建目录结构 ──────────────────────────────────────────
echo [1/5] 创建 D 盘目录...
set MODELS_ROOT=D:\work\ai\models
set CACHE_ROOT=D:\work\ai\cache

mkdir "%MODELS_ROOT%\huggingface\hub"     2>nul
mkdir "%MODELS_ROOT%\sentence_transformers" 2>nul
mkdir "%MODELS_ROOT%\torch\hub"           2>nul
mkdir "%MODELS_ROOT%\docling"             2>nul
mkdir "%CACHE_ROOT%\pip"                  2>nul
mkdir "%CACHE_ROOT%\conda_pkgs"           2>nul
mkdir "D:\work\ai\ai-agent\data"          2>nul
echo    目录已创建: D:\work\ai\models\  D:\work\ai\cache\

:: ── 2. 设置用户级环境变量（永久生效，当前用户）─────────────
echo.
echo [2/5] 设置用户级环境变量（永久）...
setx HF_HOME                   "D:\work\ai\models\huggingface"
setx SENTENCE_TRANSFORMERS_HOME "D:\work\ai\models\sentence_transformers"
setx TORCH_HOME                "D:\work\ai\models\torch"
setx PIP_CACHE_DIR             "D:\work\ai\cache\pip"
setx HF_HUB_DISABLE_TELEMETRY  "1"
echo    完成（新终端窗口自动生效）

:: ── 3. 配置 pip 缓存目录（pip.ini）─────────────────────────
echo.
echo [3/5] 配置 pip 缓存到 D 盘...
set PIP_INI=%APPDATA%\pip\pip.ini
if not exist "%APPDATA%\pip" mkdir "%APPDATA%\pip"
(
    echo [global]
    echo cache-dir = D:\work\ai\cache\pip
) > "%PIP_INI%"
echo    已写入: %PIP_INI%

:: ── 4. 配置 conda 包缓存到 D 盘 ─────────────────────────────
echo.
echo [4/5] 添加 conda 包缓存目录...
conda config --add pkgs_dirs "D:\work\ai\cache\conda_pkgs" 2>nul
conda config --show pkgs_dirs
echo    完成（D:\work\ai\cache\conda_pkgs 已加入首选）

:: ── 5. 迁移现有 HF 缓存（可选，询问用户）───────────────────
echo.
echo [5/5] 迁移现有模型缓存...

set SRC_HF=D:\python\appdata\huggingface\cache
set DST_HF=D:\work\ai\models\huggingface

if exist "%SRC_HF%\hub" (
    echo    发现旧缓存: %SRC_HF%
    set /p MIGRATE="    是否迁移到 %DST_HF% ？[Y/N] "
    if /i "!MIGRATE!"=="Y" (
        echo    迁移中，请稍候...
        robocopy "%SRC_HF%" "%DST_HF%" /E /MOVE /NP /NFL /NDL
        echo    迁移完成
    ) else (
        echo    跳过迁移（旧路径仍可用，但不是统一位置）
    )
) else (
    echo    未发现旧缓存，跳过
)

set SRC_C=C:\Users\%USERNAME%\.cache\huggingface
if exist "%SRC_C%\hub" (
    echo    发现 C 盘缓存: %SRC_C%
    set /p MIGRATE_C="    是否迁移到 %DST_HF% ？[Y/N] "
    if /i "!MIGRATE_C!"=="Y" (
        robocopy "%SRC_C%" "%DST_HF%" /E /MOVE /NP /NFL /NDL
        echo    迁移完成
    )
)

echo.
echo ============================================================
echo  初始化完成！
echo  ┌─────────────────────────────────────────────────────┐
echo  │  模型缓存  D:\work\ai\models\huggingface            │
echo  │            D:\work\ai\models\sentence_transformers  │
echo  │            D:\work\ai\models\torch                  │
echo  │  pip 缓存  D:\work\ai\cache\pip                     │
echo  │  conda包   D:\work\ai\cache\conda_pkgs              │
echo  │  项目数据  D:\work\ai\ai-agent\data\                │
echo  └─────────────────────────────────────────────────────┘
echo  启动服务请运行：start.bat
echo ============================================================
pause
