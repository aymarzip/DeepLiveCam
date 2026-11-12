@echo off
title DeepLiveCam Studio
cd /d "%~dp0"

echo ========================================================
echo               DeepLiveCam Studio Launcher
echo ========================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo [INFO] Launching DeepLiveCam Studio with DirectML GPU Acceleration...
python run.py --execution-provider dml %*

if %errorlevel% neq 0 (
    echo.
    echo [INFO] DirectML launch exited with code %errorlevel%. Fallback to auto...
    python run.py %*
)
