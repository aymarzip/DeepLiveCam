@echo off
title DeepLiveCam DirectML
echo Starting DeepLiveCam with DirectML (Intel Arc GPU acceleration)...
python run.py --execution-provider dml %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
