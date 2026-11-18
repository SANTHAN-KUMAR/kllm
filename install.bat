@echo off
rem kllm installer for Windows (double-click or run from a checkout). Same as install.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] kllm installation failed.
    pause
    exit /b 1
)
pause
