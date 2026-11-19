@echo off
setlocal
cd /d "%~dp0"
set "KLLM_ROOT=%~dp0"
if "%KLLM_ROOT:~-1%"=="\" set "KLLM_ROOT=%KLLM_ROOT:~0,-1%"

rem Banner for a foreground `serve` is printed by gateway.banner (TTY-only).
if /I "%~1"=="serve" set "KLLM_BANNER=1"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m gateway.cli %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        python -m gateway.cli %*
    ) else (
        echo [ERROR] Python environment not found. Please run install.bat first.
        exit /b 1
    )
)
