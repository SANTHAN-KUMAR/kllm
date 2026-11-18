@echo off
setlocal
cd /d "%~dp0"
set "KLLM_ROOT=%~dp0"
if "%KLLM_ROOT:~-1%"=="\" set "KLLM_ROOT=%KLLM_ROOT:~0,-1%"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m gateway.claude_launcher %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        python -m gateway.claude_launcher %*
    ) else (
        echo [ERROR] Python environment not found. Please run install.bat first.
        exit /b 1
    )
)
