@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ========================================================
echo  kllm Setup for Windows
echo ========================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10 or higher from https://python.org
    echo and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/3] Installing dependencies and kllm package...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/4] Adding kllm to user PATH for global access...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = (Get-Item '%~dp0').FullName.TrimEnd('\'); $scripts = Join-Path $root '.venv\Scripts'; $p = [Environment]::GetEnvironmentVariable('Path', 'User'); if (-not $p) { $p = '' }; $toAdd = @($root, $scripts) | Where-Object { ($p -split ';') -notcontains $_ }; if ($toAdd) { $new = (($p.TrimEnd(';') + ';' + ($toAdd -join ';')) -replace '^;', ''); [Environment]::SetEnvironmentVariable('Path', $new, 'User'); Write-Host '  [OK] Added to User PATH:' ($toAdd -join ', ') } else { Write-Host '  [OK] Already in User PATH.' }"

echo [4/4] Verifying installation (creates config.yaml from your Kaggle token)...
python -m gateway.cli account list

echo.
echo ========================================================
echo  Installation successful!
echo ========================================================
echo.
echo Quick commands (usable from ANY folder in any new terminal):
echo   claude-kllm               Launch Claude Code with kllm backend
echo   kllm serve                Start the LLM gateway service
echo   kllm doctor               Run health check
echo   kllm up                   Launch the default model
echo   kllm account list         View configured Kaggle accounts
echo.
echo Note: Close and reopen your terminal once for the PATH changes to take effect!
echo.
echo Web UI: http://127.0.0.1:8080/admin
echo OpenAI API: http://127.0.0.1:8080/v1
echo.
pause
