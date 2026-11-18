# kllm one-line installer (Windows PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SANTHAN-KUMAR/kllm/main/install.ps1 | iex"
# Installs uv if needed (it brings its own Python), then kllm as an isolated tool: `kllm` and `claude-kllm`.
# Run from a checkout (.\install.ps1 or install.bat) it installs that checkout instead. Never launches a Kaggle GPU.
# Already have Python? `pipx install kllm` or `uv tool install kllm` does the same.
# "Continue", not "Stop": uv prints progress on stderr, which Windows PowerShell 5.1 would turn into a terminating
# error. Native commands are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "kllm: installing uv (https://docs.astral.sh/uv/) ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$here = $null
if ($PSScriptRoot) { $here = $PSScriptRoot }
$pyproject = if ($here) { Join-Path $here "pyproject.toml" } else { $null }

if ($pyproject -and (Test-Path $pyproject) -and (Select-String -Path $pyproject -Pattern '^name = "kllm"' -Quiet)) {
    Write-Host "kllm: installing from this checkout: $here"
    uv tool install --force --editable "$here"
} else {
    $src = if ($env:KLLM_SOURCE) { $env:KLLM_SOURCE } else { "kllm" }
    Write-Host "kllm: installing $src ..."
    uv tool install --force $src
    if ($LASTEXITCODE -ne 0) { uv tool install --force "git+https://github.com/SANTHAN-KUMAR/kllm" }
}
if ($LASTEXITCODE -ne 0) { Write-Host "kllm: installation failed (see the messages above)."; exit 1 }
if (-not $env:KLLM_NO_MODIFY_PATH) { uv tool update-shell 2>$null | Out-Null }

Write-Host ""
Write-Host "kllm is installed. Open a new terminal, then:"
Write-Host ""
Write-Host "  kllm up          # first run writes the config from ~\.kaggle\kaggle.json, then starts the default model"
Write-Host "  claude-kllm      # Claude Code on it"
Write-Host "  kllm doctor      # check everything end to end"
Write-Host ""
Write-Host "Only prerequisite: a Kaggle API token saved as ~\.kaggle\kaggle.json"
Write-Host "(kaggle.com -> Settings -> API -> Create New Token)."
