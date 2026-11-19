# kllm one-line installer (Windows PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SANTHAN-KUMAR/kllm/main/install.ps1 | iex"
# Installs uv if needed (it brings its own Python), then kllm as an isolated tool: `kllm` and `claude-kllm`.
# Run from a checkout (.\install.ps1 or install.bat) it installs that checkout instead. Never launches a Kaggle GPU.
# Already have Python? `pipx install kllm-gateway` or `uv tool install kllm-gateway` does the same.
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

if ($pyproject -and (Test-Path $pyproject) -and (Select-String -Path $pyproject -Pattern '^name = "kllm-gateway"' -Quiet)) {
    Write-Host "kllm: installing from this checkout: $here"
    uv tool install --force --editable "$here"
} else {
    $src = if ($env:KLLM_SOURCE) { $env:KLLM_SOURCE } else { "kllm-gateway" }
    Write-Host "kllm: installing $src ..."
    uv tool install --force $src
    if ($LASTEXITCODE -ne 0) { uv tool install --force "git+https://github.com/SANTHAN-KUMAR/kllm" }
}
if ($LASTEXITCODE -ne 0) { Write-Host "kllm: installation failed (see the messages above)."; exit 1 }
if (-not $env:KLLM_NO_MODIFY_PATH) { uv tool update-shell 2>$null | Out-Null }

# desktop and Start menu icon (click: start kllm and open the panel); KLLM_NO_SHORTCUT=1 skips it
$icon = $false
if (-not $env:KLLM_NO_SHORTCUT) {
    $bin = (uv tool dir --bin 2>$null)
    $exe = if ($bin) { Join-Path $bin "kllm.exe" } else { $null }
    if (-not ($exe -and (Test-Path $exe))) { $c = Get-Command kllm -ErrorAction SilentlyContinue; if ($c) { $exe = $c.Source } }
    if ($exe -and (Test-Path $exe)) { & $exe shortcut *> $null; if ($LASTEXITCODE -eq 0) { $icon = $true } }
}

Write-Host ""
Write-Host "kllm is installed."
if ($icon) { Write-Host "Click the kllm icon on your desktop or in the Start menu to start kllm and open the panel." }
Write-Host "Or, in a new terminal:"
Write-Host ""
Write-Host "  kllm             # starts kllm and opens the panel; paste your Kaggle token there on the first run"
Write-Host "  claude-kllm      # Claude Code on it"
Write-Host "  kllm update      # upgrade when a new version is out"
