# Windows: right-click and choose "Run with PowerShell", or run: powershell -ExecutionPolicy Bypass -File start.ps1
#   start.ps1            first run: asks whether to set up in the browser or in this terminal
#   start.ps1 web        open the browser setup and import page
#   start.ps1 setup      run the terminal setup wizard
#   start.ps1 <url>      import a link from the terminal
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (Python package manager, one time)."
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host "Installing Python 3.12 and shortform-notes dependencies. The first run takes about a minute."
uv python install 3.12 --quiet
# cuda: cuBLAS/cuDNN wheels for GPU Whisper (unused, and harmless, without an NVIDIA GPU).
uv sync --python 3.12 --extra all --extra local --extra ocr --extra cuda --upgrade-package yt-dlp --quiet

if ($args.Count -gt 0) { uv run shortform-notes @args; exit $LASTEXITCODE }

$config = if ($env:SHORTFORM_NOTES_CONFIG) { $env:SHORTFORM_NOTES_CONFIG } else { "$env:USERPROFILE\.config\shortform-notes\config.env" }
if (Test-Path $config) { uv run shortform-notes web; exit $LASTEXITCODE }

Write-Host ""
Write-Host "How do you want to set up shortform-notes?"
Write-Host "  1) In the browser (recommended if you are not used to the terminal)"
Write-Host "  2) In this terminal"
$choice = Read-Host "Choose 1 or 2 [1]"
if ($choice -eq "2") { uv run shortform-notes setup } else { uv run shortform-notes web }
