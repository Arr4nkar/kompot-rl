param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or 3.12 is required.' }
    & $python -m pip install -r requirements.txt -r requirements-lite.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '_output\lab\index.html'))) {
    & $python scripts/build_site.py
    if ($LASTEXITCODE -ne 0) { throw 'JupyterLite build failed.' }
}
Write-Host "Open http://localhost:$Port/lab/index.html?path=kompot.ipynb"
Write-Host 'Keep this window open. Press Ctrl+C to stop the server.'
& $python -m http.server $Port --bind 127.0.0.1 --directory _output
