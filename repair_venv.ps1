# Reinstall wheels so they match THIS venv's Python (fixes cp311 vs cp312 NumPy mismatch).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error ".venv not found. Create it first: python -m venv .venv"
}
& $py -m pip install --upgrade pip
& $py -m pip install --no-cache-dir --force-reinstall -r requirements.txt
& $py -c "import numpy; import numpy._core._multiarray_umath; print('NumPy OK:', numpy.__version__)"
