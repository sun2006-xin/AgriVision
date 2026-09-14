param(
    [switch]$InstallDependencies,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SystemARoot = Join-Path $ProjectRoot "system_a"
$CoreRoot = Join-Path $SystemARoot "core"
$PythonPath = Join-Path $SystemARoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

$BindHost = if ($env:AGRIVISION_SYSTEM_A_BIND_HOST) {
    $env:AGRIVISION_SYSTEM_A_BIND_HOST
} else {
    "127.0.0.1"
}

if ($BindHost -notin @("127.0.0.1", "::1", "localhost") -and -not $env:AGRIVISION_API_TOKEN) {
    throw "AGRIVISION_API_TOKEN is required when System A binds outside loopback."
}

if ($InstallDependencies) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        python -m venv --system-site-packages (Join-Path $SystemARoot ".venv")
    }
    & $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-a.txt")
} elseif (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "System A virtual environment is missing. Run with -InstallDependencies first."
}

Push-Location $CoreRoot
try {
    & $PythonPath -m uvicorn app_fastapi:app --host $BindHost --port $Port --workers 1
} finally {
    Pop-Location
}
