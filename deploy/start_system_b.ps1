param(
    [switch]$InstallDependencies
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($InstallDependencies) {
    python -m pip install -r (Join-Path $ProjectRoot "requirements-b.txt")
}

$BindHost = if ($env:AGRIVISION_BIND_HOST) { $env:AGRIVISION_BIND_HOST } else { "127.0.0.1" }
if ($BindHost -notin @("127.0.0.1", "::1", "localhost") -and -not $env:AGRIVISION_API_TOKEN) {
    throw "AGRIVISION_API_TOKEN is required when AGRIVISION_BIND_HOST is not loopback."
}

python (Join-Path $ProjectRoot "system_b\core\app.py")
