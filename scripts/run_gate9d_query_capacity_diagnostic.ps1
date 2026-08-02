param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:GATE9D_QUERY_CAPACITY_WRAPPER_SMOKE -eq "1") {
    Write-Output "GATE9D_QUERY_CAPACITY_WRAPPER_SMOKE_OK"
    exit 0
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDirectory "run_gate9d_query_capacity_diagnostic.py"

& python $PythonScript --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "Gate9D query-capacity diagnostic failed with exit code $LASTEXITCODE"
}
