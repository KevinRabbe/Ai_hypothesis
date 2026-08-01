param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1, 2)]
    [int]$SeedIndex,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:GATE9D_STAGE1_WRAPPER_SMOKE -eq "1") {
    Write-Output "GATE9D_STAGE1_WRAPPER_SMOKE_OK"
    exit 0
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDirectory "run_gate9_contextual_failure_decomposition_stage1.py"

& python $PythonScript `
    --seed-index $SeedIndex `
    --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "Gate9D stage-1 execution failed with exit code $LASTEXITCODE"
}
