param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:GATE9D_SPARSE_POPULATION_WRAPPER_SMOKE -eq "1") {
    Write-Output "GATE9D_SPARSE_POPULATION_WRAPPER_SMOKE_OK"
    exit 0
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDirectory "run_gate9d_sparse_affine_worker_population.py"

& python $PythonScript --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "Gate9D sparse affine worker population failed with exit code $LASTEXITCODE"
}
