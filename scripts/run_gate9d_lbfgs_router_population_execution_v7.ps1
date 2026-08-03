param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
& python (Join-Path $PSScriptRoot 'run_gate9d_lbfgs_router_population_execution_v7.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D v7 failed: $LASTEXITCODE" }
