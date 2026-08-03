param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
& python (Join-Path $PSScriptRoot 'run_gate9d_router_convergence_robustness_v6.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D router convergence v6 failed: $LASTEXITCODE" }
