param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
& python (Join-Path $PSScriptRoot 'run_gate9d_router_factorization_sweep_v5.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D router factorization sweep failed: $LASTEXITCODE" }
