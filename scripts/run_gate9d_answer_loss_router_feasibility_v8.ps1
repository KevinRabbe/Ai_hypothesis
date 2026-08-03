param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
if (-not $env:CUBLAS_WORKSPACE_CONFIG) { $env:CUBLAS_WORKSPACE_CONFIG=':4096:8' }
& python (Join-Path $PSScriptRoot 'run_gate9d_answer_loss_router_feasibility_v8.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D v8 failed: $LASTEXITCODE" }
