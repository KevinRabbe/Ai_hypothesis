param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# PyTorch deterministic CUDA requires this variable to exist before Python
# imports torch and initializes cuBLAS. Preserve an explicit caller choice.
if ([string]::IsNullOrWhiteSpace($env:CUBLAS_WORKSPACE_CONFIG)) {
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
}

& python (Join-Path $PSScriptRoot 'run_gate9d_lbfgs_router_population_execution_v7.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D v7 failed: $LASTEXITCODE" }
