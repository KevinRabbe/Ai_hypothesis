param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [int]$Steps = 256
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $env:CUBLAS_WORKSPACE_CONFIG) {
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
}
& python (Join-Path $PSScriptRoot 'run_population_language_l0_overfit_diagnostic.py') `
    --output-root $OutputRoot `
    --steps $Steps
if ($LASTEXITCODE -ne 0) {
    throw "Population Language L0 overfit diagnostic failed: $LASTEXITCODE"
}
