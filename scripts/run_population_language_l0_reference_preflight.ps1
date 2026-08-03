param(
    [Parameter(Mandatory=$true)][string]$OutputRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $env:CUBLAS_WORKSPACE_CONFIG) {
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
}
& python (Join-Path $PSScriptRoot 'run_population_language_l0_reference_preflight.py') `
    --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Population Language L0 reference preflight failed: $LASTEXITCODE"
}
