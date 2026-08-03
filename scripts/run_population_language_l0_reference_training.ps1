param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [int]$Microbatch = 8,
    [int]$EvaluationMicrobatch = 8
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& python (Join-Path $PSScriptRoot 'run_population_language_l0_reference_training.py') `
    --output-root $OutputRoot `
    --microbatch $Microbatch `
    --evaluation-microbatch $EvaluationMicrobatch
if ($LASTEXITCODE -ne 0) {
    throw "Population Language L0 reference training failed or classified invalid: $LASTEXITCODE"
}
