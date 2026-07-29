param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate3_v1_sparse_active_reserve_robustness_seed_2"
)

& (Join-Path $PSScriptRoot "run_gate3_v1_robustness_common.ps1") `
    -TrainingSeed 2 `
    -IdleMachineAttested:$IdleMachineAttested `
    -OutputRoot $OutputRoot

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
