param(
    [Parameter(Mandatory = $true)]
    [string]$Seed0Checkpoint,

    [Parameter(Mandatory = $true)]
    [string]$Seed1Checkpoint,

    [Parameter(Mandatory = $true)]
    [string]$Seed2Checkpoint,

    [string]$OutputRoot = "F:\gate8_v1_population_science_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $wrapperSmoke = $env:GATE8_V1_POPULATION_SCIENCE_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 v1 population science requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate8-v1-population-scientific-execution-v0") {
        throw "Gate8 v1 population science must run from agent/gate8-v1-population-scientific-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 v1 population-science Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 v1 population-science output already exists: $resolvedOutputRoot"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 THREE-SEED POPULATION SCIENTIFIC EVALUATION"
    Write-Host "============================================================"
    Write-Host "Git head:         $head"
    Write-Host "Output:           $resolvedOutputRoot"
    Write-Host "Test split:       test"
    Write-Host "Test seed:        0"
    Write-Host "World indices:    0..511 per condition"
    Write-Host "Conditions:       21"
    Write-Host "Checkpoint seeds: 0, 1, 2"
    Write-Host "Reference model: CLOSED in this phase"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 v1 population-science wrapper smoke completed before Torch, checkpoints, output creation, or test-world generation."
        return
    }

    $checkpointSpecs = @(
        @{ Seed = 0; Path = [System.IO.Path]::GetFullPath($Seed0Checkpoint); Hash = "3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9" },
        @{ Seed = 1; Path = [System.IO.Path]::GetFullPath($Seed1Checkpoint); Hash = "cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07" },
        @{ Seed = 2; Path = [System.IO.Path]::GetFullPath($Seed2Checkpoint); Hash = "e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4" }
    )
    foreach ($spec in $checkpointSpecs) {
        if (-not (Test-Path -LiteralPath $spec.Path -PathType Leaf)) {
            throw "Gate8 v1 seed-$($spec.Seed) checkpoint is missing: $($spec.Path)"
        }
        $observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $spec.Path).Hash.ToLowerInvariant()
        if ($observed -ne $spec.Hash) {
            throw "Gate8 v1 seed-$($spec.Seed) checkpoint SHA-256 mismatch."
        }
    }

    $preflight = @'
import inspect
import numpy
import torch
print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
print("weights_only=" + str("weights_only" in inspect.signature(torch.load).parameters))
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 population-science Python preflight failed."
    }

    python scripts/run_gate8_v1_population_science.py `
        --seed0-checkpoint $checkpointSpecs[0].Path `
        --seed1-checkpoint $checkpointSpecs[1].Path `
        --seed2-checkpoint $checkpointSpecs[2].Path `
        --output-root $resolvedOutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 population scientific evaluation failed. Preserve the output root."
    }

    $summaryPath = Join-Path $resolvedOutputRoot "population\gate8-v1-population-summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate8 v1 population summary is missing."
    }
    $summary = Get-Content -Raw $summaryPath | ConvertFrom-Json
    if ($summary.scientific_status -ne "G8_V1_POPULATION_SCIENTIFIC_EVALUATION_COMPLETE") {
        throw "Gate8 v1 population scientific status is invalid."
    }
    if ($summary.population_evaluation_complete -ne $true) {
        throw "Gate8 v1 population evaluation did not complete."
    }
    if ($summary.scientific_test_worlds_generated -ne $true) {
        throw "Gate8 v1 population evaluation did not record test-world exposure."
    }
    if ($summary.reference_model_loaded -ne $false -or $summary.reference_inference_performed -ne $false) {
        throw "Gate8 v1 population phase crossed the reference-model boundary."
    }
    if ($summary.training_performed -ne $false) {
        throw "Gate8 v1 population phase unexpectedly performed training."
    }

    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"
    $resolvedBase = (Resolve-Path $resolvedOutputRoot).Path.TrimEnd("\")
    @(
        Get-ChildItem -LiteralPath $resolvedBase -File -Recurse |
        Where-Object { $_.FullName -ne $manifestPath } |
        ForEach-Object {
            $relative = $_.FullName.Substring($resolvedBase.Length).TrimStart("\").Replace("\", "/")
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            "$hash  $relative"
        } |
        Sort-Object
    ) | Set-Content -Encoding ASCII $manifestPath

    $summaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $summaryPath).Hash
    $rowsPath = Join-Path $resolvedOutputRoot "population\gate8-v1-population-per-world.jsonl"
    $rowsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $rowsPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 population science complete"
    Write-Host "============================================================"
    Write-Host "Status:          $($summary.scientific_status)"
    Write-Host "Scaling outcome: $($summary.population_scaling_classification)"
    Write-Host "Worlds:          $($summary.test_matrix.unique_worlds)"
    Write-Host "Raw rows:        $($summary.raw_rows.rows)"
    Write-Host "Summary SHA256:  $summaryHash"
    Write-Host "Rows SHA256:     $rowsHash"
    Write-Host "Manifest SHA256: $manifestHash"
    Write-Host "Output root:     $resolvedOutputRoot"
}
finally {
    Pop-Location
}
