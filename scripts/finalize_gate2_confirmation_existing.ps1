param(
    [string]$OutputRoot = "results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/confirmation_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MeasurementHead = "c2a26a17a94746ca88f29950197131689405917b"

Push-Location $RepoRoot
try {
    $resolvedOutputRoot = Join-Path $RepoRoot $OutputRoot
    if (-not (Test-Path $resolvedOutputRoot -PathType Container)) {
        throw "Gate-2 confirmation output root does not exist: $resolvedOutputRoot"
    }

    $recordedHeadPath = Join-Path $resolvedOutputRoot "git-head.txt"
    if (-not (Test-Path $recordedHeadPath -PathType Leaf)) {
        throw "Gate-2 confirmation git-head.txt is missing."
    }
    $recordedHead = (Get-Content -Raw $recordedHeadPath).Trim()
    if ($recordedHead -ne $MeasurementHead) {
        throw "Gate-2 confirmation artifact head mismatch. Expected $MeasurementHead, recorded $recordedHead"
    }

    $gitStatusPath = Join-Path $resolvedOutputRoot "git-status.txt"
    if (-not (Test-Path $gitStatusPath -PathType Leaf)) {
        throw "Gate-2 confirmation git-status.txt is missing."
    }
    if ((Get-Item $gitStatusPath).Length -ne 0) {
        throw "Gate-2 confirmation recorded a dirty worktree; refusing finalization."
    }

    $runConfigPath = Join-Path $resolvedOutputRoot "run-config.json"
    if (-not (Test-Path $runConfigPath -PathType Leaf)) {
        throw "Gate-2 confirmation run-config.json is missing."
    }
    $runConfig = Get-Content -Raw $runConfigPath | ConvertFrom-Json
    if ([string]$runConfig.protocol -ne "gate2-persistent-state-confirmation-v0") {
        throw "Gate-2 confirmation protocol mismatch."
    }
    if (@($runConfig.training_seeds).Count -ne 3 -or (@($runConfig.training_seeds) -join ",") -ne "3,4,5") {
        throw "Gate-2 confirmation training seeds are not exactly 3,4,5."
    }

    $suitePath = Join-Path $resolvedOutputRoot "confirmation-suite.json"
    $suiteManifestPath = Join-Path $resolvedOutputRoot "suite-manifest.sha256"
    if (Test-Path $suitePath) {
        throw "confirmation-suite.json already exists; refusing to overwrite an existing finalized suite."
    }
    if (Test-Path $suiteManifestPath) {
        throw "suite-manifest.sha256 already exists; refusing to overwrite an existing finalized suite manifest."
    }

    $seedSummaries = @()
    foreach ($seed in @(3, 4, 5)) {
        $seedRoot = Join-Path $resolvedOutputRoot "seed_$seed"
        $resultPath = Join-Path $seedRoot "gate2-confirmation.json"
        $checkpointPath = Join-Path $seedRoot "gate2-confirmation-checkpoint.pt"
        $runtimePath = Join-Path $seedRoot "runtime.json"
        $manifestPath = Join-Path $seedRoot "result-manifest.sha256"

        foreach ($required in @($resultPath, $checkpointPath, $runtimePath, $manifestPath)) {
            if (-not (Test-Path $required -PathType Leaf)) {
                throw "Expected completed Gate-2 confirmation seed artifact missing: $required"
            }
        }

        $result = Get-Content -Raw $resultPath | ConvertFrom-Json
        if ([int]$result.training.training_seed -ne $seed) {
            throw "Seed $seed result reports the wrong training seed."
        }
        if ([string]$result.scientific_status -ne "CONFIRMATION_SEED_RESULT") {
            throw "Seed $seed is not a completed confirmation result."
        }
        if ([bool]$result.confirmation_opened -ne $true -or [string]$result.evaluation_split -ne "confirmation") {
            throw "Seed $seed did not use the frozen confirmation split."
        }

        $expectedManifest = @(
            "$((Get-FileHash -Algorithm SHA256 $resultPath).Hash.ToLowerInvariant())  gate2-confirmation.json",
            "$((Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant())  gate2-confirmation-checkpoint.pt",
            "$((Get-FileHash -Algorithm SHA256 $runtimePath).Hash.ToLowerInvariant())  runtime.json"
        )
        $recordedManifest = @(Get-Content $manifestPath)
        if (($recordedManifest -join "`n") -ne ($expectedManifest -join "`n")) {
            throw "Seed $seed result-manifest.sha256 does not match the completed seed artifacts."
        }

        $seedSummaries += [pscustomobject][ordered]@{
            training_seed = $seed
            seed_passed = [bool]$result.seed_passed
            width1_identity_passed = [bool]$result.width1_identity_passed
            result_sha256 = (Get-FileHash -Algorithm SHA256 $resultPath).Hash.ToLowerInvariant()
            checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant()
            parameter_fingerprint = [string]$result.training.parameter_fingerprint
            primary_comparisons = $result.primary_comparisons
        }
    }

    $failedSeeds = @($seedSummaries | Where-Object { -not $_.seed_passed })
    $capabilityPassed = ($seedSummaries.Count -eq 3) -and ($failedSeeds.Count -eq 0)

    $suite = [ordered]@{
        protocol = "gate2-persistent-state-confirmation-v0"
        confirmation_training_seeds = @(3, 4, 5)
        capability_confirmation_passed = $capabilityPassed
        gate2_overall_verdict = "NOT_ASSIGNED_UNTIL_RESOURCE_PROTOCOL_COMPLETE"
        seeds = $seedSummaries
    }
    $suite | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path $suitePath

    $afterPath = Join-Path $resolvedOutputRoot "nvidia-smi-after.txt"
    if (-not (Test-Path $afterPath -PathType Leaf)) {
        throw "nvidia-smi-after.txt is missing; original confirmation runner did not reach final capture."
    }

    $topArtifacts = @(
        "confirmation-suite.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi-before.txt",
        "nvidia-smi-after.txt"
    )
    $topManifest = foreach ($name in $topArtifacts) {
        $path = Join-Path $resolvedOutputRoot $name
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Expected top-level confirmation artifact missing: $name"
        }
        $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
        "$hash  $name"
    }
    $topManifest | Set-Content -Encoding ASCII -Path $suiteManifestPath

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-2 existing confirmation finalized WITHOUT rerunning seeds"
    Write-Host "============================================================"
    Write-Host "Seeds found: 3,4,5"
    Write-Host "Failed seeds: $($failedSeeds.Count)"
    Write-Host "Capability confirmation declared by frozen runner rule: $capabilityPassed"
    Write-Host "Suite: $suitePath"
    Write-Host "Manifest: $suiteManifestPath"
    Write-Host "Next: run the independent raw-world confirmation auditor before any resource timing."
}
finally {
    Pop-Location
}
