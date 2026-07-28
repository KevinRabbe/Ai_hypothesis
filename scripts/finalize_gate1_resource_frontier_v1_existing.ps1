param(
    [string]$OutputRoot = "results\population_compute_scaling_v0\gate1_resource_frontier_v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedMeasurementHead = "18b201e22ca0a33feb4644c8ed8a09375e8e23ea"
$ExpectedCheckpointFileSha256 = "0b7c1f2a14fe9d2987819ed53fc0b55c04f3bb00bce356c1023778830a08ad26"
$ExpectedParameterFingerprint = "c227ade9006e47bec17a2a3d5aedf6ac95a6a94607b96b9f52ab759905536c12"
$ExpectedBenchmarkVersion = "relay-work-span-frontier-v1"
$ExpectedCorrectnessPolicy = "fp64-shadow-decoded-equivalence-v1"
$ExpectedConditionCount = 30

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    if (-not (Test-Path $OutputRoot)) {
        throw "Gate-1 v1 output root does not exist: $OutputRoot"
    }
    $OutputRoot = (Resolve-Path $OutputRoot).Path

    $checkpointPath = Join-Path $OutputRoot "frozen-confirmation\run\seed_1\model-v1.pt"
    $checkpointVerification = Join-Path $OutputRoot "checkpoint-verification.json"
    $resultPath = Join-Path $OutputRoot "relay_resource_frontier_v1.json"
    $auditPath = Join-Path $OutputRoot "relay_resource_frontier_v1.audit.json"
    $reportPath = Join-Path $OutputRoot "relay_resource_frontier_v1.report.md"
    $gitHeadPath = Join-Path $OutputRoot "git-head.txt"
    $gitStatusPath = Join-Path $OutputRoot "git-status.txt"
    $sourceManifest = Join-Path $OutputRoot "source-manifest.sha256"
    $nvidiaSmi = Join-Path $OutputRoot "nvidia-smi.txt"
    $repairRecordPath = Join-Path $OutputRoot "packaging-repair.json"
    $resultManifest = Join-Path $OutputRoot "result-manifest.sha256"

    foreach ($requiredPath in @(
        $checkpointPath,
        $checkpointVerification,
        $resultPath,
        $auditPath,
        $reportPath,
        $gitHeadPath,
        $sourceManifest
    )) {
        if (-not (Test-Path $requiredPath)) {
            throw "Required first-run evidence is missing: $requiredPath"
        }
    }

    if (Test-Path $resultManifest) {
        throw "result-manifest.sha256 already exists; refusing to overwrite an already-finalized result."
    }
    if (Test-Path $repairRecordPath) {
        throw "packaging-repair.json already exists; refusing to repeat the repair."
    }

    $measurementHead = (Get-Content -Raw $gitHeadPath).Trim()
    if ($measurementHead -ne $ExpectedMeasurementHead) {
        throw "Stored measurement Git head is not the canonical first Gate-1 v1 target run: $measurementHead"
    }

    $checkpointHash = (Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant()
    if ($checkpointHash -ne $ExpectedCheckpointFileSha256) {
        throw "Frozen checkpoint hash changed: $checkpointHash"
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    if ($result.benchmark_version -ne $ExpectedBenchmarkVersion) {
        throw "Result benchmark version is not the frozen Gate-1 v1 version."
    }
    if ($result.parameter_fingerprint -ne $ExpectedParameterFingerprint) {
        throw "Result parameter fingerprint is not the canonical checkpoint fingerprint."
    }
    if ($result.correctness_policy.name -ne $ExpectedCorrectnessPolicy) {
        throw "Result correctness policy is not the frozen Gate-1 v1 policy."
    }
    if (@($result.comparisons).Count -ne $ExpectedConditionCount) {
        throw "Result does not contain the complete 30-cell matrix."
    }

    $audit = Get-Content -Raw $auditPath | ConvertFrom-Json
    if ($audit.protocol_valid -ne $true) {
        throw "Independent Gate-1 v1 audit is not protocol-valid."
    }
    if ($audit.expected_condition_count -ne $ExpectedConditionCount -or $audit.observed_condition_count -ne $ExpectedConditionCount) {
        throw "Independent audit does not cover the complete 30-cell matrix."
    }
    if (@($audit.reasons).Count -ne 0) {
        throw "Independent audit contains protocol-failure reasons."
    }
    if ($audit.parameter_fingerprint -ne $ExpectedParameterFingerprint) {
        throw "Independent audit fingerprint differs from the canonical checkpoint."
    }

    $scientificSourcePaths = @(
        "ai_hypothesis\population_compute\relay_serial_control.py",
        "ai_hypothesis\population_compute\relay_resource_frontier.py",
        "ai_hypothesis\population_compute\relay_resource_frontier_v1.py",
        "ai_hypothesis\population_compute\relay_resource_audit_v1.py",
        "ai_hypothesis\population_compute\run_relay_resource_frontier_v1.py",
        "ai_hypothesis\population_compute\audit_relay_resource_frontier_v1.py",
        "experiments\population_compute_scaling_v0\resource_frontier_protocol_v1.md",
        "experiments\population_compute_scaling_v0\gate1_v0_cuda_equivalence_result.md"
    )
    & git diff --quiet $measurementHead HEAD -- $scientificSourcePaths
    if ($LASTEXITCODE -ne 0) {
        throw "Scientific Gate-1 v1 source changed after the measured head; packaging-only finalization is not allowed."
    }

    $statusReconstructed = $false
    if (Test-Path $gitStatusPath) {
        $storedStatus = Get-Content -Raw $gitStatusPath
        if (-not [string]::IsNullOrWhiteSpace($storedStatus)) {
            throw "Stored measurement git-status.txt is non-empty; this canonical repair only applies to the clean first target run."
        }
    }
    else {
        New-Item -ItemType File -Force -Path $gitStatusPath | Out-Null
        $statusReconstructed = $true
    }

    $packagingHead = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve packaging Git head."
    }

    $resultHash = (Get-FileHash -Algorithm SHA256 $resultPath).Hash.ToLowerInvariant()
    $auditHash = (Get-FileHash -Algorithm SHA256 $auditPath).Hash.ToLowerInvariant()
    $reportHash = (Get-FileHash -Algorithm SHA256 $reportPath).Hash.ToLowerInvariant()
    $sourceManifestHash = (Get-FileHash -Algorithm SHA256 $sourceManifest).Hash.ToLowerInvariant()

    $repairRecord = [ordered]@{
        repair_version = "gate1-v1-empty-status-packaging-repair-v0"
        measurement_git_head = $measurementHead
        packaging_git_head = $packagingHead
        benchmark_rerun = $false
        timing_rerun = $false
        audit_protocol_valid_before_packaging_repair = $true
        status_snapshot_reconstructed = $statusReconstructed
        reconstructed_git_status = "clean/empty"
        basis = "The original runner checks git status before creating output and the recorded canonical invocation omitted -AllowDirtyTree; an empty PowerShell pipeline failed to create git-status.txt only after the benchmark and independent audit had completed."
        result_sha256 = $resultHash
        audit_sha256 = $auditHash
        report_sha256 = $reportHash
        source_manifest_sha256 = $sourceManifestHash
        checkpoint_sha256 = $checkpointHash
    }
    $repairRecord | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $repairRecordPath

    $resultFiles = @(
        $checkpointPath,
        $checkpointVerification,
        $resultPath,
        $auditPath,
        $reportPath,
        $gitHeadPath,
        $gitStatusPath,
        $sourceManifest,
        $repairRecordPath
    )
    if (Test-Path $nvidiaSmi) {
        $resultFiles += $nvidiaSmi
    }
    $resultLines = foreach ($path in $resultFiles) {
        $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
        "$hash  $path"
    }
    $resultLines | Set-Content -Encoding UTF8 $resultManifest

    Write-Host ""
    Write-Host "Gate-1 v1 first target-GPU result finalized after a packaging-only repair."
    Write-Host "No benchmark or timing rerun occurred."
    Write-Host "Measurement Git HEAD: $measurementHead"
    Write-Host "Packaging Git HEAD:   $packagingHead"
    Write-Host "Result:   $resultPath"
    Write-Host "Audit:    $auditPath"
    Write-Host "Report:   $reportPath"
    Write-Host "Repair:   $repairRecordPath"
    Write-Host "Manifest: $resultManifest"
}
finally {
    Pop-Location
}
