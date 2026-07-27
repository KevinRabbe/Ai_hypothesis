param(
    [string]$OutputRoot = "results\population_compute_scaling_v0\gate1_resource_frontier_v0",
    [string]$CheckpointPath = "",
    [switch]$AllowDirtyTree
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedArtifactRunId = "30239005530"
$ExpectedArtifactName = "relay-v1-frozen-confirmation"
$ExpectedCheckpointRelativePath = "run\seed_1\model-v1.pt"
$ExpectedCheckpointFileSha256 = "0b7c1f2a14fe9d2987819ed53fc0b55c04f3bb00bce356c1023778830a08ad26"
$ExpectedParameterFingerprint = "c227ade9006e47bec17a2a3d5aedf6ac95a6a94607b96b9f52ab759905536c12"
$ExpectedTrainingSeed = 1
$ExpectedParameterCount = 26669

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    $OutputRoot = (Resolve-Path $OutputRoot).Path

    $gitHead = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve git HEAD. Run this script from a Git checkout."
    }
    $gitStatus = @(& git status --porcelain)
    if (-not $AllowDirtyTree -and $gitStatus.Count -gt 0) {
        throw "Working tree is dirty. Commit/stash changes or use -AllowDirtyTree for an explicitly noncanonical run."
    }
    $gitHead | Set-Content -Encoding UTF8 (Join-Path $OutputRoot "git-head.txt")
    $gitStatus | Set-Content -Encoding UTF8 (Join-Path $OutputRoot "git-status.txt")

    if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
        $artifactDir = Join-Path $OutputRoot "frozen-confirmation"
        $candidate = Join-Path $artifactDir $ExpectedCheckpointRelativePath
        if (-not (Test-Path $candidate)) {
            if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
                throw "GitHub CLI 'gh' is required to download the frozen confirmation artifact. Alternatively pass -CheckpointPath."
            }
            New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
            Write-Host "Downloading frozen confirmation artifact $ExpectedArtifactName from run $ExpectedArtifactRunId..."
            & gh run download $ExpectedArtifactRunId `
                --repo KevinRabbe/Ai_hypothesis `
                --name $ExpectedArtifactName `
                --dir $artifactDir
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to download frozen confirmation artifact."
            }
        }
        $CheckpointPath = $candidate
    }

    if (-not (Test-Path $CheckpointPath)) {
        throw "Checkpoint not found: $CheckpointPath"
    }
    $CheckpointPath = (Resolve-Path $CheckpointPath).Path

    $checkpointHash = (Get-FileHash -Algorithm SHA256 $CheckpointPath).Hash.ToLowerInvariant()
    if ($checkpointHash -ne $ExpectedCheckpointFileSha256) {
        throw "Checkpoint file SHA-256 mismatch. Expected $ExpectedCheckpointFileSha256, got $checkpointHash."
    }

    $verifyCode = @'
import json
import sys
import torch
from ai_hypothesis.population_compute.relay_experiment_v1 import load_relay_checkpoint_v1

path = sys.argv[1]
expected_fingerprint = sys.argv[2]
expected_seed = int(sys.argv[3])
expected_parameters = int(sys.argv[4])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch; Gate 1 requires the target GPU.")
model, payload = load_relay_checkpoint_v1(path, device="cuda")
if payload.get("training_seed") != expected_seed:
    raise SystemExit(f"training_seed mismatch: {payload.get('training_seed')} != {expected_seed}")
fingerprint = model.parameter_fingerprint()
if fingerprint != expected_fingerprint:
    raise SystemExit(f"parameter fingerprint mismatch: {fingerprint} != {expected_fingerprint}")
if model.trainable_parameter_count() != expected_parameters:
    raise SystemExit(
        f"learned parameter count mismatch: {model.trainable_parameter_count()} != {expected_parameters}"
    )
print(json.dumps({
    "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    "cuda_runtime": torch.version.cuda,
    "torch_version": torch.__version__,
    "training_seed": payload.get("training_seed"),
    "parameter_fingerprint": fingerprint,
    "learned_parameter_count": model.trainable_parameter_count(),
}, indent=2, sort_keys=True))
'@

    $checkpointVerification = Join-Path $OutputRoot "checkpoint-verification.json"
    & python -c $verifyCode $CheckpointPath $ExpectedParameterFingerprint $ExpectedTrainingSeed $ExpectedParameterCount |
        Tee-Object -FilePath $checkpointVerification
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen checkpoint/CUDA verification failed."
    }

    $nvidiaSmi = Join-Path $OutputRoot "nvidia-smi.txt"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -q | Set-Content -Encoding UTF8 $nvidiaSmi
    }

    $resultPath = Join-Path $OutputRoot "relay_resource_frontier_v0.json"
    $benchmarkStdout = Join-Path $OutputRoot "relay_resource_frontier_v0.stdout.txt"
    Write-Host "Running frozen Gate-1 resource matrix on CUDA..."
    & python -m ai_hypothesis.population_compute.run_relay_resource_frontier `
        --checkpoint $CheckpointPath `
        --device cuda `
        --population-sizes 1 4 16 64 256 `
        --batch-sizes 1 64 `
        --difficulties relay-2 relay-4 relay-8 `
        --warmup-iterations 20 `
        --measured-iterations 100 `
        --world-seed 0 `
        --output $resultPath |
        Tee-Object -FilePath $benchmarkStdout
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-1 resource benchmark failed."
    }

    $auditPath = Join-Path $OutputRoot "relay_resource_frontier_v0.audit.json"
    $reportPath = Join-Path $OutputRoot "relay_resource_frontier_v0.report.md"
    $auditStdout = Join-Path $OutputRoot "relay_resource_frontier_v0.audit.stdout.txt"
    Write-Host "Auditing complete frozen matrix..."
    & python -m ai_hypothesis.population_compute.audit_relay_resource_frontier `
        --input $resultPath `
        --audit-output $auditPath `
        --report-output $reportPath |
        Tee-Object -FilePath $auditStdout
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-1 result did not satisfy the frozen audit contract."
    }

    $sourceManifest = Join-Path $OutputRoot "source-manifest.sha256"
    $sourcePaths = @(
        "ai_hypothesis\population_compute\relay_serial_control.py",
        "ai_hypothesis\population_compute\relay_resource_frontier.py",
        "ai_hypothesis\population_compute\relay_resource_audit.py",
        "ai_hypothesis\population_compute\run_relay_resource_frontier.py",
        "ai_hypothesis\population_compute\audit_relay_resource_frontier.py",
        "experiments\population_compute_scaling_v0\resource_frontier_protocol_v0.md"
    )
    $sourceLines = foreach ($path in $sourcePaths) {
        $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
        "$hash  $path"
    }
    $sourceLines | Set-Content -Encoding UTF8 $sourceManifest

    $resultManifest = Join-Path $OutputRoot "result-manifest.sha256"
    $resultFiles = @(
        $CheckpointPath,
        $checkpointVerification,
        $resultPath,
        $auditPath,
        $reportPath,
        (Join-Path $OutputRoot "git-head.txt"),
        (Join-Path $OutputRoot "git-status.txt"),
        $sourceManifest
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
    Write-Host "Gate-1 target-GPU run completed and passed the frozen protocol audit."
    Write-Host "Git HEAD: $gitHead"
    Write-Host "Result: $resultPath"
    Write-Host "Audit:  $auditPath"
    Write-Host "Report: $reportPath"
    Write-Host "Manifest: $resultManifest"
}
finally {
    Pop-Location
}
