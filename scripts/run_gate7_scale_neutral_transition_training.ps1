param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_scale_neutral_transition_training_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 scale-neutral transition training requires explicit -IdleMachineAttested."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 transition training requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    $isSmoke = $env:GATE7_SCALE_NEUTRAL_TRAINING_WRAPPER_SMOKE -eq "1"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve current Git branch."
    }
    if ($isSmoke) {
        $allowedSmokeBranches = @(
            "agent/gate7-high-scale-frontier-prep-v0",
            "agent/gate7-scale-neutral-transition-training-v0"
        )
        if ($allowedSmokeBranches -notcontains $branch) {
            throw "Gate-7 transition-training wrapper smoke is on an unexpected branch: $branch"
        }
    }
    elseif ($branch -ne "agent/gate7-scale-neutral-transition-training-v0") {
        throw "Real Gate-7 transition training must run from agent/gate7-scale-neutral-transition-training-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-7 transition-training Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 SCALE-NEUTRAL SCORER TRANSITION TRAINING"
    Write-Host "============================================================"
    Write-Host "Git head:        $head"
    Write-Host "Output:          $resolvedOutputRoot"
    Write-Host "Training seeds:  0 / 1 / 2"
    Write-Host "Depth schedule:  6 -> 18 inclusive"
    Write-Host "Steps/checkpoint:1200"
    Write-Host "Batch size:      256"
    Write-Host "Parameters:      19,649"
    Write-Host "Bridge:          CLOSED"
    Write-Host "Gate-7 science:  CLOSED"
    Write-Host "Compiler:        OFF"
    Write-Host "CUDA graphs:     OFF"
    Write-Host "Mixed precision: OFF"
    Write-Host ""

    if ($isSmoke) {
        Write-Host "Gate-7 scale-neutral transition training wrapper smoke completed before CUDA/training."
        return
    }

    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 transition training output root already exists: $resolvedOutputRoot"
    }

    $preflight = @'
import json
import torch
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 transition training CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate7_scale_neutral_transition_training `
        --output-root $resolvedOutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 scale-neutral transition training failed. Preserve the output root for diagnosis."
    }

    $summaryPath = Join-Path $resolvedOutputRoot "training-summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate-7 transition training did not produce training-summary.json."
    }

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    Write-Host ""
    Write-Host "Transition training complete: $resolvedOutputRoot"
    Write-Host "The checkpoints remain UNBRIDGED; Gate-7 high-scale science is still closed."
}
finally {
    Pop-Location
}
