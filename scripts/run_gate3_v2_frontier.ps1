param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate3_v2_ambiguity_frontier_development_v0",
    [string]$Checkpoint0 = "F:\gate3_v1_sparse_active_reserve_development_seed_0\science\gate3-v1-development-checkpoint.pt",
    [string]$Checkpoint1 = "F:\gate3_v1_sparse_active_reserve_robustness_seed_1\science\gate3-v1-robustness-seed-1-checkpoint.pt",
    [string]$Checkpoint2 = "F:\gate3_v1_sparse_active_reserve_robustness_seed_2\science\gate3-v1-robustness-seed-2-checkpoint.pt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required Gate-3 v2 file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-3 v2 development requires explicit -IdleMachineAttested."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-3 v2 development requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "agent/gate3-v2-frontier-scaling-v0") {
        throw "Gate-3 v2 development must run from agent/gate3-v2-frontier-scaling-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-3 v2 Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-3 v2 output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate3-v2-frontier-development.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "development-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate3-v2-ambiguity-frontier-v0"
        scientific_status = "DEVELOPMENT_ONLY_NOT_ASSIGNED"
        confirmation_opened = $false
        training_performed = $false
        depth = 10
        search_rounds = 256
        active_child_lanes = 2
        recurrent_updates_per_child = 8
        learned_updates_per_world = 4096
        learned_parameter_count = 19649
        ambiguity_tiers = @("A60=0.60", "A55=0.55")
        stable_capacities = @(1, 16, 64, 256)
        frontier_controls = @("collapsed_L256", "reshuffled_L256")
        worlds_per_tier = 256
        evaluation_batch_size = 64
        bootstrap_samples = 2000
        compiler_enabled = $false
        cuda_graphs_enabled = $false
        mixed_precision_enabled = $false
        checkpoint0 = [System.IO.Path]::GetFullPath($Checkpoint0)
        checkpoint1 = [System.IO.Path]::GetFullPath($Checkpoint1)
        checkpoint2 = [System.IO.Path]::GetFullPath($Checkpoint2)
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    $wrapperSmoke = $env:GATE3_V2_WRAPPER_SMOKE -eq "1"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-3 v2 AMBIGUITY FRONTIER - DEVELOPMENT ONLY"
    Write-Host "============================================================"
    Write-Host "Git head: $head"
    Write-Host "Output:   $resolvedOutputRoot"
    Write-Host "Training: NONE - frozen checkpoints only"
    Write-Host "Confirmation remains CLOSED."
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-3 v2 wrapper smoke completed before checkpoint/CUDA/scientific execution."
        return
    }

    Assert-RequiredLeaf -Path $Checkpoint0
    Assert-RequiredLeaf -Path $Checkpoint1
    Assert-RequiredLeaf -Path $Checkpoint2

    try {
        & nvidia-smi | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" |
            Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
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
    $preflight | python - | Tee-Object -FilePath (Join-Path $resolvedOutputRoot "cuda-preflight.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-3 v2 CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate3_v2_frontier `
        --output-root $scienceRoot `
        --checkpoint0 $Checkpoint0 `
        --checkpoint1 $Checkpoint1 `
        --checkpoint2 $Checkpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-3 v2 frontier evaluator failed. Preserve the output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-3 v2 frontier artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate3_v2_frontier `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-3 v2 frontier audit rejected the artifact. Preserve all output."
    }
    Assert-RequiredLeaf -Path $auditPath

    try {
        & nvidia-smi | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" |
            Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }

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

    $audit = Get-Content -Raw $auditPath | ConvertFrom-Json
    $resultHash = (Get-FileHash -Algorithm SHA256 $resultPath).Hash
    $auditHash = (Get-FileHash -Algorithm SHA256 $auditPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-3 v2 frontier development complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Development outcome:  $($audit.directional_outcome)"
    Write-Host "Scientific status:    DEVELOPMENT ONLY - NO GATE VERDICT"
    Write-Host "Training performed:   False"
    Write-Host "Confirmation opened:  False"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Audit SHA256:          $auditHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
