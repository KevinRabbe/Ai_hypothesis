param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate6_fixed_k_population_scaling_development_v0",
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
        throw "Required Gate-6 file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-6 development requires explicit -IdleMachineAttested."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-6 development requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "agent/gate6-fixed-k-population-scaling-v0") {
        throw "Gate-6 development must run from agent/gate6-fixed-k-population-scaling-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-6 Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-6 output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate6-fixed-k-population-scaling-development.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "development-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate6-fixed-k-population-scaling-v0"
        scientific_status = "DEVELOPMENT_ONLY_NOT_ASSIGNED"
        confirmation_opened = $false
        training_performed = $false
        depth = 10
        frontier_depth = 8
        hint_reliability = 0.70
        population_ladder = @(64, 128, 256)
        stage_a_parent_slots = 255
        stage_b_parent_slots = 128
        scheduled_parent_slots = 383
        active_child_lanes = 2
        recurrent_updates_per_child = 8
        learned_updates_per_world = 6128
        learned_parameter_count = 19649
        scheduler_conditions = @(
            "global_score",
            "bounded_score_k16",
            "bounded_hash_k16",
            "bounded_score_k8"
        )
        primary_bounded_k = 16
        descriptive_bounded_k = 8
        noninferiority_margin = 0.05
        worlds = 256
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

    $wrapperSmoke = $env:GATE6_WRAPPER_SMOKE -eq "1"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-6 v0 FIXED-K POPULATION SCALING - DEVELOPMENT ONLY"
    Write-Host "============================================================"
    Write-Host "Git head: $head"
    Write-Host "Output:   $resolvedOutputRoot"
    Write-Host "Training: NONE - frozen checkpoints only"
    Write-Host "Population ladder: N64 / N128 / N256"
    Write-Host "Work: 255 common frontier + 128 routing parent slots"
    Write-Host "Learned work: 6,128 recurrent updates/world"
    Write-Host "Primary: K16 learned routing + 5pp non-inferiority to global"
    Write-Host "K8 remains descriptive only"
    Write-Host "Confirmation remains CLOSED."
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-6 wrapper smoke completed before checkpoint/CUDA/scientific execution."
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
        throw "Gate-6 CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate6_fixed_k_population_scaling `
        --output-root $scienceRoot `
        --checkpoint0 $Checkpoint0 `
        --checkpoint1 $Checkpoint1 `
        --checkpoint2 $Checkpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-6 evaluator failed. Preserve the output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-6 development artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate6_fixed_k_population_scaling `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-6 audit rejected the artifact. Preserve all output."
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
    Write-Host " Gate-6 fixed-K population scaling development complete"
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
