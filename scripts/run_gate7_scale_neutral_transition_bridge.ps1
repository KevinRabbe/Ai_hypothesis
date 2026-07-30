param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_scale_neutral_transition_bridge_v0",
    [string]$TransitionCheckpoint0 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-0.pt",
    [string]$TransitionCheckpoint1 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-1.pt",
    [string]$TransitionCheckpoint2 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-2.pt",
    [string]$OriginalCheckpoint0 = "F:\gate3_v1_sparse_active_reserve_development_seed_0\science\gate3-v1-development-checkpoint.pt",
    [string]$OriginalCheckpoint1 = "F:\gate3_v1_sparse_active_reserve_robustness_seed_1\science\gate3-v1-robustness-seed-1-checkpoint.pt",
    [string]$OriginalCheckpoint2 = "F:\gate3_v1_sparse_active_reserve_robustness_seed_2\science\gate3-v1-robustness-seed-2-checkpoint.pt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required Gate-7 transition bridge file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 transition bridge requires explicit -IdleMachineAttested."
    }

    $wrapperSmoke = $env:GATE7_TRANSITION_BRIDGE_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 transition bridge requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the current Gate-7 transition bridge branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate7-scale-neutral-transition-bridge-prep-v0") {
        throw "Gate-7 transition bridge must run from agent/gate7-scale-neutral-transition-bridge-prep-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-7 transition bridge Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 transition bridge output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate7-scale-neutral-transition-bridge.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "bridge-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate7-scale-neutral-transition-bridge-v0"
        scientific_status = "FRESH_LOW_SCALE_TRANSITION_BRIDGE_EVIDENCE"
        training_performed = $false
        checkpoint_selection_performed = $false
        high_scale_gate7_opened = $false
        depth = 10
        hint_reliability = 0.70
        populations = @(128, 256)
        scheduler_conditions = @("global_score", "bounded_score_k16", "bounded_hash_k16")
        worlds = 256
        evaluation_batch_size = 64
        bootstrap_samples = 2000
        noninferiority_margin = 0.05
        stage_a_parent_slots = 255
        stage_b_parent_slots = 128
        learned_updates_per_world = 6128
        learned_parameter_count = 19649
        compiler_enabled = $false
        cuda_graphs_enabled = $false
        mixed_precision_enabled = $false
        transition_checkpoint0 = [System.IO.Path]::GetFullPath($TransitionCheckpoint0)
        transition_checkpoint1 = [System.IO.Path]::GetFullPath($TransitionCheckpoint1)
        transition_checkpoint2 = [System.IO.Path]::GetFullPath($TransitionCheckpoint2)
        original_checkpoint0 = [System.IO.Path]::GetFullPath($OriginalCheckpoint0)
        original_checkpoint1 = [System.IO.Path]::GetFullPath($OriginalCheckpoint1)
        original_checkpoint2 = [System.IO.Path]::GetFullPath($OriginalCheckpoint2)
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 SCALE-NEUTRAL TRANSITION BRIDGE"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Transition seeds:  T0 / T1 / T2 - exact SHA256 bound"
    Write-Host "Original controls: C0 / C1 / C2 - exact SHA256 bound"
    Write-Host "Populations:       N128 / N256"
    Write-Host "Modes:             K16 learned / K16 hash / global"
    Write-Host "Worlds:            256 fresh paired worlds"
    Write-Host "Bridge classifier: 12 frozen primary criteria"
    Write-Host "High-scale Gate-7: CLOSED"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-7 transition bridge wrapper smoke completed before checkpoint/CUDA/scientific execution."
        return
    }

    @(
        $TransitionCheckpoint0,
        $TransitionCheckpoint1,
        $TransitionCheckpoint2,
        $OriginalCheckpoint0,
        $OriginalCheckpoint1,
        $OriginalCheckpoint2
    ) | ForEach-Object { Assert-RequiredLeaf -Path $_ }

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
        throw "Gate-7 transition bridge CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate7_scale_neutral_transition_bridge `
        --output-root $scienceRoot `
        --transition-checkpoint0 $TransitionCheckpoint0 `
        --transition-checkpoint1 $TransitionCheckpoint1 `
        --transition-checkpoint2 $TransitionCheckpoint2 `
        --original-checkpoint0 $OriginalCheckpoint0 `
        --original-checkpoint1 $OriginalCheckpoint1 `
        --original-checkpoint2 $OriginalCheckpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 transition bridge evaluator failed. Preserve the output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-7 transition bridge artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate7_scale_neutral_transition_bridge `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-7 transition bridge audit rejected the artifact. Preserve all output."
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
    Write-Host " Gate-7 scale-neutral transition bridge complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Transition outcome:   $($audit.transition_outcome)"
    Write-Host "High-scale Gate-7:    CLOSED"
    Write-Host "Training performed:   False"
    Write-Host "Checkpoint selection: False"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Audit SHA256:          $auditHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
