param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_high_scale_routing_bandwidth_v0",
    [string]$TransitionCheckpoint0 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-0.pt",
    [string]$TransitionCheckpoint1 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-1.pt",
    [string]$TransitionCheckpoint2 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-2.pt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required Gate-7 high-scale file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 high-scale scientific screening requires explicit -IdleMachineAttested."
    }

    $wrapperSmoke = $env:GATE7_HIGH_SCALE_SCIENCE_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 high-scale scientific screening requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the current Gate-7 high-scale branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { ([string]$branchOutput).Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate7-high-scale-routing-bandwidth-execution-v0") {
        throw "Gate-7 high-scale screening must run from agent/gate7-high-scale-routing-bandwidth-execution-v0."
    }

    $headOutput = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve exact Gate-7 high-scale Git head."
    }
    $head = ([string]$headOutput).Trim()
    if ($head.Length -ne 40) {
        throw "Resolved Gate-7 high-scale Git head is invalid."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 high-scale output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate7-high-scale-routing-bandwidth.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "high-scale-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        experiment_version = "gate7-high-scale-routing-bandwidth-screening-v0"
        scientific_status = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_SCREENING_EVIDENCE"
        execution_admitted = $true
        high_scale_gate7_screening_opened = $true
        confirmation_opened = $false
        training_performed = $false
        checkpoint_selection_performed = $false
        engineering_prerequisite_head = "5305475ea1e295c84fadbce3533f13489b10d60d"
        engineering_summary_sha256 = "e40823e3e2787151f2a63607aa3d396f18e03428b715b8864af4f549631e2953"
        engineering_manifest_sha256 = "8393f9b4f11aa90aa333c3443669306675d1e9cc746e1f1dc3aa5acd1523afe4"
        populations = @(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
        k_ladder = @(16, 32, 64, 128, 256, 512)
        worlds_per_checkpoint_tier = 64
        evaluation_batch_size = 64
        bootstrap_samples = 2000
        hint_reliability = 0.70
        noninferiority_margin = 0.05
        stage_b_parent_slots = 128
        frontier_max_recurrent_rows = 1048576
        compiler_enabled = $false
        cuda_graphs_enabled = $false
        mixed_precision_enabled = $false
        transition_checkpoint0 = [System.IO.Path]::GetFullPath($TransitionCheckpoint0)
        transition_checkpoint1 = [System.IO.Path]::GetFullPath($TransitionCheckpoint1)
        transition_checkpoint2 = [System.IO.Path]::GetFullPath($TransitionCheckpoint2)
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 HIGH-SCALE ROUTING-BANDWIDTH SCREENING"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Checkpoints:       T0 / T1 / T2 exact SHA256 bound"
    Write-Host "Populations:       N1024 -> N131072"
    Write-Host "K ladder:          K16 -> K512 sequential first-pass"
    Write-Host "Worlds:            64 fresh paired worlds/checkpoint/tier"
    Write-Host "World batch:       64 fixed"
    Write-Host "Stage-B slots:     128 fixed"
    Write-Host "Compiler/graphs/MP:OFF / OFF / OFF"
    Write-Host "Training/selection:NONE / NONE"
    Write-Host "Confirmation:      CLOSED"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-7 high-scale wrapper smoke completed before checkpoints, CUDA or world generation."
        return
    }

    @(
        $TransitionCheckpoint0,
        $TransitionCheckpoint1,
        $TransitionCheckpoint2
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
        throw "Gate-7 high-scale CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate7_high_scale_routing_bandwidth `
        --output-root $scienceRoot `
        --transition-checkpoint0 $TransitionCheckpoint0 `
        --transition-checkpoint1 $TransitionCheckpoint1 `
        --transition-checkpoint2 $TransitionCheckpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 high-scale evaluator failed. Preserve the output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-7 high-scale artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate7_high_scale_routing_bandwidth `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-7 high-scale audit rejected the artifact. Preserve all output."
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
    Write-Host " Gate-7 high-scale screening complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Campaign outcome:     $($audit.campaign_outcome)"
    Write-Host "Populations completed:$($audit.populations_completed -join ', ')"
    Write-Host "Confirmation opened:  False"
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
