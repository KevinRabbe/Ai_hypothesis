param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_high_scale_routing_bandwidth_continuation_v0",
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
        throw "Required Gate-7 continuation file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 routing-bandwidth continuation requires explicit -IdleMachineAttested."
    }

    $wrapperSmoke = $env:GATE7_CONTINUATION_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 continuation requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the current Gate-7 continuation branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate7-high-scale-routing-bandwidth-continuation-execution-v0") {
        throw "Gate-7 continuation must run from agent/gate7-high-scale-routing-bandwidth-continuation-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-7 continuation Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 continuation output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate7-high-scale-routing-bandwidth-continuation.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "continuation-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate7-high-scale-routing-bandwidth-continuation-v0"
        scientific_status = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONTINUATION_EVIDENCE"
        continuation_protocol_head = "4f05f8b1f9a33aed712edbf28691b927d2e220d3"
        confirmation_execution_head = "7afa6f204215bac7da4623e231ec34ef3b7fdc9f"
        confirmation_result_head = "ae8bd8544a03e48f4f397d2ca5ae933d9247e430"
        confirmation_result_sha256 = "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da"
        confirmation_audit_sha256 = "27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99"
        confirmation_manifest_sha256 = "e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263"
        confirmation_outcome = "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED"
        confirmed_n8192_passing_k = @(256, 512)
        confirmed_n8192_k_required = 256
        training_performed = $false
        checkpoint_selection_performed = $false
        continuation_opened = $true
        second_confirmation_opened = $false
        second_continuation_opened = $false
        populations = @(16384, 32768, 65536, 131072)
        k_ladder = @(16, 32, 64, 128, 256, 512)
        worlds_per_checkpoint_population = 512
        evaluation_batch_size = 64
        physical_batches = 8
        bootstrap_samples = 10000
        hint_reliability = 0.70
        noninferiority_margin = 0.05
        stage_b_parent_slots = 128
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
    Write-Host " Gate-7 POST-CONFIRMATION ROUTING CONTINUATION"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Checkpoints:       T0 / T1 / T2 exact SHA256 bound"
    Write-Host "Populations:       N16384 -> N131072 complete ladder"
    Write-Host "K matrix:          K16..K512 at every N"
    Write-Host "Worlds:            512 fresh paired worlds/checkpoint/N"
    Write-Host "Physical batches:  8 x B64"
    Write-Host "Bootstrap samples: 10,000"
    Write-Host "Scientific stops:  NONE"
    Write-Host "Training/selection:NONE / NONE"
    Write-Host "Second studies:    CLOSED"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-7 continuation wrapper smoke completed before checkpoint, CUDA or world generation."
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
        throw "Gate-7 continuation CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.run_gate7_high_scale_routing_bandwidth_continuation `
        --output-root $scienceRoot `
        --transition-checkpoint0 $TransitionCheckpoint0 `
        --transition-checkpoint1 $TransitionCheckpoint1 `
        --transition-checkpoint2 $TransitionCheckpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 continuation evaluator failed. Preserve the complete output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-7 continuation artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate7_high_scale_routing_bandwidth_continuation `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-7 continuation audit rejected the artifact. Preserve all output."
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
    $kRequiredJson = $audit.k_required_by_population | ConvertTo-Json -Compress
    $passingJson = $audit.passing_k_by_population | ConvertTo-Json -Compress

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 post-confirmation continuation complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Campaign outcome:     $($audit.campaign_outcome)"
    Write-Host "Completed populations:$([string]::Join(', ', @($audit.completed_populations)))"
    Write-Host "Resource frontier:    $($audit.resource_frontier_population)"
    Write-Host "K required by N:      $kRequiredJson"
    Write-Host "Passing K by N:       $passingJson"
    Write-Host "Second confirmation:  False"
    Write-Host "Second continuation:  False"
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
