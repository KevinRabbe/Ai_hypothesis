param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_information_ceiling_precision_confirmation_v0",
    [string]$TransitionCheckpoint0 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-0.pt",
    [string]$TransitionCheckpoint1 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-1.pt",
    [string]$TransitionCheckpoint2 = "F:\gate7_scale_neutral_transition_training_v0\gate7-scale-neutral-transition-seed-2.pt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required Gate-7 precision-confirmation file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 precision-confirmation execution requires explicit -IdleMachineAttested."
    }

    $wrapperSmoke = $env:GATE7_PRECISION_CONFIRMATION_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 precision-confirmation execution requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current precision-confirmation branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate7-information-ceiling-precision-confirmation-execution-v0") {
        throw "Gate-7 precision confirmation must run from agent/gate7-information-ceiling-precision-confirmation-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact precision-confirmation Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 precision-confirmation output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate7-information-ceiling-precision-confirmation.json"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "precision-confirmation-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate7-information-ceiling-precision-confirmation-v0"
        scientific_status = "FRESH_GATE7_INFORMATION_CEILING_PRECISION_CONFIRMATION_EVIDENCE"
        protocol_head = "8d7865ab01b4b04b875ed2ca627b68a6c33c81f7"
        base_result_head = "4eb3e50a3ca7898ff81aebebddb7b049ff855df3"
        base_result_sha256 = "71a383ced44419f84022738448c460d79a3fb21746f436649e5f14399704f731"
        base_audit_sha256 = "86a7dbb774119cca9bcd697978081e0872b41e4e61a3f8b08538e0cc89c8397d"
        base_recovery_record_sha256 = "ccd4bbd353aba09b8a2d38d155bb9f883b862123bf196693889b515d5452324b"
        base_manifest_sha256 = "026f75a76888efe020c57da9d719140169eedd5e024555db20da9590cfea2b45"
        execution_opened = $true
        result_opened = $false
        training_performed = $false
        checkpoint_selection_performed = $false
        communication_intervention_performed = $false
        prior_worlds_reused = $false
        populations = @(16384, 32768, 65536, 131072)
        rankers = @("learned_score_rank", "bayes_hint_likelihood_rank", "public_hash_rank")
        attempt_ladder = @(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        primary_attempts = 128
        worlds_per_checkpoint_population = 2048
        evaluation_batch_size = 64
        physical_batches = 32
        bootstrap_samples = 20000
        bootstrap_unit = "world_index_clustered_within_population_across_T0_T1_T2"
        pooled_weighting = "equal_population_then_equal_checkpoint"
        hint_reliability = 0.70
        near_ceiling_margin = 0.02
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
    Write-Host " Gate-7 INFORMATION-CEILING PRECISION CONFIRMATION"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Checkpoints:       T0 / T1 / T2 exact SHA256 bound"
    Write-Host "Populations:       N16384 -> N131072 complete ladder"
    Write-Host "Rankers:           learned / Bayes / public hash"
    Write-Host "Attempt curve:     M1..M1024, primary M128"
    Write-Host "Worlds:            2,048 fresh worlds/checkpoint/N"
    Write-Host "Physical batches:  32 x B64"
    Write-Host "Bootstrap samples: 20,000 clustered"
    Write-Host "Training/selection:NONE / NONE"
    Write-Host "Communication:     NONE"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Precision-confirmation wrapper smoke completed before checkpoints, CUDA or world generation."
        return
    }

    @($TransitionCheckpoint0, $TransitionCheckpoint1, $TransitionCheckpoint2) |
        ForEach-Object { Assert-RequiredLeaf -Path $_ }

    try {
        & nvidia-smi | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" |
            Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }

    $preflight = @'
import json
import numpy
import torch
print(json.dumps({
    "torch_version": torch.__version__,
    "numpy_version": numpy.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
'@
    $preflight | python - | Tee-Object -FilePath (Join-Path $resolvedOutputRoot "cuda-preflight.json")
    if ($LASTEXITCODE -ne 0) { throw "Precision-confirmation CUDA preflight failed." }

    python -m ai_hypothesis.population_compute.run_gate7_information_ceiling_precision_confirmation `
        --output-root $scienceRoot `
        --transition-checkpoint0 $TransitionCheckpoint0 `
        --transition-checkpoint1 $TransitionCheckpoint1 `
        --transition-checkpoint2 $TransitionCheckpoint2
    if ($LASTEXITCODE -ne 0) {
        throw "Precision-confirmation evaluator failed. Preserve the complete output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing precision-confirmation artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate7_information_ceiling_precision_confirmation `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent precision-confirmation audit rejected the artifact. Preserve all output."
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
    Write-Host " Gate-7 precision confirmation complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Campaign outcome:     $($audit.campaign_outcome)"
    Write-Host "Training performed:   False"
    Write-Host "Checkpoint selection: False"
    Write-Host "Communication changed:False"
    Write-Host "Prior worlds reused:  False"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Audit SHA256:          $auditHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
