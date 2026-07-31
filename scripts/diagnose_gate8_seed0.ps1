param(
    [string]$CheckpointPath = "",
    [string]$SourceResultPath = "",
    [string]$SourceManifestPath = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedCheckpointHash = "4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b"
$ExpectedResultHash = "5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2"
$ExpectedManifestHash = "bb814b9ebb5116f0a13ff2ce130c5ad8e32ed4bd80453ddc167143b6cbf0bb8d"
$ExpectedBranch = "agent/gate8-seed0-causal-diagnostic-execution-v0"
$ProtocolHead = "0fa9ec48c31b36c90d58da827139457fd812b98c"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $wrapperSmoke = $env:GATE8_SEED0_DIAGNOSTIC_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 seed0 diagnostic requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current Gate8 diagnostic branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne $ExpectedBranch) {
        throw "Gate8 seed0 diagnostic must run from $ExpectedBranch."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 diagnostic Git head."
    }

    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputRoot = "F:\gate8_seed0_causal_diagnostic_v0"
    }
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    $resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")
    if ($resolvedOutputRoot.StartsWith($resolvedRepoRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Gate8 diagnostic output must be outside the Git repository."
    }
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 diagnostic output already exists: $resolvedOutputRoot"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 SEED-0 CAUSAL DIAGNOSTIC"
    Write-Host "============================================================"
    Write-Host "Git head:             $head"
    Write-Host "Output:               $resolvedOutputRoot"
    Write-Host "Runtime probes:       4"
    Write-Host "Head-only steps:      256"
    Write-Host "Full-resume steps:    512"
    Write-Host "Seeds 1/2:            FORBIDDEN"
    Write-Host "Scientific test:      FORBIDDEN"
    Write-Host "Reference model:      FORBIDDEN"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 seed0 diagnostic wrapper smoke completed before source artifacts, Torch, CUDA, worlds, optimizer or checkpoints."
        return
    }

    foreach ($entry in @(
        @{ Name = "checkpoint"; Path = $CheckpointPath; Hash = $ExpectedCheckpointHash },
        @{ Name = "source result"; Path = $SourceResultPath; Hash = $ExpectedResultHash },
        @{ Name = "source manifest"; Path = $SourceManifestPath; Hash = $ExpectedManifestHash }
    )) {
        if ([string]::IsNullOrWhiteSpace($entry.Path)) {
            throw "Gate8 diagnostic $($entry.Name) path is required."
        }
        $resolved = [System.IO.Path]::GetFullPath($entry.Path)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "Gate8 diagnostic $($entry.Name) is missing: $resolved"
        }
        $observedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved).Hash.ToLowerInvariant()
        if ($observedHash -ne $entry.Hash) {
            throw "Gate8 diagnostic $($entry.Name) hash mismatch: $observedHash"
        }
        $entry.Path = $resolved
    }

    $resolvedCheckpoint = [System.IO.Path]::GetFullPath($CheckpointPath)
    $resolvedSourceResult = [System.IO.Path]::GetFullPath($SourceResultPath)
    $resolvedSourceManifest = [System.IO.Path]::GetFullPath($SourceManifestPath)
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null
    $diagnosticRoot = Join-Path $resolvedOutputRoot "diagnostic"
    $resultPath = Join-Path $diagnosticRoot "gate8-seed0-causal-diagnostic-result.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")
    $runConfig = [ordered]@{
        experiment = "gate8-seed0-causal-diagnostic-execution-v0"
        diagnostic_protocol_head = $ProtocolHead
        source_checkpoint_sha256 = $ExpectedCheckpointHash
        source_result_sha256 = $ExpectedResultHash
        source_manifest_sha256 = $ExpectedManifestHash
        seed = 0
        learned_parameter_count = 19649
        runtime_probes = @(
            "baseline",
            "forced_active",
            "message_low4_decode",
            "forced_active_message_low4_decode"
        )
        head_only_steps = 256
        head_only_world_range = @(262144, 327680)
        full_resume_steps = 512
        full_resume_world_range = @(327680, 458752)
        seeds_1_2_performed = $false
        scientific_test_worlds_generated = $false
        reference_model_loaded = $false
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    $env:PYTHONHASHSEED = "0"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $preflight = @'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the Gate8 seed0 diagnostic")
print(f"gpu={torch.cuda.get_device_name(0)}")
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 diagnostic CUDA preflight failed."
    }

    python scripts/diagnose_gate8_seed0.py `
        --checkpoint-path $resolvedCheckpoint `
        --source-result-path $resolvedSourceResult `
        --source-manifest-path $resolvedSourceManifest `
        --device cuda `
        --output-root $diagnosticRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 seed0 diagnostic failed. Preserve the complete output root for diagnosis."
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Gate8 diagnostic result is missing: $resultPath"
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    if ($result.diagnostic_status -ne "G8_SEED0_CAUSAL_DIAGNOSTIC_COMPLETE") {
        throw "Gate8 diagnostic did not complete with the frozen status."
    }
    if ($result.seed -ne 0 -or $result.learned_parameter_count -ne 19649) {
        throw "Gate8 diagnostic result identity drifted."
    }
    if ($result.baseline_reproduced -ne $true -or $result.diagnostic_performed -ne $true) {
        throw "Gate8 diagnostic result lacks required execution evidence."
    }
    if ($result.training_seeds_1_2_performed -ne $false) {
        throw "Gate8 diagnostic crossed the seed-1/2 boundary."
    }
    if ($result.scientific_test_worlds_generated -ne $false) {
        throw "Gate8 diagnostic crossed the scientific-test boundary."
    }
    if (
        $result.reference_tokenizer_loaded -ne $false -or
        $result.reference_model_weights_loaded -ne $false -or
        $result.reference_inference_performed -ne $false
    ) {
        throw "Gate8 diagnostic crossed the reference-model boundary."
    }
    if (@($result.runtime_probes).Count -ne 4) {
        throw "Gate8 diagnostic runtime probe set is incomplete."
    }
    if (@($result.head_only.checkpoints).Count -ne 4 -or @($result.full_resume.checkpoints).Count -ne 4) {
        throw "Gate8 diagnostic checkpoint set is incomplete."
    }
    if (@($result.findings.PSObject.Properties).Count -ne 5) {
        throw "Gate8 diagnostic finding vector is incomplete."
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

    $resultHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash
    $findingJson = $result.findings | ConvertTo-Json -Compress

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 seed-0 causal diagnostic complete"
    Write-Host "============================================================"
    Write-Host "Status:          $($result.diagnostic_status)"
    Write-Host "Findings:        $findingJson"
    Write-Host "Result SHA256:   $resultHash"
    Write-Host "Manifest SHA256: $manifestHash"
    Write-Host "Output root:     $resolvedOutputRoot"
}
finally {
    Pop-Location
}
