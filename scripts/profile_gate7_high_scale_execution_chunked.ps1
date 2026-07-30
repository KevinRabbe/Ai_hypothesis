param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_high_scale_execution_engineering_profile_chunked_v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 chunked high-scale engineering profile requires explicit -IdleMachineAttested."
    }

    $wrapperSmoke = $env:GATE7_HIGH_SCALE_CHUNKED_PROFILE_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 chunked engineering profile requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve current Git branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { ([string]$branchOutput).Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate7-high-scale-engineering-profile-chunked-v1") {
        throw "Gate-7 chunked engineering profile must run from agent/gate7-high-scale-engineering-profile-chunked-v1."
    }

    $headOutput = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve exact Gate-7 chunked-profile Git head."
    }
    $head = ([string]$headOutput).Trim()
    if ($head.Length -ne 40) {
        throw "Resolved Gate-7 chunked-profile Git head is invalid."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 chunked profile output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $profileRoot = Join-Path $resolvedOutputRoot "profile"
    $summaryPath = Join-Path $profileRoot "summary.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        profile_version = "gate7-high-scale-execution-engineering-profile-chunked-v1"
        engineering_only = $true
        scientific_evidence = $false
        checkpoint_loading_performed = $false
        hidden_path_constructed = $false
        gate7_high_scale_science_opened = $false
        deterministic_random_model_seed = 71007
        learned_parameter_count = 19649
        physical_world_batch_size = 64
        world_batch_preserved = $true
        frontier_row_chunking_enabled = $true
        frontier_max_recurrent_rows = 1048576
        populations = @(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
        stage_b_parent_slots = 128
        profiled_conditions = @(
            "global_score",
            "global_hash",
            "bounded_score_k16",
            "bounded_hash_k16",
            "bounded_score_k512",
            "bounded_hash_k512"
        )
        compiler_enabled = $false
        cuda_graphs_enabled = $false
        mixed_precision_enabled = $false
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 CHUNKED HIGH-SCALE EXECUTION PROFILE - ENGINEERING"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Model:             deterministic random 19,649 parameters"
    Write-Host "Checkpoint loading:NONE"
    Write-Host "Hidden worlds:     NONE"
    Write-Host "Population:        1024 -> 131072"
    Write-Host "World batch:       64 preserved"
    Write-Host "Recurrent rows:    max 1,048,576 per chunk"
    Write-Host "Stage-B slots:     128 fixed"
    Write-Host "Conditions:        global/hash + matched K16/K512"
    Write-Host "Compiler:          OFF"
    Write-Host "CUDA graphs:       OFF"
    Write-Host "Mixed precision:   OFF"
    Write-Host "Gate-7 science:    CLOSED"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-7 chunked profile wrapper smoke completed before CUDA or profiling."
        return
    }

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
        throw "Gate-7 chunked profile CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.profile_gate7_high_scale_execution_chunked `
        --output-root $profileRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 chunked engineering profiler failed. Preserve the output root for diagnosis."
    }
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate-7 chunked engineering profile did not produce summary.json."
    }

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

    $summary = Get-Content -Raw $summaryPath | ConvertFrom-Json
    $summaryHash = (Get-FileHash -Algorithm SHA256 $summaryPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 chunked engineering profile complete"
    Write-Host "============================================================"
    Write-Host "Profile status:     $($summary.status)"
    Write-Host "World batch:        $($summary.world_batch_preserved)"
    Write-Host "Max recurrent rows: $($summary.frontier_max_recurrent_rows)"
    Write-Host "Scientific evidence:False"
    Write-Host "Checkpoint loading: False"
    Write-Host "Gate-7 science:     CLOSED"
    Write-Host "Summary SHA256:     $summaryHash"
    Write-Host "Manifest SHA256:    $manifestHash"
    Write-Host "Output root:        $resolvedOutputRoot"
}
finally {
    Pop-Location
}
