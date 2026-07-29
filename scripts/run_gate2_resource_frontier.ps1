param(
    [switch]$IdleMachineAttested,
    [string]$ConfirmationRoot = "results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/confirmation_v0",
    [string]$OutputRoot = "results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/resource_frontier_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $IdleMachineAttested) {
    throw "Gate-2 resource timing requires -IdleMachineAttested after closing Factorio and other GPU-heavy applications."
}
if (Get-Process -Name "factorio" -ErrorAction SilentlyContinue) {
    throw "Factorio is still running. Close it before admitted Gate-2 resource timing."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfirmationMeasurementHead = "c2a26a17a94746ca88f29950197131689405917b"
$ScientificSourcePaths = @(
    "ai_hypothesis/population_compute/gate2_persistent_state_capacity.py",
    "ai_hypothesis/population_compute/gate2_persistent_model.py",
    "ai_hypothesis/population_compute/gate2_development.py",
    "ai_hypothesis/population_compute/gate2_confirmation.py"
)

Push-Location $RepoRoot
try {
    $gitStatus = @(& git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect Git working tree."
    }
    if ($gitStatus.Count -ne 0) {
        $gitStatus | ForEach-Object { Write-Host $_ }
        throw "Gate-2 resource timing requires a clean working tree."
    }

    $head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $head) {
        throw "Could not resolve Git HEAD."
    }

    & git cat-file -e "$ConfirmationMeasurementHead^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen Gate-2 confirmation measurement head is unavailable locally: $ConfirmationMeasurementHead"
    }
    & git diff --quiet $ConfirmationMeasurementHead HEAD -- $ScientificSourcePaths
    if ($LASTEXITCODE -eq 1) {
        throw "Scientific Gate-2 model/world/training/confirmation source drifted after the frozen confirmation head."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not verify Gate-2 scientific-source identity."
    }

    $resolvedConfirmationRoot = Join-Path $RepoRoot $ConfirmationRoot
    if (-not (Test-Path $resolvedConfirmationRoot -PathType Container)) {
        throw "Gate-2 confirmation root does not exist: $resolvedConfirmationRoot"
    }
    $checkpointPath = Join-Path $resolvedConfirmationRoot "seed_3/gate2-confirmation-checkpoint.pt"
    if (-not (Test-Path $checkpointPath -PathType Leaf)) {
        throw "Frozen seed-3 confirmation checkpoint is missing: $checkpointPath"
    }

    Write-Host "Auditing completed Gate-2 capability confirmation before resource timing..."
    $auditText = (& python -m ai_hypothesis.population_compute.audit_gate2_confirmation $resolvedConfirmationRoot | Out-String)
    if ($LASTEXITCODE -ne 0) {
        Write-Host $auditText
        throw "Gate-2 confirmation artifact audit failed."
    }
    $audit = $auditText | ConvertFrom-Json
    if (-not [bool]$audit.artifact_valid) {
        throw "Gate-2 confirmation artifacts are structurally invalid."
    }
    if (-not [bool]$audit.capability_confirmation_passed) {
        throw "Frozen Gate-2 capability confirmation did not pass; v0 resource timing is not admitted as the second half of a positive Gate-2 result."
    }

    $resolvedOutputRoot = Join-Path $RepoRoot $OutputRoot
    if (Test-Path $resolvedOutputRoot) {
        throw "Gate-2 resource output root already exists: $resolvedOutputRoot"
    }

    $cudaProbe = Join-Path ([System.IO.Path]::GetTempPath()) ('gate2-resource-cuda-' + [Guid]::NewGuid().ToString('N') + '.py')
    try {
        @(
            'import json, torch',
            'if not torch.cuda.is_available():',
            '    raise SystemExit("CUDA is not available to PyTorch")',
            'print(json.dumps({"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_device_name": torch.cuda.get_device_name(0)}, sort_keys=True))'
        ) | Set-Content -Encoding UTF8 $cudaProbe
        $cudaText = (& python $cudaProbe | Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw "Gate-2 resource CUDA preflight failed."
        }
        $cuda = $cudaText | ConvertFrom-Json
        if ([string]$cuda.cuda_device_name -ne "NVIDIA GeForce RTX 4060 Ti") {
            throw "Frozen Gate-2 resource protocol primary target is NVIDIA GeForce RTX 4060 Ti; observed: $($cuda.cuda_device_name)"
        }
        Write-Host ($cuda | ConvertTo-Json -Depth 4)
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $cudaProbe
    }

    New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null
    $audit | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "confirmation-audit.json")
    Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "git-head.txt") -Value $head
    New-Item -ItemType File -Force -Path (Join-Path $resolvedOutputRoot "git-status.txt") | Out-Null

    $checkpointSha = (Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant()
    $runConfig = [ordered]@{
        protocol = "gate2-persistent-state-resource-frontier-v0"
        scientific_status = "FROZEN_RESOURCE_TIMING"
        confirmation_measurement_head = $ConfirmationMeasurementHead
        resource_runner_head = $head
        confirmation_root = $resolvedConfirmationRoot
        checkpoint = $checkpointPath
        checkpoint_sha256 = $checkpointSha
        checkpoint_training_seed = 3
        entity_widths = [ordered]@{ "64" = @(1,4,16,64); "256" = @(1,4,16,64,256) }
        batch_sizes = @(1,64)
        warmup_iterations = 10
        timed_iterations = 50
        resource_world_seed_start = 4294967296
        execution_mode = "eager_cuda"
        compiler_enabled = $false
        idle_machine_attested = $true
        gpu = [string]$cuda.cuda_device_name
        torch_version = [string]$cuda.torch_version
        cuda_runtime = [string]$cuda.cuda_runtime
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "run-config.json")

    try {
        & nvidia-smi -q 2>&1 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }

    $resultPath = Join-Path $resolvedOutputRoot "gate2-resource-frontier.json"
    Write-Host ""
    Write-Host "Running frozen Gate-2 eager-CUDA resource frontier..."
    & python -m ai_hypothesis.population_compute.run_gate2_resource_frontier `
        --checkpoint $checkpointPath `
        --output $resultPath
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-2 resource frontier measurement failed."
    }

    try {
        & nvidia-smi -q 2>&1 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    $summary = [ordered]@{
        protocol = "gate2-persistent-state-resource-frontier-v0"
        resource_frontier_passed = [bool]$result.resource_frontier_passed
        all_preflights_passed = [bool]$result.all_preflights_passed
        decision_endpoint_passes = $result.decision_endpoint_passes
        capability_confirmation_passed = $true
        overall_gate2_v0_passed = ([bool]$result.resource_frontier_passed)
        overall_gate2_verdict = if ([bool]$result.resource_frontier_passed) { "POSITIVE_V0" } else { "NOT_POSITIVE_V0_RESOURCE_HALF_FAILED" }
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "gate2-v0-summary.json")

    $artifactNames = @(
        "gate2-resource-frontier.json",
        "gate2-v0-summary.json",
        "confirmation-audit.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi-before.txt",
        "nvidia-smi-after.txt"
    )
    $manifest = foreach ($name in $artifactNames) {
        $path = Join-Path $resolvedOutputRoot $name
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Expected Gate-2 resource artifact missing: $name"
        }
        $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
        "$hash  $name"
    }
    $manifest | Set-Content -Encoding ASCII -Path (Join-Path $resolvedOutputRoot "result-manifest.sha256")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-2 v0 resource measurement complete"
    Write-Host "============================================================"
    Write-Host "Capability confirmation passed: True"
    Write-Host "Resource frontier passed: $($result.resource_frontier_passed)"
    Write-Host "Overall Gate-2 v0 verdict: $($summary.overall_gate2_verdict)"
    Write-Host "Result: $resultPath"
}
finally {
    Pop-Location
}
