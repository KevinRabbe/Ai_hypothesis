param(
    [switch]$IdleMachineAttested,
    [string]$OutputRoot = "results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/confirmation_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $IdleMachineAttested) {
    throw "Gate-2 confirmation requires -IdleMachineAttested after closing Factorio and other GPU-heavy applications."
}

$factorio = Get-Process -Name "factorio" -ErrorAction SilentlyContinue
if ($factorio) {
    throw "Factorio is still running. Close it before Gate-2 confirmation."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $gitStatus = @(& git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect Git working tree."
    }
    if ($gitStatus.Count -ne 0) {
        Write-Host "Gate-2 confirmation requires a clean working tree for provenance."
        $gitStatus | ForEach-Object { Write-Host $_ }
        throw "Working tree is not clean."
    }

    $head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $head) {
        throw "Could not resolve Git HEAD."
    }

    $resolvedOutputRoot = Join-Path $RepoRoot $OutputRoot
    if (Test-Path $resolvedOutputRoot) {
        throw "Confirmation output root already exists: $resolvedOutputRoot"
    }

    $cudaProbe = Join-Path ([System.IO.Path]::GetTempPath()) ('gate2-confirmation-cuda-' + [Guid]::NewGuid().ToString('N') + '.py')
    try {
        @(
            'import json, torch',
            'if not torch.cuda.is_available():',
            '    raise SystemExit("CUDA is not available to PyTorch")',
            'print(json.dumps({"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_device_name": torch.cuda.get_device_name(0)}, indent=2))'
        ) | Set-Content -Encoding UTF8 $cudaProbe
        & python $cudaProbe
        if ($LASTEXITCODE -ne 0) {
            throw "Gate-2 confirmation CUDA preflight failed."
        }
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $cudaProbe
    }

    New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null
    Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "git-head.txt") -Value $head
    New-Item -ItemType File -Force -Path (Join-Path $resolvedOutputRoot "git-status.txt") | Out-Null

    $runConfig = [ordered]@{
        protocol = "gate2-persistent-state-confirmation-v0"
        scientific_status = "FROZEN_CONFIRMATION"
        training_seeds = @(3, 4, 5)
        steps = 1000
        training_batch_size = 32
        state_width = 64
        query_width = 24
        learning_rate = 3e-4
        weight_decay = 1e-4
        gradient_clip_norm = 1.0
        evaluation_world_count = 512
        evaluation_batch_size = 64
        bootstrap_samples = 2000
        device = "cuda"
        idle_machine_attested = $true
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "run-config.json")

    try {
        & nvidia-smi -q 2>&1 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-before.txt")
    }

    $seedSummaries = @()
    foreach ($seed in @(3, 4, 5)) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host " Gate-2 CONFIRMATION seed $seed / seeds 3,4,5"
        Write-Host "============================================================"
        Write-Host ""

        $seedRoot = Join-Path $resolvedOutputRoot "seed_$seed"
        & python -m ai_hypothesis.population_compute.run_gate2_confirmation `
            --output-root $seedRoot `
            --training-seed $seed `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "Gate-2 confirmation seed $seed failed."
        }

        $resultPath = Join-Path $seedRoot "gate2-confirmation.json"
        $checkpointPath = Join-Path $seedRoot "gate2-confirmation-checkpoint.pt"
        $runtimePath = Join-Path $seedRoot "runtime.json"
        foreach ($required in @($resultPath, $checkpointPath, $runtimePath)) {
            if (-not (Test-Path $required)) {
                throw "Expected Gate-2 confirmation artifact missing: $required"
            }
        }

        $result = Get-Content -Raw $resultPath | ConvertFrom-Json
        $manifest = @(
            "$(($(Get-FileHash -Algorithm SHA256 $resultPath).Hash.ToLowerInvariant()))  gate2-confirmation.json",
            "$(($(Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant()))  gate2-confirmation-checkpoint.pt",
            "$(($(Get-FileHash -Algorithm SHA256 $runtimePath).Hash.ToLowerInvariant()))  runtime.json"
        )
        $manifest | Set-Content -Encoding ASCII -Path (Join-Path $seedRoot "result-manifest.sha256")

        $seedSummaries += [ordered]@{
            training_seed = $seed
            seed_passed = [bool]$result.seed_passed
            width1_identity_passed = [bool]$result.width1_identity_passed
            result_sha256 = (Get-FileHash -Algorithm SHA256 $resultPath).Hash.ToLowerInvariant()
            checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 $checkpointPath).Hash.ToLowerInvariant()
            parameter_fingerprint = [string]$result.training.parameter_fingerprint
            primary_comparisons = $result.primary_comparisons
        }

        Write-Host ""
        Write-Host "Seed $seed confirmation pass: $($result.seed_passed)"
    }

    try {
        & nvidia-smi -q 2>&1 | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }

    $capabilityPassed = ($seedSummaries.Count -eq 3) -and (($seedSummaries | Where-Object { -not $_.seed_passed }).Count -eq 0)
    $suite = [ordered]@{
        protocol = "gate2-persistent-state-confirmation-v0"
        confirmation_training_seeds = @(3, 4, 5)
        capability_confirmation_passed = $capabilityPassed
        gate2_overall_verdict = "NOT_ASSIGNED_UNTIL_RESOURCE_PROTOCOL_COMPLETE"
        seeds = $seedSummaries
    }
    $suitePath = Join-Path $resolvedOutputRoot "confirmation-suite.json"
    $suite | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path $suitePath

    $topArtifacts = @(
        "confirmation-suite.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi-before.txt",
        "nvidia-smi-after.txt"
    )
    $topManifest = foreach ($name in $topArtifacts) {
        $path = Join-Path $resolvedOutputRoot $name
        if (-not (Test-Path $path)) {
            throw "Expected top-level confirmation artifact missing: $name"
        }
        $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
        "$hash  $name"
    }
    $topManifest | Set-Content -Encoding ASCII -Path (Join-Path $resolvedOutputRoot "suite-manifest.sha256")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-2 capability confirmation complete"
    Write-Host "============================================================"
    Write-Host "Capability confirmation passed: $capabilityPassed"
    Write-Host "Overall Gate-2 verdict: NOT ASSIGNED until resource protocol completes"
    Write-Host "Suite: $suitePath"
}
finally {
    Pop-Location
}
