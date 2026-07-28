param(
    [int]$TrainingSeed = 0,
    [int]$Steps = 1000,
    [int]$TrainingBatchSize = 32,
    [int]$EvaluationWorldCount = 256,
    [int]$EvaluationBatchSize = 64,
    [int]$BootstrapSamples = 2000,
    [string]$OutputRoot = "results/population_compute_scaling_v0/gate2_persistent_state_capacity_v0/development_seed_0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $gitStatus = @(& git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect Git working tree."
    }
    if ($gitStatus.Count -ne 0) {
        Write-Host "Gate-2 development requires a clean working tree for provenance."
        $gitStatus | ForEach-Object { Write-Host $_ }
        throw "Working tree is not clean."
    }

    $head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $head) {
        throw "Could not resolve Git HEAD."
    }

    $resolvedOutputRoot = Join-Path $RepoRoot $OutputRoot
    if (Test-Path $resolvedOutputRoot) {
        throw "Output root already exists: $resolvedOutputRoot"
    }

    $cudaProbe = Join-Path ([System.IO.Path]::GetTempPath()) ('gate2-cuda-probe-' + [Guid]::NewGuid().ToString('N') + '.py')
    try {
        @(
            'import json, torch',
            'if not torch.cuda.is_available():',
            '    raise SystemExit("CUDA is not available to PyTorch")',
            'print(json.dumps({"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_device_name": torch.cuda.get_device_name(0)}, indent=2))'
        ) | Set-Content -Encoding UTF8 $cudaProbe
        & python $cudaProbe
        if ($LASTEXITCODE -ne 0) {
            throw "Gate-2 development CUDA preflight failed."
        }
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $cudaProbe
    }

    New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null
    $gitHeadPath = Join-Path $resolvedOutputRoot "git-head.txt"
    $gitStatusPath = Join-Path $resolvedOutputRoot "git-status.txt"
    $runConfigPath = Join-Path $resolvedOutputRoot "run-config.json"
    $nvidiaSmiPath = Join-Path $resolvedOutputRoot "nvidia-smi.txt"

    Set-Content -Encoding UTF8 -Path $gitHeadPath -Value $head
    if ($gitStatus.Count -eq 0) {
        New-Item -ItemType File -Force -Path $gitStatusPath | Out-Null
    }
    else {
        $gitStatus | Set-Content -Encoding UTF8 -Path $gitStatusPath
    }

    $runConfig = [ordered]@{
        scientific_status = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
        training_seed = $TrainingSeed
        steps = $Steps
        training_batch_size = $TrainingBatchSize
        evaluation_world_count = $EvaluationWorldCount
        evaluation_batch_size = $EvaluationBatchSize
        bootstrap_samples = $BootstrapSamples
        device = "cuda"
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $runConfigPath

    try {
        & nvidia-smi -q 2>&1 | Set-Content -Encoding UTF8 -Path $nvidiaSmiPath
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" | Set-Content -Encoding UTF8 -Path $nvidiaSmiPath
    }

    Write-Host "Running Gate-2 development-only training/evaluation on CUDA..."
    & python -m ai_hypothesis.population_compute.run_gate2_development `
        --output-root $resolvedOutputRoot `
        --training-seed $TrainingSeed `
        --steps $Steps `
        --training-batch-size $TrainingBatchSize `
        --evaluation-world-count $EvaluationWorldCount `
        --evaluation-batch-size $EvaluationBatchSize `
        --bootstrap-samples $BootstrapSamples `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-2 development run failed."
    }

    $manifestPath = Join-Path $resolvedOutputRoot "result-manifest.sha256"
    $artifactNames = @(
        "gate2-development.json",
        "gate2-development-checkpoint.pt",
        "runtime.json",
        "run-config.json",
        "git-head.txt",
        "git-status.txt",
        "nvidia-smi.txt"
    )
    $manifestLines = foreach ($name in $artifactNames) {
        $path = Join-Path $resolvedOutputRoot $name
        if (-not (Test-Path $path)) {
            throw "Expected Gate-2 development artifact is missing: $name"
        }
        $hash = (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLowerInvariant()
        "$hash  $name"
    }
    $manifestLines | Set-Content -Encoding ASCII -Path $manifestPath

    Write-Host ""
    Write-Host "Gate-2 development-only run completed."
    Write-Host "This is NOT confirmation evidence and assigns no Gate-2 verdict."
    Write-Host "Git HEAD:  $head"
    Write-Host "Result:    $(Join-Path $resolvedOutputRoot 'gate2-development.json')"
    Write-Host "Checkpoint:$(Join-Path $resolvedOutputRoot 'gate2-development-checkpoint.pt')"
    Write-Host "Manifest:  $manifestPath"
}
finally {
    Pop-Location
}
