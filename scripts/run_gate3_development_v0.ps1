param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate3_hypothesis_population_v0_development_seed_0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Gate-3 development runner did not produce required file: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-3 development requires explicit -IdleMachineAttested."
    }

    if (Get-Process -Name "factorio" -ErrorAction SilentlyContinue) {
        throw "Factorio is running. Close it before the admitted Gate-3 development run."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-3 development requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "agent/gate3-hypothesis-population-v0") {
        throw "Gate-3 development must run from agent/gate3-hypothesis-population-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-3 Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-3 output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    # Initialize every later artifact path before any preflight or banner output. This keeps the
    # Windows PowerShell 5.1 + StrictMode execution path deterministic and avoids deferred-variable
    # failures before the scientific Python runner has started.
    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate3-development.json"
    $checkpointPath = Join-Path $scienceRoot "gate3-development-checkpoint.pt"
    $runtimePath = Join-Path $scienceRoot "runtime.json"
    $auditPath = Join-Path $resolvedOutputRoot "development-audit.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate3-hypothesis-population-v0"
        development_recipe = "gate3-development-recipe-v0"
        scientific_status = "DEVELOPMENT_ONLY_NOT_ASSIGNED"
        confirmation_opened = $false
        training_seed = 0
        steps = 1200
        training_batch_size = 128
        learning_rate = 0.0003
        weight_decay = 0.0001
        gradient_clip_norm = 1.0
        learned_parameter_count = 19873
        score_quantization = 0.001
        evaluation_world_start = 1073741824
        evaluation_world_count = 256
        evaluation_batch_size = 64
        bootstrap_samples = 2000
        device = "cuda"
        idle_machine_attested = $true
        compiler_enabled = $false
        cuda_graphs_enabled = $false
        mixed_precision_enabled = $false
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    # CI-only wrapper smoke mode exercises Windows PowerShell 5.1 + StrictMode through the banner
    # without touching CUDA or the scientific Python runner. It cannot create scientific evidence.
    $wrapperSmoke = $env:GATE3_WRAPPER_SMOKE -eq "1"

    if (-not $wrapperSmoke) {
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
            throw "Gate-3 CUDA preflight failed."
        }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-3 v0 DEVELOPMENT ONLY - seed 0"
    Write-Host "============================================================"
    Write-Host "Git head: $head"
    Write-Host "Output:   $resolvedOutputRoot"
    Write-Host "Confirmation remains CLOSED."
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate-3 wrapper smoke completed before scientific execution."
        return
    }

    python -m ai_hypothesis.population_compute.run_gate3_development_v0 `
        --output-root $scienceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-3 development science runner failed. Preserve the output root for diagnosis."
    }

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $checkpointPath
    Assert-RequiredLeaf -Path $runtimePath

    Write-Host ""
    Write-Host "Independently auditing Gate-3 development artifact..."
    python -m ai_hypothesis.population_compute.analyze_gate3_development `
        $resultPath `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Gate-3 development audit rejected the artifact. Preserve all output."
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

    $audit = Get-Content -Raw $auditPath | ConvertFrom-Json
    $checkpointHash = (Get-FileHash -Algorithm SHA256 $checkpointPath).Hash
    $resultHash = (Get-FileHash -Algorithm SHA256 $resultPath).Hash
    $auditHash = (Get-FileHash -Algorithm SHA256 $auditPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-3 development seed 0 complete"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Development outcome:  $($audit.directional_outcome)"
    Write-Host "Scientific status:    DEVELOPMENT ONLY - NO GATE VERDICT"
    Write-Host "Confirmation opened:  False"
    Write-Host "Checkpoint SHA256:     $checkpointHash"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Audit SHA256:          $auditHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
    Write-Host ""
    Write-Host "Primary paired effects:"
    $audit.primary_deltas | ConvertTo-Json -Depth 4 | Write-Host
}
finally {
    Pop-Location
}
