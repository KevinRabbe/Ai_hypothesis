param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1, 2)]
    [int]$Seed,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $smoke = $env:GATE9_CONTEXTUAL_TRAINING_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate9 contextual training requires a clean Git working tree."
    }
    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $smoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate9-contextual-training-execution-v0") {
        throw "Gate9 contextual training must run from agent/gate9-contextual-training-execution-v0."
    }
    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate9 training Git head."
    }
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutput) {
        throw "Gate9 training output already exists: $resolvedOutput"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-9 CONTEXTUAL WORKER TRAINING — SEED $Seed"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Output:            $resolvedOutput"
    Write-Host "Training episodes: 262,144"
    Write-Host "Batch / steps:     512 / 512"
    Write-Host "Validation:        32,768 unseen operators"
    Write-Host "Checkpoint:        fixed final step only"
    Write-Host "Scientific test:   CLOSED"
    Write-Host ""

    if ($smoke) {
        Write-Host "Gate9 training wrapper smoke completed before package imports, CUDA access, operator generation, output creation, optimizer construction, training, validation or checkpoint serialization."
        return
    }

    $preflight = @'
import importlib.metadata
import platform
import torch
expected = {
    "python": "3.11.9",
    "torch": "2.9.1+cu130",
    "numpy": "2.3.5",
}
observed = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": importlib.metadata.version("numpy"),
}
for name, value in expected.items():
    if observed[name] != value:
        raise SystemExit(f"software drift: {name} expected={value} observed={observed[name]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(observed)
print("cuda_device=" + torch.cuda.get_device_name(0))
print("cuda_capability=" + str(torch.cuda.get_device_capability(0)))
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate9 training Python/CUDA preflight failed."
    }

    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    python scripts/run_gate9_contextual_training.py `
        --seed $Seed `
        --output-root $resolvedOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Gate9 seed-$Seed training failed. Preserve the complete output root."
    }

    $seedRoot = Join-Path $resolvedOutput "seed-$Seed"
    $summaryPath = Join-Path $seedRoot "summary.json"
    $validationPath = Join-Path $seedRoot "validation-per-episode.jsonl"
    $checkpointPath = Join-Path $seedRoot "selected-checkpoint.pt"
    $manifestPath = Join-Path $resolvedOutput "manifest.sha256"
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if ($summary.scientific_status -ne "G9_CONTEXTUAL_TRAINING_SEED_COMPLETE") {
        throw "Gate9 seed summary status is invalid."
    }
    if ($summary.boundaries.scientific_test_generated -ne $false -or
        $summary.boundaries.scientific_execution_performed -ne $false -or
        $summary.boundaries.local_test_operator_accessed -ne $false -or
        $summary.boundaries.graph_test_operator_accessed -ne $false -or
        $summary.boundaries.scientific_assignment_key_accessed -ne $false) {
        throw "Gate9 training crossed a closed scientific boundary."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-9 seed-$Seed training complete"
    Write-Host "============================================================"
    Write-Host "Validation passes:  $($summary.validation_evidence.admission_passes)"
    Write-Host "Exact accuracy:      $($summary.validation_evidence.validation_exact_accuracy)"
    Write-Host "Bit accuracy:        $($summary.validation_evidence.validation_bit_accuracy)"
    Write-Host "Shuffled accuracy:   $($summary.validation_evidence.shuffled_context_accuracy)"
    Write-Host "Query-only accuracy: $($summary.validation_evidence.query_only_accuracy)"
    Write-Host "Summary SHA256:      $((Get-FileHash -Algorithm SHA256 -LiteralPath $summaryPath).Hash)"
    Write-Host "Validation SHA256:   $((Get-FileHash -Algorithm SHA256 -LiteralPath $validationPath).Hash)"
    Write-Host "Checkpoint SHA256:   $((Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointPath).Hash)"
    Write-Host "Manifest SHA256:     $((Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash)"
    Write-Host "Output root:         $resolvedOutput"
}
finally {
    Pop-Location
}
