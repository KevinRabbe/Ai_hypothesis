param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_execution_engine_profile_v0",
    [string]$Checkpoint = "F:\gate3_v1_sparse_active_reserve_development_seed_0\science\gate3-v1-development-checkpoint.pt",
    [int]$WorldCount = 64,
    [int]$FrontierDepth = 8,
    [int]$Repeats = 3
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required Gate-7 engineering file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 execution profiling requires explicit -IdleMachineAttested."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 execution profiling requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "agent/gate7-high-scale-frontier-prep-v0") {
        throw "Gate-7 execution profiling must run from agent/gate7-high-scale-frontier-prep-v0."
    }

    if ($WorldCount -lt 1 -or $WorldCount -gt 256) {
        throw "WorldCount must be in 1..256 for this engineering profile."
    }
    if ($FrontierDepth -lt 1 -or $FrontierDepth -gt 9) {
        throw "FrontierDepth must be in 1..9 for the existing frozen checkpoint."
    }
    if ($Repeats -lt 1) {
        throw "Repeats must be positive."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-7 preparation Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 engineering output root already exists: $resolvedOutputRoot"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 EXECUTION ENGINE PROFILE - ENGINEERING ONLY"
    Write-Host "============================================================"
    Write-Host "Git head:       $head"
    Write-Host "Output:         $resolvedOutputRoot"
    Write-Host "Worlds:         $WorldCount synthetic/public engineering worlds"
    Write-Host "Frontier depth: $FrontierDepth"
    Write-Host "Repeats:        $Repeats"
    Write-Host "Compiler:       OFF"
    Write-Host "CUDA graphs:    OFF"
    Write-Host "Mixed precision:OFF"
    Write-Host "NO Gate-7 scientific namespace or result is used."
    Write-Host ""

    $wrapperSmoke = $env:GATE7_PROFILE_WRAPPER_SMOKE -eq "1"
    if ($wrapperSmoke) {
        Write-Host "Gate-7 engineering profile wrapper smoke completed before checkpoint/CUDA execution."
        return
    }

    Assert-RequiredLeaf -Path $Checkpoint

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
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 engineering CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.profile_gate7_execution_engine `
        --output-root $resolvedOutputRoot `
        --checkpoint $Checkpoint `
        --world-count $WorldCount `
        --frontier-depth $FrontierDepth `
        --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 engineering profiler failed. Preserve the output root for diagnosis."
    }

    Write-Host ""
    Write-Host "Profile complete: $resolvedOutputRoot"
    Write-Host "This is engineering evidence only, not Gate-7 scientific evidence."
}
finally {
    Pop-Location
}
