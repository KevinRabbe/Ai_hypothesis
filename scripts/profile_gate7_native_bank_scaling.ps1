param(
    [Parameter(Mandatory = $true)]
    [switch]$IdleMachineAttested,

    [string]$OutputRoot = "F:\gate7_native_bank_scaling_profile_v0",
    [int]$BatchSize = 32,
    [int]$Slots = 128,
    [int]$Repeats = 3
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $IdleMachineAttested) {
        throw "Gate-7 native-bank engineering profile requires explicit -IdleMachineAttested."
    }

    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate-7 native-bank profile requires a clean Git working tree."
    }

    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "agent/gate7-high-scale-frontier-prep-v0") {
        throw "Gate-7 native-bank profile must run from agent/gate7-high-scale-frontier-prep-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate-7 preparation Git head."
    }
    if ($BatchSize -le 0 -or $Slots -le 0 -or $Slots -gt 128 -or $Repeats -le 0) {
        throw "Invalid engineering profile execution parameters."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    $summaryPath = Join-Path $resolvedOutputRoot "summary.json"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 NATIVE BANK SCALING PROFILE - ENGINEERING ONLY"
    Write-Host "============================================================"
    Write-Host "Git head:       $head"
    Write-Host "Output:         $resolvedOutputRoot"
    Write-Host "Population:     512 -> 131072"
    Write-Host "K profile:      16 / 64 / 256 / 512 where bounded"
    Write-Host "Batch size:     $BatchSize fixed across all N"
    Write-Host "Routing slots:  $Slots"
    Write-Host "Repeats:        $Repeats"
    Write-Host "Compiler:       OFF"
    Write-Host "CUDA graphs:    OFF"
    Write-Host "Mixed precision:OFF"
    Write-Host "NO checkpoint, hidden path, or Gate-7 scientific world is used."
    Write-Host ""

    if ($env:GATE7_NATIVE_PROFILE_WRAPPER_SMOKE -eq "1") {
        Write-Host "Gate-7 native-bank profile wrapper smoke completed before CUDA execution."
        return
    }

    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate-7 native-bank profile output root already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null
    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

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
        throw "Gate-7 native-bank CUDA preflight failed."
    }

    python -m ai_hypothesis.population_compute.profile_gate7_native_bank_scaling `
        --output $summaryPath `
        --batch-size $BatchSize `
        --slots $Slots `
        --repeats $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "Gate-7 native-bank scaling profiler failed. Preserve the output root for diagnosis."
    }
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate-7 native-bank profile did not produce summary.json."
    }

    try {
        & nvidia-smi | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }
    catch {
        "nvidia-smi capture failed: $($_.Exception.Message)" |
            Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "nvidia-smi-after.txt")
    }

    Write-Host ""
    Write-Host "Profile complete: $resolvedOutputRoot"
    Write-Host "This is engineering evidence only, not Gate-7 scientific evidence."
}
finally {
    Pop-Location
}
