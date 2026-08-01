param(
    [Parameter(Mandatory = $true)]
    [string]$PopulationRoot,

    [Parameter(Mandatory = $true)]
    [string]$ReferenceRoot,

    [string]$OutputRoot = "F:\gate8_v1_final_comparison_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $wrapperSmoke = $env:GATE8_V1_FINAL_COMPARISON_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 v1 final comparison requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate8-v1-final-comparison-execution-v0") {
        throw "Gate8 v1 final comparison must run from agent/gate8-v1-final-comparison-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 v1 final-comparison Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 v1 final-comparison output already exists: $resolvedOutputRoot"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 FINAL POPULATION-vs-GEMMA COMPARISON"
    Write-Host "============================================================"
    Write-Host "Git head:         $head"
    Write-Host "Output:           $resolvedOutputRoot"
    Write-Host "Conditions:       21"
    Write-Host "Worlds:           10,752"
    Write-Host "Population seeds: 0, 1, 2"
    Write-Host "Bootstrap:        20,000 paired world-index replicates"
    Write-Host "Condition weight: equal across all 21 conditions"
    Write-Host "Model loading:    NONE"
    Write-Host "Inference:        NONE"
    Write-Host "Population rerun: NONE"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 v1 final-comparison wrapper smoke completed before package imports, source-artifact access, output creation, row parsing, bootstrap execution, or classification."
        return
    }

    $resolvedPopulation = (Resolve-Path -LiteralPath $PopulationRoot).Path
    $resolvedReference = (Resolve-Path -LiteralPath $ReferenceRoot).Path

    $preflight = @'
import importlib.metadata
import platform
expected = {"python": "3.11.9", "numpy": "2.3.5"}
observed = {
    "python": platform.python_version(),
    "numpy": importlib.metadata.version("numpy"),
}
for name, value in expected.items():
    if observed[name] != value:
        raise SystemExit(f"software drift: {name} expected={value} observed={observed[name]}")
print(observed)
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 final-comparison Python/NumPy preflight failed."
    }

    python scripts/run_gate8_v1_final_comparison.py `
        --population-root $resolvedPopulation `
        --reference-root $resolvedReference `
        --output-root $resolvedOutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 final comparison failed. Preserve the source evidence and any diagnostic output."
    }

    $summaryPath = Join-Path $resolvedOutputRoot "comparison\gate8-v1-final-comparison-summary.json"
    $conditionPath = Join-Path $resolvedOutputRoot "comparison\gate8-v1-final-comparison-per-condition.jsonl"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate8 v1 final-comparison summary is missing."
    }
    if (-not (Test-Path -LiteralPath $conditionPath -PathType Leaf)) {
        throw "Gate8 v1 final-comparison condition ledger is missing."
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Gate8 v1 final-comparison manifest is missing."
    }

    $summary = Get-Content -Raw $summaryPath | ConvertFrom-Json
    if ($summary.scientific_status -ne "G8_V1_FINAL_COMPARISON_COMPLETE") {
        throw "Gate8 v1 final-comparison scientific status is invalid."
    }
    $allowed = @(
        "G8_POPULATION_EXCEEDS_1B_REFERENCE",
        "G8_POPULATION_NONINFERIOR_TO_1B_REFERENCE",
        "G8_1B_REFERENCE_SUPERIOR",
        "G8_1B_REFERENCE_MIXED",
        "G8_1B_REFERENCE_COMPARISON_INCONCLUSIVE"
    )
    if ($summary.reference_comparison_classification -notin $allowed) {
        throw "Gate8 v1 final-comparison classifier returned an unknown outcome."
    }
    if ($summary.boundaries.population_execution_performed -ne $false -or
        $summary.boundaries.reference_model_loaded -ne $false -or
        $summary.boundaries.reference_inference_performed -ne $false -or
        $summary.boundaries.training_performed -ne $false -or
        $summary.boundaries.world_generation_performed -ne $false) {
        throw "Gate8 v1 final comparison crossed a closed execution boundary."
    }
    if ($summary.boundaries.joint_reference_comparison_classified -ne $true) {
        throw "Gate8 v1 final comparison did not classify the frozen comparison."
    }

    $summaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $summaryPath).Hash
    $conditionHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $conditionPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 final comparison complete"
    Write-Host "============================================================"
    Write-Host "Status:                   $($summary.scientific_status)"
    Write-Host "Population scaling:       $($summary.population_scaling_classification)"
    Write-Host "Reference comparison:     $($summary.reference_comparison_classification)"
    Write-Host "Population accuracy:      $($summary.population_mean_accuracy)"
    Write-Host "Reference accuracy:       $($summary.reference_mean_accuracy)"
    Write-Host "Population - reference:   $($summary.pooled_comparison.population_minus_reference_delta)"
    Write-Host "Paired 95% CI:            [$($summary.pooled_comparison.bootstrap_ci_low), $($summary.pooled_comparison.bootstrap_ci_high)]"
    Write-Host "Summary SHA256:           $summaryHash"
    Write-Host "Condition ledger SHA256:  $conditionHash"
    Write-Host "Manifest SHA256:          $manifestHash"
    Write-Host "Output root:              $resolvedOutputRoot"
}
finally {
    Pop-Location
}
