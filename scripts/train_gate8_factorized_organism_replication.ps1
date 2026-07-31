param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$Seed,

    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $wrapperSmoke = $env:GATE8_V1_REPLICATION_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 v1 replication requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve current Gate8 v1 replication branch."
    }
    $branch = if ($null -eq $branchOutput) {
        ""
    } else {
        "$branchOutput".Trim()
    }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if (
        $branch -ne
        "agent/gate8-factorized-message-training-replication-execution-v1"
    ) {
        throw (
            "Gate8 v1 replication must run from " +
            "agent/gate8-factorized-message-training-replication-execution-v1."
        )
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 v1 replication Git head."
    }

    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputRoot = "F:\gate8_factorized_organism_training_seed${Seed}_v1"
    }
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    $resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")
    if (
        $resolvedOutputRoot.StartsWith(
            $resolvedRepoRoot + "\",
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Gate8 v1 replication output must be outside the Git repository."
    }
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 v1 replication output already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $trainingRoot = Join-Path $resolvedOutputRoot "training"
    $resultPath = Join-Path (
        $trainingRoot
    ) "gate8-factorized-organism-training-result.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (
        Join-Path $resolvedOutputRoot "git-head.txt"
    )
    (git status --porcelain) | Set-Content -Encoding UTF8 (
        Join-Path $resolvedOutputRoot "git-status.txt"
    )

    $runConfig = [ordered]@{
        experiment = "gate8-factorized-message-training-execution-v1"
        replication_execution = (
            "gate8-factorized-message-training-replication-execution-v1"
        )
        qualified_seed0_result_head = (
            "f259620f7d3beab2f886c76271c753e9ebf96dc9"
        )
        protocol_head = "a33dc123d090268a531d112251ea3ab53cb50062"
        runtime_head = "333d88ac4fc52f1651741fba224e0b4605feedd3"
        architecture_head = "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
        seed = $Seed
        admitted_seeds = @(1, 2)
        training_worlds = 262144
        world_batch_size = 256
        optimizer_steps = 1024
        checkpoint_steps = @(256, 512, 768, 1024)
        validation_world_index_start = 512
        validation_world_index_end_inclusive = 1023
        parameter_dtype = "float32"
        autocast = $false
        tf32 = $false
        scientific_test_worlds_generated = $false
        reference_model_loaded = $false
        git_head = $head
    }
    $runConfig |
        ConvertTo-Json -Depth 8 |
        Set-Content -Encoding UTF8 (
            Join-Path $resolvedOutputRoot "run-config.json"
        )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 V1 FACTORIZED REPLICATION TRAINING"
    Write-Host "============================================================"
    Write-Host "Git head:          $head"
    Write-Host "Seed:              $Seed"
    Write-Host "Output:            $resolvedOutputRoot"
    Write-Host "Training worlds:   262,144"
    Write-Host "Optimizer steps:   1,024"
    Write-Host "Checkpoints:       256, 512, 768, 1024"
    Write-Host "Validation indices: 512..1023"
    Write-Host "Seed 0 rerun:      FORBIDDEN"
    Write-Host "Scientific test:   FORBIDDEN"
    Write-Host "Reference model:   FORBIDDEN"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host (
            "Gate8 v1 replication wrapper smoke completed before Torch, " +
            "CUDA, worlds, optimizer or checkpoints."
        )
        return
    }

    $env:PYTHONHASHSEED = "$Seed"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $preflight = @'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Gate8 v1 replication")
print(f"gpu={torch.cuda.get_device_name(0)}")
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 replication CUDA preflight failed."
    }

    python scripts/train_gate8_factorized_organism_replication.py `
        --seed $Seed `
        --device cuda `
        --output-root $trainingRoot
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Gate8 v1 replication failed. Preserve the complete output root."
        )
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Gate8 v1 replication result is missing: $resultPath"
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    if ($result.seed -ne $Seed) {
        throw "Gate8 v1 replication result seed drifted."
    }
    if (
        $result.replication_execution_version -ne
        "gate8-factorized-message-training-replication-execution-v1"
    ) {
        throw "Gate8 v1 replication result version drifted."
    }
    if (
        $result.qualified_seed0_result_head -ne
        "f259620f7d3beab2f886c76271c753e9ebf96dc9"
    ) {
        throw "Gate8 v1 replication result lost seed-0 provenance."
    }
    if ($result.learned_parameter_count -ne 19649) {
        throw "Gate8 v1 replication parameter count drifted."
    }
    if (
        $result.training_worlds -ne 262144 -or
        $result.optimizer_steps -ne 1024
    ) {
        throw "Gate8 v1 replication schedule drifted."
    }
    if (
        $result.validation.world_index_start -ne 512 -or
        $result.validation.world_index_end_inclusive -ne 1023
    ) {
        throw "Gate8 v1 replication validation range drifted."
    }
    if (
        $result.training_performed -ne $true -or
        $result.validation_performed -ne $true
    ) {
        throw "Gate8 v1 replication lacks completed train/validation evidence."
    }
    if ($result.seeds_1_and_2_executed -ne $true) {
        throw "Gate8 v1 replication result did not mark replication execution."
    }
    if ($result.scientific_test_worlds_generated -ne $false) {
        throw "Gate8 v1 replication crossed the scientific-test boundary."
    }
    if (
        $result.reference_tokenizer_loaded -ne $false -or
        $result.reference_model_weights_loaded -ne $false -or
        $result.reference_inference_performed -ne $false
    ) {
        throw "Gate8 v1 replication crossed the reference-model boundary."
    }
    if (@($result.checkpoint_candidates).Count -ne 4) {
        throw "Gate8 v1 replication did not preserve four checkpoints."
    }
    if (
        -not (
            Test-Path -LiteralPath (
                Join-Path $trainingRoot "selected-checkpoint.pt"
            ) -PathType Leaf
        )
    ) {
        throw "Gate8 v1 replication selected checkpoint is missing."
    }

    $resolvedBase = (Resolve-Path $resolvedOutputRoot).Path.TrimEnd("\")
    @(
        Get-ChildItem -LiteralPath $resolvedBase -File -Recurse |
        Where-Object { $_.FullName -ne $manifestPath } |
        ForEach-Object {
            $relative = $_.FullName.Substring(
                $resolvedBase.Length
            ).TrimStart("\").Replace("\", "/")
            $hash = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
            ).Hash.ToLowerInvariant()
            "$hash  $relative"
        } |
        Sort-Object
    ) | Set-Content -Encoding ASCII $manifestPath

    $resultHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath
    ).Hash
    $manifestHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath
    ).Hash
    $selected = $result.selected_checkpoint

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 replication training complete"
    Write-Host "============================================================"
    Write-Host "Status:                   $($result.scientific_status)"
    Write-Host "Seed:                     $Seed"
    Write-Host "Selected step:            $($selected.step)"
    Write-Host "Selected admitted:        $($selected.admitted)"
    Write-Host "Selected checkpoint SHA:  $($selected.sha256)"
    Write-Host "Result SHA256:             $resultHash"
    Write-Host "Manifest SHA256:           $manifestHash"
    Write-Host "Output root:               $resolvedOutputRoot"
}
finally {
    Pop-Location
}
