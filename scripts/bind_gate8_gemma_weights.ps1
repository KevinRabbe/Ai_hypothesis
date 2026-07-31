param(
    [Parameter(Mandatory = $true)]
    [switch]$HuggingFaceLicenseAndAccessAttested,

    [string]$OutputRoot = "F:\gate8_gemma_weight_binding_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $HuggingFaceLicenseAndAccessAttested) {
        throw "Gate8 Gemma weight binding requires explicit license/access attestation."
    }

    $wrapperSmoke = $env:GATE8_GEMMA_WEIGHT_BINDING_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 Gemma weight binding requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the current Gate8 Gemma weight-binding branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate8-gemma-weight-binding-execution-v0") {
        throw "Gate8 Gemma weight binding must run from agent/gate8-gemma-weight-binding-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve the exact Gate8 Gemma weight-binding Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 Gemma weight-binding output already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $bindingRoot = Join-Path $resolvedOutputRoot "binding"
    $resultPath = Join-Path $bindingRoot "gate8-gemma-weight-binding.json"
    $snapshotRoot = Join-Path $bindingRoot "model-snapshot"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate8-gemma-weight-binding-v0"
        scientific_protocol_head = "6bb89111a47713bea0a23bb1cae662ed5ec56b42"
        tokenizer_result_head = "c7f5260189ef9ac1a1beb73596446316631090c7"
        repo_id = "google/gemma-3-1b-it"
        revision = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
        required_files = @(
            "config.json",
            "generation_config.json",
            "model.safetensors"
        )
        qualified_config_sha256 = "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e"
        download_and_hash_only = $true
        model_instantiated = $false
        tokenizer_loaded = $false
        training_performed = $false
        inference_performed = $false
        scientific_test_worlds_generated = $false
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 GEMMA MODEL-FILE BINDING"
    Write-Host "============================================================"
    Write-Host "Git head:        $head"
    Write-Host "Output:          $resolvedOutputRoot"
    Write-Host "Repository:      google/gemma-3-1b-it"
    Write-Host "Revision:        dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    Write-Host "Download scope:  config + generation config + model.safetensors"
    Write-Host "Model loading:   FORBIDDEN"
    Write-Host "Tokenizer load:  FORBIDDEN"
    Write-Host "Inference:       NONE"
    Write-Host "Scientific test: NONE"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 Gemma weight-binding wrapper smoke completed before packages, network, model files, or benchmark data."
        return
    }

    $preflight = @'
import importlib.metadata
print(f"huggingface-hub={importlib.metadata.version('huggingface-hub')}")
'@
    try {
        $preflight | python -
    }
    catch {
        throw "Gate8 Gemma weight binding requires huggingface_hub."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 Gemma weight-binding package preflight failed."
    }

    python scripts/bind_gate8_gemma_weights.py --output-root $bindingRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 Gemma weight binding failed. Preserve the complete output root for diagnosis."
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Gate8 Gemma weight-binding result is missing: $resultPath"
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    if ($result.scientific_status -ne "GATE8_GEMMA_MODEL_FILE_BINDING_COMPLETE") {
        throw "Gate8 Gemma weight binding returned an unexpected status."
    }
    if ($result.model_binding.repo_id -ne "google/gemma-3-1b-it") {
        throw "Gate8 Gemma weight binding changed the repository identity."
    }
    if ($result.model_binding.revision -ne "dcc83ea841ab6100d6b47a070329e1ba4cf78752") {
        throw "Gate8 Gemma weight binding changed the frozen revision."
    }
    if ($result.model_binding.file_sha256.'config.json' -ne "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e") {
        throw "Gate8 Gemma config hash disagrees with the tokenizer binding."
    }
    if ($result.model_binding.model_file_binding_complete -ne $true) {
        throw "Gate8 Gemma model-file binding did not complete."
    }
    if ($result.model_binding.model_files_downloaded -ne $true) {
        throw "Gate8 Gemma required model files were not downloaded."
    }
    if ($result.model_binding.safetensors.parameter_count -lt 900000000 -or $result.model_binding.safetensors.parameter_count -gt 1100000000) {
        throw "Gate8 Gemma parameter count is outside the frozen 1B class."
    }
    $dtypeNames = @($result.model_binding.safetensors.dtype_parameter_counts.PSObject.Properties.Name)
    if ($dtypeNames.Count -ne 1 -or $dtypeNames[0] -ne "BF16") {
        throw "Gate8 Gemma model weights are not exclusively BF16."
    }
    if (
        $result.model_instantiated -ne $false -or
        $result.tokenizer_loaded -ne $false -or
        $result.training_performed -ne $false -or
        $result.inference_performed -ne $false -or
        $result.scientific_test_worlds_generated -ne $false
    ) {
        throw "Gate8 Gemma weight binding crossed a forbidden execution boundary."
    }

    $visibleSnapshotFiles = @(
        Get-ChildItem -LiteralPath $snapshotRoot -File -Recurse |
        Where-Object { $_.FullName -notmatch '[\\/]\.cache[\\/]' } |
        ForEach-Object {
            $_.FullName.Substring($snapshotRoot.Length).TrimStart("\").Replace("\", "/")
        } |
        Sort-Object
    )
    $expectedSnapshotFiles = @(
        "config.json",
        "generation_config.json",
        "model.safetensors"
    )
    if (($visibleSnapshotFiles -join "`n") -ne ($expectedSnapshotFiles -join "`n")) {
        $visibleSnapshotFiles | ForEach-Object { Write-Host $_ }
        throw "Gate8 Gemma snapshot file set changed."
    }

    $forbiddenFiles = @(
        Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse |
        Where-Object {
            $_.Name -match '(?i)(pytorch_model.*\.bin$|\.gguf$|\.pth$|\.pt$)'
        }
    )
    if ($forbiddenFiles.Count -ne 0) {
        $forbiddenFiles | ForEach-Object { Write-Host $_.FullName }
        throw "Gate8 Gemma weight-binding output contains a forbidden file."
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

    $resultHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash.ToLowerInvariant()
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
    $modelHash = $result.model_binding.file_sha256.'model.safetensors'
    $modelBytes = $result.model_binding.file_sizes.'model.safetensors'
    $tensorCount = $result.model_binding.safetensors.tensor_count
    $parameterCount = $result.model_binding.safetensors.parameter_count

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 Gemma model-file binding complete"
    Write-Host "============================================================"
    Write-Host "Model file binding:    True"
    Write-Host "Model instantiated:    False"
    Write-Host "Tokenizer loaded:      False"
    Write-Host "Inference performed:   False"
    Write-Host "Scientific test:       False"
    Write-Host "Tensor count:          $tensorCount"
    Write-Host "Parameter count:       $parameterCount"
    Write-Host "Model bytes:           $modelBytes"
    Write-Host "Model SHA256:          $modelHash"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
