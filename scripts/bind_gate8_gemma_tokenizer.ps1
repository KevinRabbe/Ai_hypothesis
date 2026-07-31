param(
    [Parameter(Mandatory = $true)]
    [switch]$HuggingFaceLicenseAndAccessAttested,

    [string]$OutputRoot = "F:\gate8_gemma_tokenizer_binding_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $HuggingFaceLicenseAndAccessAttested) {
        throw "Gate8 tokenizer binding requires explicit Gemma license/access attestation."
    }

    $wrapperSmoke = $env:GATE8_GEMMA_TOKENIZER_BINDING_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 tokenizer binding requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current Gate8 tokenizer branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate8-gemma-tokenizer-binding-execution-v0") {
        throw "Gate8 tokenizer binding must run from agent/gate8-gemma-tokenizer-binding-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 tokenizer-binding Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 tokenizer binding output already exists: $resolvedOutputRoot"
    }
    New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null

    $bindingRoot = Join-Path $resolvedOutputRoot "binding"
    $resultPath = Join-Path $bindingRoot "gate8-gemma-tokenizer-binding.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    $head | Set-Content -Encoding ASCII (Join-Path $resolvedOutputRoot "git-head.txt")
    (git status --porcelain) | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "git-status.txt")

    $runConfig = [ordered]@{
        protocol = "gate8-gemma-tokenizer-binding-v0"
        encoder_head = "9882256ae0152bc266dc4d96cab3bbeb0c4ef95b"
        repo_id = "google/gemma-3-1b-it"
        revision = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
        tokenizer_files_only = $true
        model_binding_performed = $false
        model_weights_downloaded = $false
        training_performed = $false
        inference_performed = $false
        scientific_test_worlds_generated = $false
        git_head = $head
    }
    $runConfig | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $resolvedOutputRoot "run-config.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 GEMMA TOKENIZER BINDING"
    Write-Host "============================================================"
    Write-Host "Git head:        $head"
    Write-Host "Output:          $resolvedOutputRoot"
    Write-Host "Repository:      google/gemma-3-1b-it"
    Write-Host "Revision:        dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    Write-Host "Download scope:  tokenizer/config files only"
    Write-Host "Model weights:   FORBIDDEN"
    Write-Host "Inference:       NONE"
    Write-Host "Scientific test: NONE"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 tokenizer-binding wrapper smoke completed before packages, network, tokenizer files, or prompts."
        return
    }

    $preflight = @'
import importlib.metadata
required = ("transformers", "tokenizers", "huggingface-hub")
for package in required:
    print(f"{package}={importlib.metadata.version(package)}")
'@
    try {
        $preflight | python -
    }
    catch {
        throw "Gate8 tokenizer packages are missing. Install transformers, tokenizers, and huggingface_hub before retrying."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 tokenizer package preflight failed."
    }

    python scripts/bind_gate8_gemma_tokenizer.py --output-root $bindingRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 Gemma tokenizer binding failed. Preserve the complete output root for diagnosis."
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Gate8 tokenizer binding result is missing: $resultPath"
    }

    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
    if ($result.tokenizer_binding.tokenizer_bound -ne $true) {
        throw "Gate8 tokenizer binding did not establish tokenizer_bound=true."
    }
    if ($result.tokenizer_binding.model_bound -ne $false) {
        throw "Gate8 tokenizer binding unexpectedly opened model binding."
    }
    if ($result.model_weights_downloaded -ne $false -or $result.inference_performed -ne $false) {
        throw "Gate8 tokenizer binding crossed the forbidden model/inference boundary."
    }
    if ($result.scientific_test_worlds_generated -ne $false) {
        throw "Gate8 tokenizer binding generated scientific-test worlds."
    }

    $forbiddenFiles = @(
        Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse |
        Where-Object {
            $_.Name -match '(?i)(model.*\.safetensors$|pytorch_model.*\.bin$|\.gguf$|\.pth$|\.pt$)'
        }
    )
    if ($forbiddenFiles.Count -ne 0) {
        $forbiddenFiles | ForEach-Object { Write-Host $_.FullName }
        throw "Gate8 tokenizer output contains forbidden model-weight files."
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

    $resultHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash
    $maximum = $result.tokenizer_binding.maximum_condition

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 Gemma tokenizer binding complete"
    Write-Host "============================================================"
    Write-Host "Tokenizer bound:      True"
    Write-Host "Model bound:          False"
    Write-Host "Model weights:        False"
    Write-Host "Inference performed:  False"
    Write-Host "Maximum condition:    P=$($maximum.population) D=$($maximum.depth)"
    Write-Host "Maximum input tokens: $($maximum.input_tokens) / 24576"
    Write-Host "Result SHA256:         $resultHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
