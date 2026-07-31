param(
    [Parameter(Mandatory = $true)]
    [string]$TokenizerSnapshot,

    [Parameter(Mandatory = $true)]
    [string]$ModelSnapshot,

    [string]$OutputRoot = "F:\gate8_v1_gemma_reference_v0",

    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $wrapperSmoke = $env:GATE8_V1_GEMMA_REFERENCE_WRAPPER_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Gate8 v1 Gemma reference requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve current branch." }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $wrapperSmoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    if ($branch -ne "agent/gate8-v1-gemma-reference-execution-v0") {
        throw "Gate8 v1 Gemma reference must run from agent/gate8-v1-gemma-reference-execution-v0."
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "Could not resolve exact Gate8 v1 Gemma-reference Git head."
    }

    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $resolvedOutputRoot -PathType Container)) {
            throw "Gate8 v1 Gemma reference resume output does not exist: $resolvedOutputRoot"
        }
    }
    elseif (Test-Path -LiteralPath $resolvedOutputRoot) {
        throw "Gate8 v1 Gemma-reference output already exists: $resolvedOutputRoot"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 GEMMA 3 1B REFERENCE EVALUATION"
    Write-Host "============================================================"
    Write-Host "Git head:         $head"
    Write-Host "Output:           $resolvedOutputRoot"
    Write-Host "Resume:           $($Resume.IsPresent)"
    Write-Host "Test split:       test"
    Write-Host "Test seed:        0"
    Write-Host "World indices:    0..511 per condition"
    Write-Host "Conditions:       21"
    Write-Host "Reference rows:   10,752"
    Write-Host "Decoding:         greedy, one beam, max 64 new tokens"
    Write-Host "Batch size:       1"
    Write-Host "Joint classifier: CLOSED in this phase"
    Write-Host ""

    if ($wrapperSmoke) {
        Write-Host "Gate8 v1 Gemma-reference wrapper smoke completed before package imports, snapshot access, output creation, test-world generation, model loading, or inference."
        return
    }

    $resolvedTokenizer = (Resolve-Path -LiteralPath $TokenizerSnapshot).Path
    $resolvedModel = (Resolve-Path -LiteralPath $ModelSnapshot).Path

    $tokenizerExpected = [ordered]@{
        "added_tokens.json" = "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946"
        "config.json" = "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e"
        "special_tokens_map.json" = "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
        "tokenizer.json" = "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
        "tokenizer.model" = "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"
        "tokenizer_config.json" = "bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1"
    }
    $modelExpected = [ordered]@{
        "config.json" = "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e"
        "generation_config.json" = "fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3"
        "model.safetensors" = "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6"
    }

    function Assert-ExactSnapshot {
        param(
            [string]$Root,
            [System.Collections.Specialized.OrderedDictionary]$Expected,
            [string]$Label
        )
        $visible = @(
            Get-ChildItem -LiteralPath $Root -File -Recurse |
            Where-Object { $_.FullName -notmatch '[\\/]\.cache[\\/]' } |
            ForEach-Object { $_.FullName.Substring($Root.Length).TrimStart("\").Replace("\", "/") } |
            Sort-Object
        )
        $wanted = @($Expected.Keys | Sort-Object)
        if (($visible -join "`n") -ne ($wanted -join "`n")) {
            throw "$Label snapshot file set drifted."
        }
        foreach ($name in $Expected.Keys) {
            $path = Join-Path $Root $name
            $observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
            if ($observed -ne $Expected[$name]) {
                throw "$Label snapshot SHA-256 mismatch: $name"
            }
        }
    }

    Assert-ExactSnapshot -Root $resolvedTokenizer -Expected $tokenizerExpected -Label "Tokenizer"
    Assert-ExactSnapshot -Root $resolvedModel -Expected $modelExpected -Label "Model"

    $preflight = @'
import importlib.metadata
import platform
import torch
expected = {
    "python": "3.11.9",
    "torch": "2.9.1+cu130",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "numpy": "2.3.5",
    "huggingface-hub": "0.36.2",
}
observed = {"python": platform.python_version()}
observed["torch"] = torch.__version__
for name in ("transformers", "tokenizers", "numpy", "huggingface-hub"):
    observed[name] = importlib.metadata.version(name)
for name, value in expected.items():
    if observed[name] != value:
        raise SystemExit(f"software drift: {name} expected={value} observed={observed[name]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA BF16 is unavailable")
print(observed)
print("cuda_device=" + torch.cuda.get_device_name(0))
print("cuda_capability=" + str(torch.cuda.get_device_capability(0)))
'@
    $preflight | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 Gemma-reference Python/CUDA preflight failed."
    }

    $arguments = @(
        "scripts/run_gate8_v1_gemma_reference.py",
        "--tokenizer-snapshot", $resolvedTokenizer,
        "--model-snapshot", $resolvedModel,
        "--output-root", $resolvedOutputRoot
    )
    if ($Resume) { $arguments += "--resume" }
    python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Gate8 v1 Gemma reference failed. Preserve the complete output root and resume only through this wrapper."
    }

    $summaryPath = Join-Path $resolvedOutputRoot "reference\gate8-v1-gemma-reference-summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Gate8 v1 Gemma-reference summary is missing."
    }
    $summary = Get-Content -Raw $summaryPath | ConvertFrom-Json
    if ($summary.scientific_status -ne "G8_V1_GEMMA_REFERENCE_EVALUATION_COMPLETE") {
        throw "Gate8 v1 Gemma-reference scientific status is invalid."
    }
    if ($summary.reference_model_loaded -ne $true -or $summary.reference_inference_performed -ne $true) {
        throw "Gate8 v1 Gemma-reference inference did not complete."
    }
    if ($summary.training_performed -ne $false -or $summary.population_execution_performed -ne $false) {
        throw "Gate8 v1 Gemma-reference phase crossed a closed execution boundary."
    }
    if ($summary.joint_reference_comparison_classified -ne $false) {
        throw "Gate8 v1 Gemma-reference phase ran the closed joint classifier."
    }

    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"
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

    $perWorldPath = Join-Path $resolvedOutputRoot "reference\gate8-v1-gemma-reference-per-world.jsonl"
    $promptIndexPath = Join-Path $resolvedOutputRoot "reference\gate8-v1-gemma-reference-prompt-index.jsonl"
    $summaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $summaryPath).Hash
    $perWorldHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $perWorldPath).Hash
    $promptIndexHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $promptIndexPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-8 v1 Gemma reference complete"
    Write-Host "============================================================"
    Write-Host "Status:              $($summary.scientific_status)"
    Write-Host "Reference accuracy:  $($summary.pooled_reference_accuracy)"
    Write-Host "Valid parse rate:    $($summary.valid_parse_rate)"
    Write-Host "Summary SHA256:      $summaryHash"
    Write-Host "Per-world SHA256:    $perWorldHash"
    Write-Host "Prompt-index SHA256: $promptIndexHash"
    Write-Host "Manifest SHA256:     $manifestHash"
    Write-Host "Output root:         $resolvedOutputRoot"
}
finally {
    Pop-Location
}
