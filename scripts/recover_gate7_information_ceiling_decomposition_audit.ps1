param(
    [string]$OutputRoot = "F:\gate7_information_ceiling_decomposition_v0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-RequiredLeaf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required information-ceiling recovery file is missing: $Path"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $smoke = $env:GATE7_INFORMATION_CEILING_AUDIT_RECOVERY_SMOKE -eq "1"
    $status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read Git working-tree status."
    }
    if ($status.Count -ne 0) {
        $status | ForEach-Object { Write-Host $_ }
        throw "Information-ceiling audit recovery requires a clean Git working tree."
    }

    $branchOutput = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the current audit-recovery branch."
    }
    $branch = if ($null -eq $branchOutput) { "" } else { "$branchOutput".Trim() }
    if ($branch.Length -eq 0 -and $smoke -and $env:GITHUB_HEAD_REF) {
        $branch = $env:GITHUB_HEAD_REF
    }
    $expectedBranch = "agent/gate7-information-ceiling-decomposition-audit-recovery-v0"
    if ($branch -ne $expectedBranch) {
        throw "Audit recovery must run from $expectedBranch."
    }

    $recoveryHead = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $recoveryHead.Length -ne 40) {
        throw "Could not resolve exact audit-recovery Git head."
    }

    if ($smoke) {
        Write-Host "Information-ceiling audit-recovery wrapper smoke completed before artifact access."
        return
    }

    $resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
    $scienceRoot = Join-Path $resolvedOutputRoot "science"
    $resultPath = Join-Path $scienceRoot "gate7-information-ceiling-decomposition.json"
    $originalHeadPath = Join-Path $resolvedOutputRoot "git-head.txt"
    $invalidAuditPath = Join-Path $resolvedOutputRoot "information-ceiling-audit.json"
    $preservedInvalidAuditPath = Join-Path $resolvedOutputRoot "information-ceiling-audit.pre-recovery-invalid.json"
    $recoveredAuditPath = Join-Path $resolvedOutputRoot "information-ceiling-audit.json"
    $recoveryMetadataPath = Join-Path $resolvedOutputRoot "information-ceiling-audit-recovery.json"
    $manifestPath = Join-Path $resolvedOutputRoot "manifest.sha256"

    Assert-RequiredLeaf -Path $resultPath
    Assert-RequiredLeaf -Path $originalHeadPath
    $originalExecutionHead = (Get-Content -Raw -LiteralPath $originalHeadPath).Trim()
    if ($originalExecutionHead -ne "161142c1e5552cb9464216c774397def6a4100be") {
        throw "Preserved scientific artifact was not produced by the exact admitted execution head."
    }

    if (Test-Path -LiteralPath $preservedInvalidAuditPath) {
        throw "Preserved invalid audit already exists; refusing a second recovery mutation."
    }
    Assert-RequiredLeaf -Path $invalidAuditPath
    $invalidAuditHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $invalidAuditPath).Hash.ToLowerInvariant()
    Move-Item -LiteralPath $invalidAuditPath -Destination $preservedInvalidAuditPath

    $resultHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash.ToLowerInvariant()

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 INFORMATION-CEILING AUDIT RECOVERY"
    Write-Host "============================================================"
    Write-Host "Scientific execution head: $originalExecutionHead"
    Write-Host "Audit recovery head:       $recoveryHead"
    Write-Host "Result SHA256:              $resultHashBefore"
    Write-Host "Recovery action:            ranker-key order canonicalization only"
    Write-Host "Scientific execution:       NOT REPEATED"
    Write-Host ""

    python scripts/recover_gate7_information_ceiling_decomposition_audit.py `
        $resultPath `
        --output $recoveredAuditPath `
        --metadata-output $recoveryMetadataPath
    if ($LASTEXITCODE -ne 0) {
        throw "Recovered information-ceiling audit still rejected the artifact. Preserve all output."
    }

    Assert-RequiredLeaf -Path $recoveredAuditPath
    Assert-RequiredLeaf -Path $recoveryMetadataPath
    $resultHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash.ToLowerInvariant()
    if ($resultHashAfter -ne $resultHashBefore) {
        throw "Scientific result artifact changed during audit recovery."
    }

    $audit = Get-Content -Raw -LiteralPath $recoveredAuditPath | ConvertFrom-Json
    if (-not $audit.artifact_valid -or @($audit.errors).Count -ne 0) {
        throw "Recovered audit did not produce artifact_valid=true with errors=[]."
    }

    $metadata = Get-Content -Raw -LiteralPath $recoveryMetadataPath | ConvertFrom-Json
    $metadata | Add-Member -NotePropertyName original_execution_head -NotePropertyValue $originalExecutionHead
    $metadata | Add-Member -NotePropertyName recovery_git_head -NotePropertyValue $recoveryHead
    $metadata | Add-Member -NotePropertyName rejected_audit_sha256 -NotePropertyValue $invalidAuditHash
    $metadata | Add-Member -NotePropertyName rejected_audit_preserved_as -NotePropertyValue $preservedInvalidAuditPath
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $recoveryMetadataPath

    @(
        Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse |
        Where-Object { $_.FullName -ne $manifestPath } |
        ForEach-Object {
            $relative = $_.FullName.Substring($resolvedOutputRoot.Length).TrimStart("\").Replace("\", "/")
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            "$hash  $relative"
        } |
        Sort-Object
    ) | Set-Content -Encoding ASCII -LiteralPath $manifestPath

    $recoveredAuditHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $recoveredAuditPath).Hash
    $recoveryMetadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $recoveryMetadataPath).Hash
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Gate-7 information-ceiling audit recovered"
    Write-Host "============================================================"
    Write-Host "Artifact valid:       $($audit.artifact_valid)"
    Write-Host "Campaign outcome:     $($audit.campaign_outcome)"
    Write-Host "Errors:               $([string]::Join(', ', @($audit.errors)))"
    Write-Host "Scientific rerun:     False"
    Write-Host "Result SHA256:         $($resultHashAfter.ToUpperInvariant())"
    Write-Host "Rejected audit SHA256:$($invalidAuditHash.ToUpperInvariant())"
    Write-Host "Recovered audit SHA:  $recoveredAuditHash"
    Write-Host "Recovery record SHA:  $recoveryMetadataHash"
    Write-Host "Manifest SHA256:       $manifestHash"
    Write-Host "Output root:           $resolvedOutputRoot"
}
finally {
    Pop-Location
}
