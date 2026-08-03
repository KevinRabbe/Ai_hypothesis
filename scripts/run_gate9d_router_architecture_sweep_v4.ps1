param([Parameter(Mandatory=$true)][string]$OutputRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
& python (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'run_gate9d_router_architecture_sweep_v4.py') --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Gate9D router architecture sweep failed with exit code $LASTEXITCODE" }
