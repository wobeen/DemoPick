# E0 한 번에 실행 (Windows PowerShell)
# 사용: .\run_e0.ps1 C:\data\robomimic\lift_mh_low_dim_v15.hdf5
param([Parameter(Mandatory=$true)][string]$Hdf5)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv run python scripts/01_inspect_hdf5.py --hdf5 $Hdf5
if ($LASTEXITCODE -ne 0) { exit 1 }
uv run python scripts/10_extract_features.py --hdf5 $Hdf5
if ($LASTEXITCODE -ne 0) { exit 1 }
uv run python scripts/11_e0_skill_vs_heuristic.py
