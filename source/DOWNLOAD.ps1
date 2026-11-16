param([switch]$Volumes)
$ErrorActionPreference = 'Stop'
$experiment = 'exp172_20261116_fullvolume_rms_mamba_bf16_200e'
$serverRoot = "/home/HPC_USER/experience/$experiment"
& scp -r "HPC_USER@HPC_HOST_REDACTED:${serverRoot}/analysis_return" $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Results are not ready or directory transfer failed.' }
& python -B "$PSScriptRoot/return_results.py" verify --folder "$PSScriptRoot/analysis_return"
if ($LASTEXITCODE -ne 0) { throw 'Downloaded result verification failed.' }
if ($Volumes) {
    $receipt = Get-Content -Raw "$PSScriptRoot/analysis_return/test_complete.json" | ConvertFrom-Json
    $selected = 'best_e{0:000}' -f [int]$receipt.best.epoch
    $target = Join-Path $PSScriptRoot 'downloaded_test'
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    & scp -r "HPC_USER@HPC_HOST_REDACTED:${serverRoot}/runs/test/$selected" $target
    if ($LASTEXITCODE -ne 0) { throw 'Full test-volume transfer failed.' }
}
