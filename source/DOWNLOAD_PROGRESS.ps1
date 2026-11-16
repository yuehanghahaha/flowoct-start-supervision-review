$ErrorActionPreference = 'Stop'
$serverRoot = '/home/HPC_USER/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e'
$destination = Join-Path $PSScriptRoot 'progress_return'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($relative in @('runs/status.json', 'runs/a100', 'runs/train', 'runs/validation', 'runs/normalization.json', 'runs/execution.json', 'verification')) {
    $target = Join-Path $destination (Split-Path $relative -Leaf)
    & scp -r "HPC_USER@HPC_HOST_REDACTED:${serverRoot}/$relative" $target
    if ($LASTEXITCODE -ne 0) { Write-Warning "Not available yet: $relative" }
}
# Intermediate validation 3D arrays are intentionally absent.
