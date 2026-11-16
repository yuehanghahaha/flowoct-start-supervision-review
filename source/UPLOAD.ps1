$ErrorActionPreference = 'Stop'
$remote = '/home/HPC_USER/experience/exp172_20261116_fullvolume_rms_mamba_bf16_200e'
# Create/inspect the owned destination and mapping before this command.
# This uploads the frozen sources listed in sources.sha256; it does not start training.
Get-Content "$PSScriptRoot/sources.sha256" | ForEach-Object {
    $relative = ($_ -split '  ', 2)[1]
    & scp (Join-Path $PSScriptRoot $relative) "HPC_USER@HPC_HOST_REDACTED:${remote}/$relative"
    if ($LASTEXITCODE -ne 0) { throw "Transfer failed: $relative" }
}
