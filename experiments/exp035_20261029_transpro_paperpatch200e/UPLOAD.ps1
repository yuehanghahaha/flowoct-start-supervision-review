param(
    [string]$Destination = 'eias-member.eitech.edu.cn',
    [string]$RemoteExperience = '/docsystem/niuke/lvyuehang/Flow_oct2octa/experience'
)
$ErrorActionPreference = 'Stop'
if ($RemoteExperience -notmatch '^/[a-zA-Z0-9_./-]+$') { throw 'Use an absolute remote path without spaces or quotes.' }
& ssh -o StrictHostKeyChecking=yes $Destination "mkdir -p '$RemoteExperience'"
if ($LASTEXITCODE -ne 0) { throw 'SSH connection failed; no training was started.' }
& scp -r $PSScriptRoot "${Destination}:$RemoteExperience/"
if ($LASTEXITCODE -ne 0) { throw 'Directory upload incomplete; no training was started.' }
Write-Output 'Source directory synced. Preparation/check/training remain separate actions.'
