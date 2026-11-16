param(
    [string]$Destination = 'lvyuehang@10.10.101.35',
    [string]$RemoteExperience = '/home/lvyuehang/experience'
)
$ErrorActionPreference = 'Stop'
if ($RemoteExperience -notmatch '^/[a-zA-Z0-9_./-]+$') { throw 'Use an absolute remote directory without spaces or quotes.' }
& ssh -o StrictHostKeyChecking=yes $Destination "mkdir -p '$RemoteExperience'"
if ($LASTEXITCODE -ne 0) { throw 'Connection failed. Nothing was started.' }
& scp -r $PSScriptRoot "${Destination}:$RemoteExperience/"
if ($LASTEXITCODE -ne 0) { throw 'Directory upload incomplete. No training was started.' }
Write-Output 'Directory upload finished. Prepare/check/train are separate server commands.'
