# Save this downloaded public code snapshot only. Cloud-private logs are not included.
$ErrorActionPreference = 'Stop'
$destination = 'D:\大创\B_截至2026-11-16_代码'
if (-not (Test-Path 'D:\')) { throw 'D drive unavailable.' }
if (Test-Path $destination) { throw "Preserve existing directory: $destination" }
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($name in @('l3_system','experiments','configs','docs','vendor','patches','scripts','requirements.txt','requirements-lock.txt','README.md','main.py')) {
    Copy-Item -LiteralPath (Join-Path $root $name) -Destination $destination -Recurse
}
Write-Host "Code saved: $destination. Private cloud reports/logs were not copied."
