# Save this downloaded public code snapshot only. Cloud-private logs are not included.
$ErrorActionPreference = 'Stop'
$destination = 'D:\大创\B_截至2026-11-20_代码'
if (-not (Test-Path 'D:\')) { throw 'D drive unavailable.' }
if (Test-Path $destination) { throw "Preserve existing directory: $destination" }
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($name in @('l3_system','experiments','data_generation','configs','docs','vendor','patches','scripts','requirements.txt','requirements-lock.txt','requirements-runtime-20261120.lock','README.md','main.py')) {
    $sourcePath = Join-Path $root $name
    if (Test-Path -LiteralPath $sourcePath) {
        Copy-Item -LiteralPath $sourcePath -Destination $destination -Recurse
    }
}
Write-Host "Code-only snapshot; private companion configurations may still be required for reproduction."
Write-Host "Code saved: $destination. Private cloud reports/logs were not copied."
