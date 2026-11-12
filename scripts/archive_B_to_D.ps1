# Run with PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\archive_B_to_D.ps1
$ErrorActionPreference = 'Stop'
$destination = 'D:\大创\B_11.11-11.12_2026'
if (-not (Test-Path 'D:\')) { throw 'D drive is unavailable.' }
if (Test-Path $destination) { throw "Destination already exists; preserve it: $destination" }
New-Item -ItemType Directory -Path $destination | Out-Null
$repoRoot = Split-Path $PSScriptRoot -Parent
# Copy this downloaded repository snapshot, excluding local virtual environments and scratch files.
foreach ($name in @('docs','experiments','l3_system','vendor','patches','scripts','requirements.txt','requirements-lock.txt','README.md','main.py')) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $name) -Destination $destination -Recurse
}
Get-ChildItem $destination -Recurse -File | ForEach-Object {
    [PSCustomObject]@{ Path = $_.FullName.Substring($destination.Length + 1); SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
} | Export-Csv (Join-Path $destination '本机归档校验.csv') -NoTypeInformation -Encoding UTF8
Write-Host "Archived to $destination"
