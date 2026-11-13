# Run from a downloaded/extracted repository snapshot. Does not run the simulator on Windows.
$ErrorActionPreference = 'Stop'
$destination = 'D:\大创\B_2026-11-13'
if (-not (Test-Path 'D:\')) { throw 'D drive is unavailable.' }
if (Test-Path $destination) { throw "Destination already exists. Preserve the previous archive: $destination" }
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $repoRoot 'deliveries\B_20261113\manifest.json'))) { throw 'Missing delivery manifest; extract the complete repository archive first.' }
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($name in @('deliveries\B_20261113','experiments','l3_system','patches','scripts','vendor','requirements.txt','requirements-lock.txt','README.md','main.py')) {
    if ($name -eq 'deliveries\B_20261113') {
        Copy-Item -LiteralPath (Join-Path $repoRoot $name) -Destination (Join-Path $destination '任务报告与运行记录') -Recurse
    } else {
        Copy-Item -LiteralPath (Join-Path $repoRoot $name) -Destination $destination -Recurse
    }
}
$reportDir = Join-Path $destination '任务报告与运行记录'
$manifest = Get-Content -LiteralPath (Join-Path $reportDir 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $relativePath = $entry.Name.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $actual = (Get-FileHash -LiteralPath (Join-Path $reportDir $relativePath) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "Delivery hash mismatch: $relativePath" }
}
Write-Host "Archived and verified: $destination"
