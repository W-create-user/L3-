$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$root = $PSScriptRoot
$python = $env:SMPL_PYTHON
if (-not $python) { $python = 'D:\SMPL_Project\member_b_1101\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Python missing. Set SMPL_PYTHON to the installed Python 3.11 interpreter; see README.' }
$env:SMPL_SOURCE_DIR = Join-Path $root 'code\vendor\smpl'
$monitor = Join-Path $root 'code\external_D\pr1_8c105883\source\agents\monitoring_agent.py'
$log = Join-Path $root 'evidence\delivery_demo_terminal.log'
Push-Location (Join-Path $root 'code')
try {
  $ErrorActionPreference = 'Continue'
  & $python -u -m integration_abd_1103.demo_actual_a --monitor-file $monitor 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  if ($exitCode -ne 0) { throw "Demo failed: exit $exitCode; inspect $log" }
  Write-Host 'PASS: actual A + actual D + B + real SMPL. Approved: one step; rejected: zero steps.'
  Write-Host "Log: $log"
} finally { Pop-Location }
