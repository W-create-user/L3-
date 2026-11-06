param([ValidateSet('normal','reject','inspect')][string]$Scenario='normal',[int]$Seed=1101)
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$python=$env:SMPL_PYTHON
if (-not $python) { $python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { $python='D:\SMPL_Project\member_b_1101\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Install Python 3.11 dependencies first, or set SMPL_PYTHON.' }
$env:SMPL_SOURCE_DIR=Join-Path $PSScriptRoot '执行智能体\code\vendor\smpl'
Push-Location $PSScriptRoot
try {
 $ErrorActionPreference='Continue'
 & $python -u .\main.py --scenario $Scenario --seed $Seed
 $runExit=$LASTEXITCODE
 $ErrorActionPreference='Stop'
 if ($runExit -ne 0) { throw "System failed: $runExit" }
} finally { Pop-Location }
