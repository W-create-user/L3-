param([ValidateSet('normal','reject','inspect')][string]$Scenario='normal',[int]$Seed=1101,[int]$Cycles=1)
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
$python=$env:SMPL_PYTHON
if (-not $python) { $python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.txt, or set SMPL_PYTHON.' }
Push-Location $PSScriptRoot
try {
 & $python -u .\main.py --scenario $Scenario --seed $Seed --cycles $Cycles
 if ($LASTEXITCODE -ne 0) { throw "System failed: $LASTEXITCODE" }
} finally { Pop-Location }
