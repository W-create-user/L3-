param([string]$TaskFile = '', [string]$MonitorFile = '')
$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$project = Split-Path -Parent $package
Set-Location $project
$python = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$logs = Join-Path $package 'launch_logs_v12'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$terminal = Join-Path $logs ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
$extra = @()
if ($TaskFile) { $extra += @('--task-file', $TaskFile) }
if ($MonitorFile) { $extra += @('--monitor-file', $MonitorFile) }
Write-Host 'B v1.2：接收 A 最新 Task，在本地生成动作，再审查和执行。' -ForegroundColor Cyan
Write-Host '正在验证原始格式、三字段格式及拒绝分支，请等待。'
$oldPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $python -u -m integration_abd_1103.demo_current_task @extra 2>&1 |
        ForEach-Object { $_.ToString() } | Out-File -LiteralPath $terminal -Encoding utf8
    $pythonExit = $LASTEXITCODE
}
finally { $ErrorActionPreference = $oldPreference }
if ($pythonExit -ne 0) {
    Get-Content -LiteralPath $terminal -Tail 15 | Out-Host
    throw "Task 联调失败，请查看 $terminal"
}
$line = Get-Content -LiteralPath $terminal -Encoding utf8 |
    Where-Object { $_.StartsWith('Evidence:') } | Select-Object -Last 1
if (-not $line) { throw '没有找到本次证据目录' }
$evidence = $line.Substring(9).Trim()
$summary = Get-Content -LiteralPath "$evidence/verification.json" -Raw -Encoding utf8 | ConvertFrom-Json
if ($summary.status -ne 'PASS') { throw '验证结果不是 PASS' }
Write-Host ('保留的 A 任务编号：' + $summary.task_id)
Write-Host 'A 无需提供 action 或 action_policy；B 使用本地中点接口基线。'
Write-Host ('原始 Task：推进 ' + $summary.raw_step_delta + ' 步；三字段 Task：推进 ' + $summary.mapped_step_delta + ' 步。')
Write-Host ('拒绝分支：推进 ' + $summary.reject_step_delta + ' 步，任务未标为全部完成。')
if ($summary.actual_D_code_used) {
    Write-Host 'D：已加载你指定的源文件。'
}
else {
    Write-Host 'D：仍是接口参考实现，尚非 D 原始源文件测试。' -ForegroundColor Yellow
}
Write-Host '本次输入是用户提供的 Task 字典；没有取得最新 DecisionAgent 生成器源码。'
Write-Host ('结果：' + $summary.status) -ForegroundColor Green
Write-Host "原始证据：$evidence"
Write-Host "终端记录：$terminal"
