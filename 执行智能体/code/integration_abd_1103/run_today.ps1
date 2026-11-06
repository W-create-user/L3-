param([string]$MonitorFile = '')
$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$project = Split-Path -Parent $package
Set-Location $project
$python = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$launchFolder = Join-Path $package 'launch_logs'
New-Item -ItemType Directory -Path $launchFolder -Force | Out-Null
$terminal = Join-Path $launchFolder ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
$extra = @()
if ($MonitorFile) {
    $extra = @('--monitor-file', $MonitorFile)
}
Write-Host '开始今天的 B 联调验证：实际 A 任务 → 候选动作 → 监管接口 → 真实 SMPL' -ForegroundColor Cyan
Write-Host '正在运行，请等待；完整输出和警告会写入日志。'
$oldPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $python -u -m integration_abd_1103.demo_actual_a @extra 2>&1 |
        ForEach-Object { $_.ToString() } | Out-File -LiteralPath $terminal -Encoding utf8
    $pythonExit = $LASTEXITCODE
}
finally { $ErrorActionPreference = $oldPreference }
if ($pythonExit -ne 0) {
    Get-Content -LiteralPath $terminal -Tail 15 | Out-Host
    throw "联调失败，请查看 $terminal"
}
$evidenceLine = Get-Content -LiteralPath $terminal -Encoding utf8 |
    Where-Object { $_.StartsWith('Evidence:') } | Select-Object -Last 1
if (-not $evidenceLine) { throw "未找到本次证据目录，请查看 $terminal" }
$evidence = $evidenceLine.Substring(9).Trim()
$summary = Get-Content -LiteralPath (Join-Path $evidence 'verification.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($summary.status -ne 'PASS') { throw '检查结果不是 PASS' }
Write-Host 'A：已调用压缩包内的真实 DecisionAgent 和 HTNPlanner。'
Write-Host ('SMPL：真实模型；放行后推进 ' + $summary.normal_step_delta + ' 步。')
Write-Host ('拒绝动作：推进 ' + $summary.rejected_step_delta + ' 步，原因已反馈到 A 的任务。')
if ($summary.actual_D_code_used) {
    Write-Host 'D：本次加载了你指定的 MonitoringAgent 源码。'
}
else {
    Write-Host 'D：尚未提供源码，本次仅验证与其描述一致的接口参考实现。' -ForegroundColor Yellow
}
Write-Host 'R-03 状态阈值规则和 C 调度模块未纳入本次验证。'
Write-Host ('本次结果：' + $summary.status) -ForegroundColor Green
Write-Host "原始证据：$evidence"
Write-Host "终端记录：$terminal"
