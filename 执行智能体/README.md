# 执行智能体（成员 B）

本目录由仓库根目录的 `执行智能体.zip` 展开，便于直接浏览和协作；对应该压缩包版本，后续接口更新需另行合并。

日期：2026-11-06。用途：给 A/C/D 进行主程序集成、接口联调与后续代码维护。包含全部相关 Python 代码、测试、数据采集与重复实验工具，以及可直接加载的固定 SMPL 模型源文件；不包含虚拟环境、大体量实验数据、缓存和无关报告。

## 先看哪些代码

| 路径（相对本包） | 用途 |
|---|---|
| code/integration_abd_1103/execution_agent.py | B 执行智能体主体 |
| code/integration_abd_1103/environment_adapter.py | SMPL 初始化、观测、真实 step 与终止标志适配 |
| code/integration_abd_1103/local_policy.py | 当前动作基线策略；后续反馈策略扩展位置 |
| code/integration_abd_1103/task_contract.py | A 原始 Task 与三字段格式兼容 |
| code/integration_abd_1103/safety_bridge.py | D 审核结果格式适配 |
| code/integration_abd_1103/a_integration.py | A 任务计划与 B 执行结果的对接 |
| code/integration_abd_1103/demo_actual_a.py | 实际 A 代码 + 实际 D 代码 + SMPL 联调入口 |
| code/integration_abd_1103/demo_current_task.py / examples | A 最新提供 Task 字典示例及验证入口 |
| code/integration_abd_1103/test_*.py | 接口、任务映射、安全门控测试 |
| code/data_collection_1104、data_collection_1105 | 逐步记录、连续采集、终止语义及数据校验代码 |
| code/midterm_checks | 59 项检查入口、固定条件重复实验代码 |
| code/vendor/smpl | 固定 SMPL 运行源码、模型配置、许可证、版本及哈希 |
| code/external_D | D 固定版本实际源码与来源；为联调快照，不代表 D 最新分支 |
| code/integration_abd_1103/member_A_11_02 | A 实际提供代码快照；为联调依赖 |

## 给 A / C 的接入说明

在主项目中保留 integration_abd_1103 包名，或在统一迁移时同步修改 import。执行前先 reset。Task 原 task_id 和执行 subtask_id 保留；可直接传 A 原始字典。constraints/subtasks 被携带和记录，不代表 B 已实现全部约束的生产语义。

```python
from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent

# a_task 来自 A；MonitoringAgent 来自小组 D 模块。
with EnvironmentAdapter() as env:
    agent = ExecutionAgent(env, log_path="logs/B_events.jsonl")
    agent.reset(seed=1101)
    monitor = MonitoringAgent(action_space=env.action_space)
    request = agent.prepare_action(a_task)  # 只准备，不执行
    review = monitor.check_action(request["observation"], request["action"])
    if review["approved"]:
        feedback = agent.execute_action(request, review)
    else:
        feedback = agent.reject_action_request(request, review)
```

此片段由小组主程序提供 a_task 和 MonitoringAgent，不是无需上下文的独立脚本。完整可运行入口为一键联调演示.ps1。支持 D 的 check_request/decision 格式的完整流程见 a_integration.py；请保持审核和执行同步关联，不能跨请求复用批准结果。

## 小组必须保持的接口约定

1. get_observation() 读取观测，不推进模型。
2. prepare_action(task) 返回独立 ActionRequest，不调用 env.step()。
3. 候选动作交 D 审核；D 必须由真实 env.action_space 构造。
4. 通过后 execute_action(request, review) 才推进模型。拒绝时调用 reject_action_request，不继续 step。
5. request_id、state_version、task_id 和候选内容需匹配；不能把旧批准用于新请求。
6. 反馈携带任务关联、执行状态、action、observation、reward、终止标志、info 和安全决定，供 A/C 决定下一步。

## B 电脑直接演示

终端进入本文件夹：

```powershell
powershell -ExecutionPolicy Bypass -File .\一键联调演示.ps1
```

本机默认复用 D:/SMPL_Project/member_b_1101/.venv/Scripts/python.exe。脚本强制加载本包的模型与实际 D 源码。预期末尾 PASS，正常执行 1 步、拒绝执行 0 步。Gym/CasADi 旧接口提示保留在日志，成功以退出码和 PASS 为准。

## 队友电脑安装

安装 Python 3.11 后，终端进入本文件夹：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\code\requirements-lock.txt
$env:SMPL_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
powershell -ExecutionPolicy Bypass -File .\一键联调演示.ps1
cd .\code
& $env:SMPL_PYTHON -m midterm_checks.final_tests
```

依赖安装需要网络；无须 Git 即可使用本包模型。换电脑仍需实际运行检查。环境目录不要上传 GitHub。

## GitHub 协作建议

由 A/C 主程序集成时，按上表挑选 B 包、依赖模型和测试目录，保持包结构；A/D 快照用来复现本次联调，接入小组正式模块后统一选择一个权威来源，避免重复维护。当前 import 路径以本包 code 为工作目录。新增策略放 local_policy.py；修改接口同步更新 README 和 test_*.py。提交源代码、依赖锁定和必要配置，实验输出放约定的数据位置。本包未代替你推送到 GitHub。

## 当前验证与限制

核心六个接口文件与已验证中期版本逐文件哈希一致。证据目录保留 59 项测试全通过、真实 A/D/B/SMPL 正常 1 步及拒绝 0 步的记录。本次协作包另在独立目录运行测试和演示。

当前策略为接口基线，不是生产优化控制。C SchedulingAgent/CNP 尚未在本包真实接入。固定 D 版本仅验证 R-01/R-02，未验证 R-03。reference_monitor.py 是 B 的接口参考实现；正式演示显式加载 external_D，不能将参考实现说成 D 的实际代码。

900 步原始数据在此前中期完整交付包中；本包集中提供代码。SMPL 源码许可证和各方来源清单随包保留，文件校验见 SHA256SUMS.csv。
