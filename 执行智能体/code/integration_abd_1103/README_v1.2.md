# B 侧 A Task 兼容接口 v1.2

日期：2026 年 11 月 3 日

本版按 A 最新提供的 Task 字典实现。A 只提供任务与子任务计划，无需提供 action，也无需在 Task 内指定 action_policy。B 对 control_cycle 默认采用本地 space_midpoint 接口基线生成候选，候选与执行分离；D 放行后才调用真实 SMPL。

中点基线读取当前 observation 检查数值有效性，再从真实 action_space 计算动作中点。它不根据状态偏差优化输入，不是生产控制算法。更复杂的反馈策略后续在 B 的 local_policy.py 中增加，A 的任务协议不必因此承担控制计算。

## 一 两种输入都接受

1. A 原始字典：task_id、task_type、description、priority、preconditions、dependencies、constraints、completion_criteria、subtasks、status、source。
2. 三字段格式：task_id、task_name、parameters；parameters 携带上述任务类型、优先级、条件、约束、完成标准和完整子任务列表。

原始字典可直接传入，不需要 A 额外复制顶层 subtasks。转换格式的 parameters.subtasks 同样可以识别，并保留每个原始 subtask_id。

```python
from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent, normalize_task

with EnvironmentAdapter() as adapter:
    executor = ExecutionAgent(adapter, log_path="data/B_events.jsonl")
    executor.reset(seed=1101)

    request = executor.prepare_action(a_task)  # A 不传动作或策略
    # 也可以：executor.prepare_action(normalize_task(a_task))
    # 此时 env.step 尚未被调用。

    # monitor 为 D 提供的实际对象，应由真实 action_space 构造。
    # raw_review = monitor.check_action(request["observation"], request["action"])
    # if raw_review["approved"]:
    #     result = executor.execute_action(request, raw_review)
    # else:
    #     result = executor.reject_action_request(request, raw_review)
```

示例中的 a_task 和 monitor 由主程序提供，不是未定义对象也能直接运行的完整脚本；可运行入口见第三节。

## 二 字段与行为

- task_id 原样保留，绝不重新生成总任务编号。
- task_name 使用 task_name，或从 A.description 取得。
- task_type 从原始 task_type、旧 goal_type 或 parameters.task_type 识别；多处矛盾时拒绝，不静默覆盖。
- priority、preconditions、dependencies、constraints、completion_criteria、status、source、subtasks 完整保留在 request.task_context.parameters 和日志中。
- 默认关联唯一的 execute_approved_action 子任务。若存在多个执行子任务，需要调用方明确指定 subtask。
- request_id 是每次候选的独立编号；state_version 防止旧观测请求被再次执行。
- prepare_action 只读取缓存观测、计算候选并保存日志，不 reset、不 step；原始 A 字典不会被修改。
- get_observation 返回最近一次 reset/step 的观测副本，不计算或推进模型。
- execute_action 兼容 D 的 approved 布尔格式及旧版 request_id + decision + reason 结构；审查结果单独传入，不修改 ActionRequest。
- 执行反馈保留 task_id、parent_task_id、subtask_id、request_id、execution_status、success、executed、observation、action、reward、done、info、step_count、state_version、failure_reason、safety_decision。

ActionRequest 中的 task_context 用于保留任务语义，不代表 B 或 D 已实现所有生产约束校验。当前 D 描述的 R-03 仍未实现，不编造工艺阈值。

## 三 在 B 电脑运行

```powershell
cd D:\SMPL_Project\member_b_1101
powershell -NoProfile -ExecutionPolicy Bypass -File .\integration_abd_1103\run_current_task.ps1
```

默认读取 examples/a_task_current.json，该文件逐字段保存用户这次提供的完整 Task，包括 TASK-5b95693d 和六个指定子任务 ID。

可用 -TaskFile 指定 A 导出的实际 JSON 文件。该验证入口按给定六步控制模板运行，包含正常与拒绝测试；如父任务有外部依赖，必须先由主程序解决，不能当作 C 的调度器。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\integration_abd_1103\run_current_task.ps1 -TaskFile "D:\A输出\task.json"
```

拿到 D 源码后加 -MonitorFile "D:\实际路径\monitoring_agent.py"。如果没有提供该文件，默认只运行 B 明确标注的 D 接口参考实现，绝不报告 D 原代码已通过。

## 四 新旧前置条件和完成标准

主程序参考入口 run_task_dict 兼容 observation_received、goal_evaluated、action_intent_created、safety_decision_received、safety_passed、execution_result_received、result_verified，也保留旧版 observation_available、candidate_action_available、monitoring_passed、execution_feedback_available 的语义。

每个子任务要满足实际前置条件才能开始，其完成标准满足后才标为 completed。只有全部子任务均完成，才满足 all_subtasks_completed。监管拒绝时，后续子任务 blocked，父任务完成标准不满足。

未知的生产质量、产量等完成标准不会被自动判为通过。父任务依赖、前置条件需由 A/C 提供已满足信息；参考入口未获这些信息时拒绝执行。约束数值和生产目标评价仍需后续实现。

## 五 验证范围

34 项单元与接口测试通过，包含旧协议回归和本次新 Task。计数替身只用于函数契约测试，不当作模型证据。

真实 SMPL 验证：原始 Task 和三字段 Task 在同样初始观测下生成相同候选，各放行执行一次；拒绝分支执行零次。总任务 ID 和执行子任务 ID 保留，原始输入没有被修改，Task 中没有 action/action_policy。

logs_v12 下保存 A_Task_original.json、B_Task_mapped.json、两种 ActionRequest、各分支结果与事件、manifest.json、B_events.jsonl 和 verification.json。任务源注明为用户提供的 Task 字典，不声称运行了尚未提供的最新版 DecisionAgent 源码。旧版 A 源码回归另由已有测试覆盖。

## 六 模型和依赖

依旧使用 Python 3.11、固定官方 SMPL 源码及 requirements-lock.txt。默认源码位置为项目根目录 vendor/smpl，也可用 SMPL_SOURCE_DIR 指定。当前 normalize=False、uss_subtracted=True，不使用 [-1,1] 演示边界。

交付包不含本机虚拟环境和完整官方源码。移机按 README 配置解释器、依赖和相同模型版本。B 旧 v1.1 交付包和当前版本升级前源码快照保留，便于回退。
