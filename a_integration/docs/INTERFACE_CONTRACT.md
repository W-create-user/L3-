# A/B/C/D 接口契约（基于 2026-11-06 提供源码）

## A → C：HTN 子任务注册

入口：`convert_subtasks_for_c(parent_task)`，输出每个子任务的 `task_id`、唯一 `subtask_id`、`task_type`、`priority`、`dependencies`、`status`、`constraints`、能力标签和完成条件。C 使用 `subtask_id` 作为内部唯一键。

## CNP：Bid / Assignment

Bid 必须包含 `task_id`、`subtask_id`、`agent_id`、`estimated_cost`、`capability_match`、`load`。评分为：

`0.50 × capability_match + 0.30 × 1/(1+estimated_cost) + 0.20 × (1-load)`。

Assignment 至少包含 `task_id`、`subtask_id`、`winner_id`、`score`、`assignment_status`。A 在派发前检查 `winner_id`。

## A/C → B：执行任务

B 的正式入口：

```python
executor.execute_assignment(task, safety_checker, action=..., subtask=...)
```

B 接收 A 原始 Task（含原始 HTN `subtasks`）或其约定传输结构；不是把 C Assignment 直接当成 B Task。A 用 `build_b_task()` 校验父/子任务 ID 并保留原始 HTN 内容。

B 的当前执行器 ID：`ExecutionAgent-01`。

## B → D：动作安全审核

D 初始化必须使用真实环境的 `adapter.action_space`：

```python
monitor = MonitoringAgent(action_space=adapter.action_space)
assert monitor.demo_mode is False
```

B 回调格式：`safety_checker(action_request, observation)`。A 的适配器内部调用 `monitor.check_request(action_request, observation)`。只有明确 `decision == "pass"` 才能执行；`reject`、异常、未知值、任务编号不匹配、演示模式均 fail closed。

D 返回字段：`task_id`、`decision`、`rule_id`、`reason`、`risk_level`。R-01/R-02 已启用；R-03 未启用。

## B → C：执行反馈

B 的 `ExecutionResult` 保留 `task_id`、`subtask_id`、`request_id`、`execution_status`、`success`、`executed`、观测、奖励、终止信息、步数、`safety_decision` 和失败原因。A 的 `execution_result_to_feedback()` 将 `rejected` 映射为 C 支持的 `failed`，安全拒绝时 `exclude_failed_agent=False`。

## 状态规则
- `completed` 必须来自真实处理器结果或真实 B 执行反馈。
- `rejected` 不等于 `completed`。
- 只读观测任务不得调用 `env.step()`。
- 本次控制执行成功的验收条件：B 返回 `completed`、`success=True`、`executed=True`，且 `step_calls` 差值恰为 1。
