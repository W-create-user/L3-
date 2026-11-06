# 成员 A：11 月 2 日阶段代码

## 本次实现范围
- DecisionAgent：目标识别、输入校验、任务对象生成。
- HTNPlanner：有限任务模板、子任务分解、依赖关系生成与检查。
- main.py：打印任务规划结果。
- tests：验证任务模板、依赖关系、输入校验和 JSON 输出。

## 目录
```text
member_A_11_02/
├── agents/
│   ├── __init__.py
│   └── decision_agent.py
├── algorithms/
│   ├── __init__.py
│   └── htn_planner.py
├── tests/
│   └── test_decision_agent.py
├── main.py
└── README.md
```

## 运行
在 `member_A_11_02` 目录下运行：

```bash
python main.py
python -m unittest discover -s tests -v
```

也可以运行字符串目标：

```bash
python main.py --goal "检查当前仿真状态"
```

## 输出边界
当前 main.py 只生成结构化任务计划。它没有连接 SMPL、SchedulingAgent、
MonitoringAgent 或 ExecutionAgent，也没有调用环境 `step()`。这部分需要后续联调，
不能将本次输出表述为四 Agent 端到端运行成果。

## 与 C 对接前需确认
本版任务字段包括：
- 总任务：`task_id`, `task_name`, `goal_type`, `description`, `priority`,
  `preconditions`, `constraints`, `subtasks`, `status`, `source`
- 子任务：`subtask_id`, `name`, `description`, `order`, `dependencies`,
  `required_capability`, `preconditions`, `status`

请 A、C 在联调时确认是否保留这些字段名，尤其是 `task_id`、`priority`、
`dependencies`、`status`。不要在未沟通的情况下自行改字段。
