"""DecisionAgent MVP for the L3 multi-agent prototype.

Scope: deterministic goal mapping and finite task templates. This module does
not call SMPL, execute actions, or perform safety approval.
"""
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class SubTask:
    subtask_id: str
    name: str
    description: str
    order: int
    dependencies: list[str] = field(default_factory=list)
    required_capability: str = "decision"
    preconditions: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class Task:
    # Minimum shared fields agreed in the project implementation plan.
    task_id: str
    task_type: str
    priority: int
    preconditions: list[str]
    dependencies: list[str]
    status: str
    completion_criteria: list[str]
    # Additional fields retained for useful descriptions and routing.
    description: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    subtasks: list[SubTask] = field(default_factory=list)
    source: str = "rule_mapping"


class DecisionAgent:
    """Map supported production goals to finite, ordered task plans."""

    def __init__(self) -> None:
        self.goal_mapping = {
            "检查当前仿真状态": "inspect_state",
            "检查状态": "inspect_state",
            "执行一次仿真控制任务": "control_cycle",
            "执行一次控制任务": "control_cycle",
        }
        # Explicit finite templates. Dependencies refer to earlier step indexes.
        self.templates = {
            "inspect_state": [
                {"name": "collect_observation", "description": "获取当前仿真环境观测", "capability": "state_read", "depends_on": (), "preconditions": [], "criteria": ["observation_received"]},
                {"name": "evaluate_state", "description": "依据任务要求分析当前状态", "capability": "decision", "depends_on": (0,), "preconditions": ["observation_received"], "criteria": ["state_evaluated"]},
                {"name": "return_state_result", "description": "整理并返回状态检查结果", "capability": "decision", "depends_on": (1,), "preconditions": ["state_evaluated"], "criteria": ["result_returned"]},
            ],
            "control_cycle": [
                {"name": "collect_observation", "description": "获取当前仿真环境观测", "capability": "state_read", "depends_on": (), "preconditions": [], "criteria": ["observation_received"]},
                {"name": "evaluate_goal", "description": "根据任务目标分析当前状态", "capability": "decision", "depends_on": (0,), "preconditions": ["observation_received"], "criteria": ["goal_evaluated"]},
                {"name": "generate_action_intent", "description": "生成待审核的抽象控制动作意图", "capability": "decision", "depends_on": (1,), "preconditions": ["goal_evaluated"], "criteria": ["action_intent_created"]},
                {"name": "safety_review", "description": "将候选动作提交监管模块检查", "capability": "monitoring", "depends_on": (2,), "preconditions": ["action_intent_created"], "criteria": ["safety_decision_received"]},
                {"name": "execute_approved_action", "description": "仅在监管放行后请求执行模块执行动作", "capability": "environment_execution", "depends_on": (3,), "preconditions": ["safety_passed"], "criteria": ["execution_result_received"]},
                {"name": "verify_result", "description": "检查执行反馈与任务完成情况", "capability": "decision", "depends_on": (4,), "preconditions": ["execution_result_received"], "criteria": ["result_verified"]},
            ],
        }

    def recognize_goal(self, goal: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(goal, str):
            text = goal.strip()
            if not text:
                raise ValueError("目标字符串不能为空")
            if text not in self.goal_mapping:
                raise ValueError(f"暂不支持该目标：{text}")
            return {"task_type": self.goal_mapping[text], "description": text,
                    "priority": 3, "preconditions": [], "dependencies": [],
                    "completion_criteria": ["all_subtasks_completed"], "constraints": {}}

        if not isinstance(goal, dict):
            raise TypeError("输入必须是字符串或字典")
        task_type = goal.get("task_type", goal.get("goal_type"))
        if task_type not in self.templates:
            raise ValueError(f"不支持的任务类型：{task_type}")
        description = goal.get("description", task_type)
        priority = goal.get("priority", 3)
        preconditions = goal.get("preconditions", [])
        dependencies = goal.get("dependencies", [])
        completion_criteria = goal.get("completion_criteria", ["all_subtasks_completed"])
        constraints = goal.get("constraints", {})
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description 必须是非空字符串")
        if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
            raise ValueError("priority 必须是 1 到 5 之间的整数")
        for field_name, value in (("preconditions", preconditions), ("dependencies", dependencies), ("completion_criteria", completion_criteria)):
            if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
                raise ValueError(f"{field_name} 必须是非空字符串组成的列表（允许空列表）")
        if not isinstance(constraints, dict):
            raise ValueError("constraints 必须是字典")
        return {"task_type": task_type, "description": description.strip(), "priority": priority,
                "preconditions": [x.strip() for x in preconditions],
                "dependencies": [x.strip() for x in dependencies],
                "completion_criteria": [x.strip() for x in completion_criteria],
                "constraints": constraints.copy()}

    def decompose_task(self, task_type: str) -> list[SubTask]:
        if task_type not in self.templates:
            raise ValueError(f"没有对应的任务模板：{task_type}")
        steps = self.templates[task_type]
        ids = [f"SUB-{uuid4().hex[:8]}" for _ in steps]
        result: list[SubTask] = []
        for index, step in enumerate(steps):
            if any(dep < 0 or dep >= index for dep in step["depends_on"]):
                raise ValueError(f"模板 {task_type} 第 {index + 1} 步存在非法依赖")
            result.append(SubTask(
                subtask_id=ids[index], name=step["name"], description=step["description"],
                order=index + 1, dependencies=[ids[d] for d in step["depends_on"]],
                required_capability=step["capability"], preconditions=list(step["preconditions"]),
                completion_criteria=list(step["criteria"])))
        self.validate_subtasks(result)
        return result

    @staticmethod
    def validate_subtasks(subtasks: list[SubTask]) -> None:
        ids = [item.subtask_id for item in subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("子任务 ID 不得重复")
        order_by_id = {item.subtask_id: item.order for item in subtasks}
        for item in subtasks:
            if not item.name or not item.description or not item.required_capability:
                raise ValueError("子任务名称、描述和能力标签不能为空")
            if item.order < 1:
                raise ValueError("子任务 order 必须为正整数")
            for dep in item.dependencies:
                if dep not in order_by_id:
                    raise ValueError(f"依赖不存在：{dep}")
                if order_by_id[dep] >= item.order:
                    raise ValueError("子任务只能依赖顺序更早的任务")

    def create_task(self, goal: str | dict[str, Any]) -> Task:
        info = self.recognize_goal(goal)
        subtasks = self.decompose_task(info["task_type"])
        task = Task(
            task_id=f"TASK-{uuid4().hex[:8]}", task_type=info["task_type"],
            priority=info["priority"], preconditions=info["preconditions"],
            dependencies=info["dependencies"], status="created",
            completion_criteria=info["completion_criteria"], description=info["description"],
            constraints=info["constraints"], subtasks=subtasks)
        self.validate_task(task)
        return task

    @classmethod
    def validate_task(cls, task: Task) -> None:
        if not task.task_id or not task.task_type:
            raise ValueError("task_id 和 task_type 不能为空")
        if isinstance(task.priority, bool) or not isinstance(task.priority, int) or not 1 <= task.priority <= 5:
            raise ValueError("priority 必须是 1 到 5 之间的整数")
        if task.status not in {"created", "pending", "running", "completed", "failed", "blocked"}:
            raise ValueError(f"不支持的任务状态：{task.status}")
        if not task.subtasks:
            raise ValueError("任务至少包含一个子任务")
        cls.validate_subtasks(task.subtasks)

    @staticmethod
    def task_to_dict(task: Task) -> dict[str, Any]:
        return asdict(task)
